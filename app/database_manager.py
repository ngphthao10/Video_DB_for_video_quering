import sys
import datetime
import cv2
from pathlib import Path
from typing import Dict, List, Any
from pymongo import MongoClient
from bson.objectid import ObjectId

from app.segment_tree import FrameSegmentTree

class DatabaseManager:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        db_config = config.get('mongodb', {})
        uri = db_config.get('uri', 'mongodb://localhost:27017')
        db_name = db_config.get('db_name', 'visdrone_db')
        
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.videos = self.db["videos"]
            self.frames = self.db["frames"]
            self.annotations = self.db["annotations"]
            self.segment_trees = self.db["segment_trees"]
            print(f"Connected to MongoDB at {uri}")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            sys.exit(1)

    def create_indices(self):
        self.frames.create_index([("video_id", 1), ("frame_number", 1)])
        self.annotations.create_index([("frame_id", 1)])
        self.annotations.create_index([("class_id", 1)])
        self.segment_trees.create_index([("video_id", 1), ("object_class", 1)])
        print("Database indices created")
    
    def import_visdrone_dataset(self, dataset_path: str, fps: int = 30) -> List[ObjectId]:
        dataset_path = Path(dataset_path)
        images_path = dataset_path / "images"
        annotations_path = dataset_path / "annotations"
        
        if not images_path.exists() or not annotations_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        image_files = sorted(list(images_path.glob("*.jpg")))
        total_frames = len(image_files)
        
        if total_frames == 0:
            raise ValueError("No images found in dataset")
        
        print(f"Found {total_frames} images in dataset")
        
        videos = {}
        for image_file in image_files:
            parts = image_file.stem.split('_')
            if len(parts) >= 3:
                video_id = parts[0]  
                frame_number = int(parts[-1])  
                
                if video_id not in videos:
                    videos[video_id] = []
                
                videos[video_id].append((frame_number, image_file))
        
        if not videos:
            raise ValueError("Could not parse video IDs from filenames")
        
        print(f"Found {len(videos)} videos in dataset")
        
        created_video_ids = []
        
        for video_id, frames in videos.items():
            frames.sort(key=lambda x: x[0])
            
            first_image = cv2.imread(str(frames[0][1]))
            height, width, _ = first_image.shape
            resolution = f"{width}x{height}"
            
            video_name = f"VisDrone Video {video_id}"
            
            video_data = {
                "name": video_name,
                "video_id": video_id, 
                "total_frames": len(frames),
                "resolution": resolution,
                "fps": fps,
                "duration": len(frames) / fps,
                "created_at": datetime.datetime.now()
            }
            
            existing_video = self.videos.find_one({"video_id": video_id})
            if existing_video:
                print(f"Video '{video_name}' already exists, updating...")
                mongo_video_id = existing_video["_id"]
                self.videos.update_one({"_id": mongo_video_id}, {"$set": video_data})
                
                self.frames.delete_many({"video_id": mongo_video_id})
                frame_ids = [f["_id"] for f in self.frames.find({"video_id": mongo_video_id})]
                self.annotations.delete_many({"frame_id": {"$in": frame_ids}})
            else:
                result = self.videos.insert_one(video_data)
                mongo_video_id = result.inserted_id
                print(f"Created new video document with ID: {mongo_video_id}")
            
            created_video_ids.append(mongo_video_id)
            class_names = self.config.get('classes', [])
            frame_annotations = {}
            
            for frame_number, image_file in frames:
                print(f"Processing frame {frame_number} from {image_file}")
                
                frame_data = {
                    "video_id": mongo_video_id,
                    "original_video_id": video_id,
                    "frame_number": frame_number, 
                    "image_path": str(image_file),
                    "timestamp": frame_number / fps
                }
                
                frame_result = self.frames.insert_one(frame_data)
                frame_id = frame_result.inserted_id
                
                frame_annotations[frame_number] = []
                
                annotation_file = annotations_path / f"{image_file.stem}.txt"
                if annotation_file.exists():
                    with open(annotation_file, "r") as f:
                        for line in f:
                            parts = line.strip().split(",")
                            if len(parts) >= 6:
                                x, y, w, h = map(int, parts[:4])
                                score = int(parts[4])
                                class_id = int(parts[5]) - 1  
                                
                                if score == 0:
                                    continue
                                
                                if 0 <= class_id < len(class_names):
                                    annotation_data = {
                                        "frame_id": frame_id,
                                        "bbox": [x, y, w, h],
                                        "class_id": class_id,
                                        "class_name": class_names[class_id],
                                        "confidence": 1.0  # Default confidence
                                    }
                                    
                                    annotation_result = self.annotations.insert_one(annotation_data)
                                    
                                    frame_annotations[frame_number].append({
                                        "_id": annotation_result.inserted_id,
                                        "class_id": class_id
                                    })
            
            print(f"Processed {len(frames)} frames for video {video_id}")
            
            self._build_segment_trees(mongo_video_id, frame_annotations, max(frame_annotations.keys()) + 1)
    
        return created_video_ids
  
    def _build_segment_trees(self, video_id, frame_annotations, max_frame_number):
        print("Building segment trees...")
        
        self.segment_trees.delete_many({"video_id": video_id})
        
        general_tree = FrameSegmentTree(max_frame_number)
        general_tree.build(frame_annotations)
        
        tree_data = {
            "video_id": video_id,
            "object_class": None,  
            "tree_structure": general_tree.to_dict()
        }
        self.segment_trees.insert_one(tree_data)
        
        class_names = self.config.get('classes', [])
        for class_id in range(len(class_names)):
            class_tree = FrameSegmentTree(max_frame_number, class_id)
            class_tree.build(frame_annotations)
            
            tree_data = {
                "video_id": video_id,
                "object_class": class_id,
                "tree_structure": class_tree.to_dict()
            }
            self.segment_trees.insert_one(tree_data)
        
        print("Segment trees built and stored")
    
    def get_video_info(self, video_id: ObjectId) -> Dict:
        return self.videos.find_one({"_id": video_id})
    
    def get_video_by_name(self, name: str) -> Dict:
        return self.videos.find_one({"name": name})
    
    def get_all_videos(self) -> List[Dict]:
        return list(self.videos.find())
    
    def get_frame(self, video_id: ObjectId, frame_number: int) -> Dict:
        frame = self.frames.find_one({"video_id": video_id, "frame_number": frame_number})
        
        if frame:
            return frame
        
        all_frames = list(self.frames.find({"video_id": video_id}).sort("frame_number", 1))
        
        if not all_frames:
            return None
        
        if frame_number < 0:
            return all_frames[0]
        elif frame_number >= len(all_frames):
            return all_frames[-1]
        else:
            return all_frames[frame_number]
    
    def get_frame_annotations(self, frame_id: ObjectId) -> List[Dict]:
        return list(self.annotations.find({"frame_id": frame_id}))
    
    def query_frame_range(self, video_id, start_frame, end_frame, object_class=None):
        print(f"Querying frames {start_frame}-{end_frame} for video {video_id}, class: {object_class}")
        
        tree_doc = self.segment_trees.find_one({"video_id": video_id, "object_class": object_class})
        if not tree_doc:
            print(f"No segment tree found for video {video_id}, class {object_class}")
            return {}
        
        tree = FrameSegmentTree.from_dict(tree_doc["tree_structure"])
        
        object_ids = tree.query(start_frame, end_frame)
        print(f"Found {len(object_ids)} objects in range")
        
        annotations = list(self.annotations.find({"_id": {"$in": list(object_ids)}}))
        
        result = {}
        for annotation in annotations:
            frame = self.frames.find_one({"_id": annotation["frame_id"]})
            if frame:
                frame_number = frame["frame_number"]
                if frame_number not in result:
                    result[frame_number] = []
                result[frame_number].append(annotation)
        
        return result

    def cleanup_duplicates(self):
        videos = self.videos.find()
        
        for video in videos:
            video_id = video["_id"]
            
            frames = list(self.frames.find({"video_id": video_id}))
            
            frame_groups = {}
            for frame in frames:
                frame_number = frame["frame_number"]
                if frame_number not in frame_groups:
                    frame_groups[frame_number] = []
                frame_groups[frame_number].append(frame)
            
            for frame_number, group in frame_groups.items():
                if len(group) > 1:
                    print(f"Found {len(group)} duplicates for frame {frame_number} in video {video['name']}")
                    
                    keep_id = group[0]["_id"]
                    delete_ids = [frame["_id"] for frame in group[1:]]
                    
                    self.frames.delete_many({"_id": {"$in": delete_ids}})
                    
                    for frame_id in delete_ids:
                        self.annotations.update_many(
                            {"frame_id": frame_id},
                            {"$set": {"frame_id": keep_id}}
                        )

    def import_video(self, video_data: Dict[str, Any]) -> ObjectId:
        result = self.videos.insert_one(video_data)
        return result.inserted_id
    
    def store_frame(self, frame_data: Dict[str, Any]) -> ObjectId:
        result = self.frames.insert_one(frame_data)
        return result.inserted_id
    
    def store_annotations(self, annotations: List[Dict[str, Any]]) -> List[ObjectId]:
        if not annotations:
            return []
            
        result = self.annotations.insert_many(annotations)
        return result.inserted_ids
        
    def get_videos_by_source_type(self, source_type: str) -> List[Dict]:
        return list(self.videos.find({"source_type": source_type}))

    def delete_video_and_related(self, video_id: ObjectId) -> None:
        frames = list(self.frames.find({"video_id": video_id}))
        frame_ids = [frame["_id"] for frame in frames]
        
        self.annotations.delete_many({"frame_id": {"$in": frame_ids}})
        
        self.frames.delete_many({"video_id": video_id})
        
        self.segment_trees.delete_many({"video_id": video_id})
        
        self.videos.delete_one({"_id": video_id})