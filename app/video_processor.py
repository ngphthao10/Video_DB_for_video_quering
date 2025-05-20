import os
import cv2
import numpy as np
import tempfile
import threading
import shutil
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import time

from bson.objectid import ObjectId
from app.database_manager import DatabaseManager

class VideoProcessor:
    """Processes video files using YOLOv5 for object detection"""
    
    def __init__(self, db_manager: DatabaseManager, config: Dict[str, Any]):
        """
        Initialize video processor
        
        Args:
            db_manager: Database manager
            config: Configuration dictionary
        """
        self.db_manager = db_manager
        self.config = config
        self.yolo_model = None
        self.temp_dir = None
        self.stop_processing = False
        
        # Initialize YOLO model
        self._init_yolo_model()
    
    def _init_yolo_model(self):
        """Initialize YOLOv5 model from Ultralytics"""
        try:
            # Import here to not require ultralytics for the entire application
            from ultralytics import YOLO
            
            # Get model configuration
            yolo_config = self.config.get('yolo', {})
            model_name = yolo_config.get('model', 'yolov5s')
            
            # Fix for model name format - use full path or correct format
            # YOLOv5 requires a specific format or pre-built model name
            if model_name in ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']:
                # Convert to correct format for Ultralytics YOLO
                model_name = f"{model_name}.pt"
            
            print(f"Loading YOLO model: {model_name}")
            self.yolo_model = YOLO(model_name)
            
            # Map YOLO COCO classes to VisDrone classes if needed
            self._map_yolo_to_visdrone_classes()
            
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None
    
    def _map_yolo_to_visdrone_classes(self):
        """
        Create mapping between YOLO COCO classes and VisDrone classes
        
        YOLO COCO classes include: person, bicycle, car, motorcycle, bus, truck, etc.
        VisDrone classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
        """
        # Get mapping from config if available
        yolo_config = self.config.get('yolo', {})
        config_mapping = yolo_config.get('class_mapping', {})
        
        # Create mapping dictionary with defaults
        self.class_mapping = {
            0: 0,   # person -> pedestrian
            1: 2,   # bicycle -> bicycle
            2: 3,   # car -> car
            3: 9,   # motorcycle -> motor
            5: 8,   # bus -> bus
            7: 5,   # truck -> truck
            # Add more mappings as needed
        }
        
        # Override with config values if present
        for yolo_idx_str, visdrone_idx in config_mapping.items():
            try:
                yolo_idx = int(yolo_idx_str)
                self.class_mapping[yolo_idx] = visdrone_idx
            except (ValueError, TypeError):
                continue
        
        # Default mapping for unmapped classes
        self.default_class_id = 3  # Default to car
    
    def import_video(self, video_path: str, callback: Optional[callable] = None) -> ObjectId:
        """
        Import video file, extract frames, perform detection, and store in database
        
        Args:
            video_path: Path to video file
            callback: Optional callback function for progress updates
            
        Returns:
            MongoDB ID of the imported video
        """
        if self.yolo_model is None:
            raise RuntimeError("YOLO model is not initialized")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.temp_dir}")
        
        try:
            # Reset stop flag
            self.stop_processing = False
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames, {duration:.2f} seconds")
            
            # Create video document
            video_name = os.path.basename(video_path)
            video_data = {
                "name": video_name,
                "source_type": "imported_video",
                "total_frames": total_frames,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "duration": duration,
                "file_path": video_path,
                "created_at": time.time()
            }
            
            # Insert video into database
            video_id = self.db_manager.import_video(video_data)
            
            # Process frames in parallel
            yolo_config = self.config.get('yolo', {})
            frame_skip = yolo_config.get('frame_skip', 1)
            max_workers = self.config.get('video_import', {}).get('max_workers', 4)
            
            # Process frames
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                frame_idx = 0
                
                while True:
                    if self.stop_processing:
                        print("Stopping video processing")
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every Nth frame
                    if frame_idx % frame_skip == 0:
                        # Save frame to temp directory
                        frame_path = os.path.join(self.temp_dir, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        
                        # Submit frame for processing
                        future = executor.submit(
                            self._process_frame,
                            frame,
                            frame_path,
                            frame_idx,
                            video_id,
                            fps
                        )
                        futures.append(future)
                        
                        # Update progress
                        progress = (frame_idx + 1) / total_frames * 100
                        if callback:
                            callback(progress, f"Processing frame {frame_idx + 1}/{total_frames}")
                    
                    frame_idx += 1
                
                # Wait for all tasks to complete
                for future in futures:
                    future.result()
            
            # Create segment trees for efficient querying
            self._build_segment_trees(video_id)
            
            return video_id
        
        finally:
            # Clean up
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"Removed temporary directory: {self.temp_dir}")
    
    def _process_frame(self, frame: np.ndarray, frame_path: str, frame_idx: int, 
                      video_id: ObjectId, fps: float) -> None:
        """
        Process a single frame with YOLOv5
        
        Args:
            frame: Frame image
            frame_path: Path to saved frame
            frame_idx: Frame index
            video_id: MongoDB ID of the video
            fps: Frames per second
        """
        try:
            # Store frame in database
            frame_data = {
                "video_id": video_id,
                "frame_number": frame_idx,
                "image_path": frame_path,
                "timestamp": frame_idx / fps
            }
            
            frame_id = self.db_manager.store_frame(frame_data)
            
            # Run detection on frame
            results = self.yolo_model(frame)
            
            # Convert detections to annotations
            annotations = []
            
            # Process detections
            if len(results) > 0:
                # Get the first result - single image result
                result = results[0]
                
                # Convert to database format
                yolo_config = self.config.get('yolo', {})
                conf_threshold = yolo_config.get('confidence_threshold', 0.5)
                
                # For YOLOv8 compatibility (using different result format)
                if hasattr(result, 'boxes'):
                    boxes = result.boxes
                    for box in boxes:
                        # Convert to numpy for easier handling
                        box_data = box.data.cpu().numpy()[0]
                        confidence = box_data[4]
                        
                        if confidence >= conf_threshold:
                            # Get coordinates
                            x1, y1, x2, y2 = box_data[0:4]
                            # Convert to x, y, width, height format
                            x = int(x1)
                            y = int(y1)
                            w = int(x2 - x1)
                            h = int(y2 - y1)
                            
                            # Get class info
                            class_id = int(box.cls.cpu().numpy()[0])
                            
                            # Map YOLO class to VisDrone class
                            visdrone_class_id = self._map_class_id(class_id)
                            class_names = self.config.get('classes', [])
                            class_name = class_names[visdrone_class_id] if visdrone_class_id < len(class_names) else "unknown"
                            
                            # Create annotation
                            annotation = {
                                "frame_id": frame_id,
                                "bbox": [x, y, w, h],
                                "class_id": visdrone_class_id,
                                "class_name": class_name,
                                "confidence": float(confidence)
                            }
                            
                            annotations.append(annotation)
            
            # Store annotations in database
            if annotations:
                self.db_manager.store_annotations(annotations)
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
    
    def _map_class_id(self, yolo_class_id: int) -> int:
        """
        Map YOLO COCO class ID to VisDrone class ID
        
        Args:
            yolo_class_id: YOLO class ID
            
        Returns:
            VisDrone class ID
        """
        return self.class_mapping.get(yolo_class_id, self.default_class_id)
    
    def _build_segment_trees(self, video_id: ObjectId) -> None:
        """
        Build segment trees for the imported video
        
        Args:
            video_id: MongoDB ID of the video
        """
        print(f"Building segment trees for video {video_id}")
        
        # Get all frames for this video
        frames = list(self.db_manager.frames.find({"video_id": video_id}).sort("frame_number", 1))
        
        if not frames:
            print("No frames found, skipping segment tree creation")
            return
        
        # Get max frame number
        max_frame_number = max(frame["frame_number"] for frame in frames) + 1
        
        # Create a dictionary mapping frame numbers to annotations
        frame_annotations = {}
        
        for frame in frames:
            frame_id = frame["_id"]
            frame_number = frame["frame_number"]
            
            # Get annotations for this frame
            annotations = self.db_manager.get_frame_annotations(frame_id)
            
            # Add to frame_annotations
            frame_annotations[frame_number] = [
                {"_id": annotation["_id"], "class_id": annotation["class_id"]}
                for annotation in annotations
            ]
        
        # Build segment trees using the database manager method
        self.db_manager._build_segment_trees(video_id, frame_annotations, max_frame_number)
        
        print("Segment trees built successfully")
    
    def stop(self) -> None:
        """Stop ongoing processing"""
        self.stop_processing = True