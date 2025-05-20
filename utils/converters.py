from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from bson.objectid import ObjectId

def visdrone_to_mongodb_format(annotation_file: Path, frame_id: ObjectId, class_names: List[str]) -> List[Dict[str, Any]]:
    annotations = []
    
    try:
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
                        
                        annotations.append(annotation_data)
    except Exception as e:
        print(f"Error processing annotation file {annotation_file}: {e}")
    
    return annotations

def convert_box_to_yolo_format(size: Tuple[int, int], box: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    dw = 1. / size[0]
    dh = 1. / size[1]
    x, y, w, h = box
    
    x_center = (x + w / 2) * dw
    y_center = (y + h / 2) * dh
    w_norm = w * dw
    h_norm = h * dh
    
    return (x_center, y_center, w_norm, h_norm)

def yolo_to_absolute_format(size: Tuple[int, int], box: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    width, height = size
    x_center, y_center, w_norm, h_norm = box
    
    w = int(w_norm * width)
    h = int(h_norm * height)
    x = int(x_center * width - w / 2)
    y = int(y_center * height - h / 2)
    
    return (x, y, w, h)