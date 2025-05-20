import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from PIL import Image, ImageTk

def draw_bounding_boxes(image: np.ndarray, annotations: List[Dict[str, Any]], class_colors: List[Tuple[int, int, int]]) -> np.ndarray:
    result = image.copy()
    
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        class_id = annotation["class_id"]
        class_name = annotation["class_name"]
        
        color = class_colors[class_id % len(class_colors)]
        
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        
        text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result, (x, y-text_size[1]-5), (x+text_size[0], y), color, -1)
        
        cv2.putText(result, class_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result

def resize_image_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    
    aspect = width / height
    
    if width > height:
        new_width = min(width, max_width)
        new_height = int(new_width / aspect)
        if new_height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect)
    else:
        new_height = min(height, max_height)
        new_width = int(new_height * aspect)
        if new_width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized

def cv2_to_pil(image: np.ndarray) -> Image.Image:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image

def pil_to_tkinter(pil_image: Image.Image) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(image=pil_image)