from concurrent.futures import process
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv 
load_dotenv() 

DEFAULT_CONFIG = {
     'mongodb': {
        'uri': os.getenv("MONGODB_URL"),
        'db_name': 'visdrone_db'
    },
    'classes': [
        'pedestrian',
        'people',
        'bicycle',
        'car',
        'van',
        'truck',
        'tricycle',
        'awning-tricycle',
        'bus',
        'motor'
    ],
    'class_colors': [
        (255, 0, 0),      # pedestrian (Red)
        (255, 128, 0),    # people (Orange)
        (255, 255, 0),    # bicycle (Yellow)
        (0, 255, 0),      # car (Green)
        (0, 255, 255),    # van (Cyan)
        (0, 0, 255),      # truck (Blue)
        (255, 0, 255),    # tricycle (Magenta)
        (128, 0, 255),    # awning-tricycle (Purple)
        (128, 128, 128),  # bus (Gray)
        (255, 255, 255)   # motor (White)
    ],
    'default_fps': 30,
    'default_dataset_path': '../data',
    'yolo': {
        'model': 'yolov8s',  
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'frame_skip': 1,  
        'class_mapping': {
            '0': 0,  # person -> pedestrian
            '1': 2,  # bicycle -> bicycle
            '2': 3,  # car -> car
            '3': 9,  # motorcycle -> motor
            '5': 8,  # bus -> bus
            '7': 5   # truck -> truck
        }
    },
    'video_import': {
        'default_video_path': '../videos',
        'temp_frames_dir': 'temp_frames',
        'max_workers': 4  
    }
}

def load_config(config_file: str = 'config.json') -> Dict[str, Any]:
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            merged_config = DEFAULT_CONFIG.copy()
            for key, value in config.items():
                if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
                    
            return merged_config
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration")
            return DEFAULT_CONFIG
    else:
        print("Config file not found, using default configuration")
        return DEFAULT_CONFIG

def save_config(config: Dict[str, Any], config_file: str = 'config.json') -> None:
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_file}")
    except Exception as e:
        print(f"Error saving config file: {e}")