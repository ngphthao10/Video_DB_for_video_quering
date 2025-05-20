import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import time
import threading
from typing import Dict, Any
import os

from app.database_manager import DatabaseManager
from utils.visualization import resize_image_to_fit, cv2_to_pil, pil_to_tkinter

class RealTimeVideoPlayer(ttk.Frame):    
    def __init__(self, parent: ttk.Frame, config: Dict[str, Any]):
        super().__init__(parent)
        self.config = config
        
        self.video_path = None
        self.video_cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.playing = False
        self.play_thread = None
        self.is_realtime = False
        
        self.yolo_model = None
        self._init_yolo_model()
        
        self._setup_ui()
    
    def _init_yolo_model(self):
        try:
            from ultralytics import YOLO
            
            yolo_config = self.config.get('yolo', {})
            model_name = yolo_config.get('model', 'yolov8s')
            
            print(f"Loading YOLO model: {model_name}")
            self.yolo_model = YOLO(model_name)
            
            self._init_class_mapping()
            
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None
    
    def _init_class_mapping(self):
        yolo_config = self.config.get('yolo', {})
        config_mapping = yolo_config.get('class_mapping', {})
        
        self.class_mapping = {
            0: 0,   # person -> pedestrian
            1: 2,   # bicycle -> bicycle
            2: 3,   # car -> car
            3: 9,   # motorcycle -> motor
            5: 8,   # bus -> bus
            7: 5,   # truck -> truck
        }
        
        for yolo_idx_str, visdrone_idx in config_mapping.items():
            try:
                yolo_idx = int(yolo_idx_str)
                self.class_mapping[yolo_idx] = visdrone_idx
            except (ValueError, TypeError):
                continue
        
        self.default_class_id = 3
    
    def _setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        image_frame = ttk.LabelFrame(self, text="Video View with YOLO Detection")
        image_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(image_frame, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        file_frame = ttk.Frame(controls_frame)
        file_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.video_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.video_path_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse", command=self._browse_video).pack(side=tk.LEFT, padx=5)
        
        playback_frame = ttk.Frame(controls_frame)
        playback_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        settings_frame = ttk.LabelFrame(playback_frame, text="Detection Settings")
        settings_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Label(settings_frame, text="Confidence:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.confidence_var = tk.DoubleVar(value=self.config.get('yolo', {}).get('confidence_threshold', 0.5))
        confidence_scale = ttk.Scale(
            settings_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.confidence_var,
            command=lambda val: self.conf_label_var.set(f"{float(val):.2f}")
        )
        confidence_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        
        self.conf_label_var = tk.StringVar(value=f"{self.confidence_var.get():.2f}")
        ttk.Label(settings_frame, textvariable=self.conf_label_var, width=5).grid(row=0, column=2, padx=5)
        
        buttons_frame = ttk.Frame(playback_frame)
        buttons_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)
        
        self.play_button = ttk.Button(buttons_frame, text="Play", width=10, command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(buttons_frame, text="Stop", width=10, command=self._stop_video).pack(side=tk.LEFT, padx=5)
        
        status_frame = ttk.Frame(playback_frame)
        status_frame.pack(side=tk.RIGHT, padx=5, fill=tk.Y)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.RIGHT, padx=5)
    
    def _browse_video(self):
        video_file = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )
        if video_file:
            self.video_path_var.set(video_file)
            self._load_video(video_file)
    
    def _load_video(self, video_path):
        self._stop_video()
        
        try:
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            self.fps = int(self.video_cap.get(cv2.CAP_PROP_FPS))
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.video_path = video_path
            
            self.status_var.set(f"Loaded: {os.path.basename(video_path)} ({width}x{height}, {self.fps} FPS)")
            
            self._show_frame(0)
            
            self.play_button.config(text="Play")
            self.playing = False
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.video_cap = None
            self.video_path = None
    
    def _show_frame(self, frame_number=None):
        if not self.video_cap:
            return
        
        if frame_number is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = self.video_cap.read()
        if not ret:
            self._stop_video()
            return
        
        if self.yolo_model:
            try:
                confidence = self.confidence_var.get()
                
                results = self.yolo_model(frame, conf=confidence)
                
                if len(results) > 0:
                    result = results[0]
                    frame = self._draw_detection_boxes(frame, result)
            except Exception as e:
                print(f"Detection error: {e}")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            frame = resize_image_to_fit(frame, canvas_width, canvas_height)
        
        pil_image = cv2_to_pil(frame)
        self.photo = pil_to_tkinter(pil_image)
        
        self.canvas.config(width=pil_image.width, height=pil_image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        self.current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def _draw_detection_boxes(self, frame, result):
        class_colors = self.config.get('class_colors', [])
        if not class_colors:
            class_colors = [
                (255, 0, 0),      # Red
                (255, 128, 0),    # Orange
                (255, 255, 0),    # Yellow
                (0, 255, 0),      # Green
                (0, 255, 255),    # Cyan
                (0, 0, 255),      # Blue
                (255, 0, 255),    # Magenta
                (128, 0, 255),    # Purple
                (128, 128, 128),  # Gray
                (255, 255, 255)   # White
            ]
        
        annotated_frame = frame.copy()
        class_names = self.config.get('classes', [])
        
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            for box in boxes:
                box_data = box.data.cpu().numpy()[0]
                x1, y1, x2, y2 = map(int, box_data[0:4])
                confidence = box_data[4]
                
                class_id = int(box.cls.cpu().numpy()[0])
                
                visdrone_class_id = self.class_mapping.get(class_id, self.default_class_id)
                visdrone_class_name = class_names[visdrone_class_id] if visdrone_class_id < len(class_names) else "unknown"
                
                color = class_colors[visdrone_class_id % len(class_colors)]
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{visdrone_class_name} {confidence:.2f}"
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def _toggle_play(self):
        if not self.video_cap:
            if self.video_path_var.get():
                self._load_video(self.video_path_var.get())
            else:
                return
        
        if self.playing:
            self.playing = False
            self.play_button.config(text="Play")
        else:
            self.playing = True
            self.play_button.config(text="Pause")
            
            if not self.play_thread or not self.play_thread.is_alive():
                self.play_thread = threading.Thread(target=self._play_video)
                self.play_thread.daemon = True
                self.play_thread.start()
    
    def _play_video(self):
        start_time = time.time()
        frames_displayed = 0
        
        while self.playing and self.video_cap:
            frame_time = 1.0 / self.fps
            
            self._show_frame()
            
            frames_displayed += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:  
                actual_fps = frames_displayed / elapsed
                self.status_var.set(f"FPS: {actual_fps:.1f} | Frame: {self.current_frame}/{self.total_frames}")
                frames_displayed = 0
                start_time = time.time()
            
            elapsed_frame_time = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed_frame_time)
            time.sleep(sleep_time)
            
            if self.current_frame >= self.total_frames - 1:
                self.playing = False
                self.play_button.config(text="Play")
                break
        
    def _stop_video(self):
        self.playing = False
        
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
        
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        
        self.play_button.config(text="Play")
        self.status_var.set("Stopped")
    
    def set_video_path(self, video_path):
        if os.path.exists(video_path):
            self.video_path_var.set(video_path)
            self._load_video(video_path)