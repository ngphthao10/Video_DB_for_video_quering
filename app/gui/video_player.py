import tkinter as tk
from tkinter import ttk
import cv2
import time
import threading
from typing import Dict, Any, Callable

from bson.objectid import ObjectId
from app.database_manager import DatabaseManager
from utils.visualization import draw_bounding_boxes, resize_image_to_fit, cv2_to_pil, pil_to_tkinter

class VideoPlayer(ttk.Frame):
    
    def __init__(self, parent: ttk.Frame, db_manager: DatabaseManager, config: Dict[str, Any]):
        super().__init__(parent)
        self.db_manager = db_manager
        self.config = config
        
        self.current_video_id = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.playing = False
        self.play_thread = None
        
        self.on_frame_change = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        image_frame = ttk.LabelFrame(self, text="Frame View")
        image_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(image_frame, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Frame:").pack(side=tk.LEFT, padx=5)
        self.frame_slider = ttk.Scale(
            controls_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL,
            command=self._on_slider_change
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.frame_var = tk.StringVar(value="0 / 0")
        ttk.Label(controls_frame, textvariable=self.frame_var, width=10).pack(side=tk.LEFT, padx=5)
        
        playback_frame = ttk.Frame(controls_frame)
        playback_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(playback_frame, text="⏮", width=3, command=self._first_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_frame, text="◀", width=3, command=self._prev_frame).pack(side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(playback_frame, text="▶", width=3, command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_frame, text="▶", width=3, command=self._next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_frame, text="⏭", width=3, command=self._last_frame).pack(side=tk.LEFT, padx=2)
        
        fps_frame = ttk.Frame(controls_frame)
        fps_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="30")
        fps_spinner = ttk.Spinbox(
            fps_frame, 
            from_=1, 
            to=60, 
            textvariable=self.fps_var,
            width=5,
            wrap=True,
            command=self._on_fps_change
        )
        fps_spinner.pack(side=tk.LEFT, padx=2)
        fps_spinner.bind("<Return>", lambda e: self._on_fps_change())
    
    def load_video(self, video_id: ObjectId):
        if self.playing:
            self._toggle_play()
        
        print(f"Loading video with ID: {video_id}")
        
        video_info = self.db_manager.get_video_info(video_id)
        if not video_info:
            print(f"Error: Video with ID {video_id} not found")
            return
        
        print(f"Found video: {video_info['name']} with {video_info['total_frames']} frames")
        
        self.current_video_id = video_id
        self.total_frames = video_info["total_frames"]
        self.fps = video_info["fps"]
        
        self.frame_slider.configure(to=self.total_frames-1)
        self.fps_var.set(str(self.fps))

        self.current_frame = 0
        self._load_frame(0)
    
    def _load_frame(self, frame_index: int):
        if not self.current_video_id:
            print("No video loaded")
            return
        
        try:
            frames = list(self.db_manager.frames.find(
                {"video_id": self.current_video_id}
            ).sort("frame_number", 1))
            
            if not frames:
                print(f"No frames found for video {self.current_video_id}")
                return
            
            frame_index = max(0, min(frame_index, len(frames) - 1))
            
            frame = frames[frame_index]
            frame_number = frame["frame_number"]
            
            image_path = frame["image_path"]
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return
            
            annotations = self.db_manager.get_frame_annotations(frame["_id"])
            
            image = draw_bounding_boxes(image, annotations, self.config["class_colors"])
            
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                image = resize_image_to_fit(image, canvas_width, canvas_height)
            
            pil_image = cv2_to_pil(image)
            self.photo = pil_to_tkinter(pil_image)
            
            self.canvas.config(width=pil_image.width, height=pil_image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            self.current_frame = frame_index
            self.frame_var.set(f"{frame_index} / {len(frames) - 1} (#{frame_number})")
            self.frame_slider.set(frame_index)
            
            if self.on_frame_change:
                self.on_frame_change(frame_index)
        except Exception as e:
            print(f"Error loading frame: {e}")
            import traceback
            traceback.print_exc()
    
    def set_on_frame_change(self, callback: Callable[[int], None]):
        self.on_frame_change = callback
    
    def _on_slider_change(self, value):
        frame = int(float(value))
        if frame != self.current_frame:
            self._load_frame(frame)
    
    def _on_fps_change(self):
        try:
            fps = int(self.fps_var.get())
            if fps > 0:
                self.fps = fps
        except ValueError:
            self.fps_var.set(str(self.fps))
    
    def _first_frame(self):
        self._load_frame(0)
    
    def _prev_frame(self):
        self._load_frame(max(0, self.current_frame - 1))
    
    def _next_frame(self):
        self._load_frame(min(self.total_frames - 1, self.current_frame + 1))
    
    def _last_frame(self):
        self._load_frame(self.total_frames - 1)
    
    def _toggle_play(self):
        if self.playing:
            self.playing = False
            self.play_button.configure(text="▶")
            if self.play_thread and self.play_thread.is_alive():
                self.play_thread.join()
        else:
            self.playing = True
            self.play_button.configure(text="⏸")
            self.play_thread = threading.Thread(target=self._play_video)
            self.play_thread.daemon = True
            self.play_thread.start()
    
    def _play_video(self):
        while self.playing and self.current_frame < self.total_frames - 1:
            start_time = time.time()
            
            next_frame = self.current_frame + 1
            
            self.after(0, lambda f=next_frame: self._load_frame(f))
            
            frame_time = 1.0 / self.fps
            elapsed = time.time() - start_time
            delay = max(0, frame_time - elapsed)
            
            time.sleep(delay)
            
            if not self.playing:
                break
        
        if self.current_frame >= self.total_frames - 1:
            self.playing = False
            self.after(0, lambda: self.play_button.configure(text="▶"))
    
    def jump_to_frame(self, frame_number: int):
        self._load_frame(frame_number)