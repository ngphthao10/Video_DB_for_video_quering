import tkinter as tk
from tkinter import ttk

from app.realtime_video_player import RealTimeVideoPlayer

class RealTimeVideoWindow(tk.Toplevel):    
    def __init__(self, parent: tk.Tk, config: dict):
        super().__init__(parent)
        self.parent = parent
        self.config = config
        
        self.title("Video Player with Real-time Detection")
        self.geometry("1000x700")
        self.minsize(800, 600)
        
        self.transient(parent)
        
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (parent.winfo_width() - width) // 2 + parent.winfo_x()
        y = (parent.winfo_height() - height) // 2 + parent.winfo_y()
        self.geometry(f"+{x}+{y}")
        
        self._setup_ui()
    
    def _setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.video_player = RealTimeVideoPlayer(self, self.config)
        self.video_player.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        bottom_frame = ttk.Frame(self)
        bottom_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        ttk.Button(bottom_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=5)
    
    def set_video_path(self, video_path):
        self.video_player.set_video_path(video_path)