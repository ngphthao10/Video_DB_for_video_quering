import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any

from bson.objectid import ObjectId
from app.database_manager import DatabaseManager
from app.gui.video_player import VideoPlayer

class QueryPanel(ttk.Frame):
    
    def __init__(self, parent: ttk.Frame, db_manager: DatabaseManager, config: Dict[str, Any]):
        super().__init__(parent)
        self.db_manager = db_manager
        self.config = config
        
        self.current_video_id = None
        self.video_player = None
        self.current_results = {}
        self._setup_ui()
    
    def _setup_ui(self):
        query_frame = ttk.LabelFrame(self, text="Query Frames")
        query_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(query_frame, text="Object Class:").pack(anchor=tk.W, padx=5, pady=2)
        
        self.class_var = tk.StringVar(value="All")
        class_names = ["All"] + [name.capitalize() for name in self.config.get('classes', [])]
        class_combo = ttk.Combobox(query_frame, textvariable=self.class_var, values=class_names)
        class_combo.pack(fill=tk.X, padx=5, pady=2)
        
        range_frame = ttk.Frame(query_frame)
        range_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(range_frame, text="From:").pack(side=tk.LEFT)
        self.start_frame_var = tk.StringVar(value="0")
        ttk.Entry(range_frame, textvariable=self.start_frame_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(range_frame, text="To:").pack(side=tk.LEFT, padx=5)
        self.end_frame_var = tk.StringVar(value="100")
        ttk.Entry(range_frame, textvariable=self.end_frame_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(query_frame, text="Run Query", command=self._run_query).pack(fill=tk.X, padx=5, pady=5)
        
        results_frame = ttk.LabelFrame(self, text="Query Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_tree = ttk.Treeview(
            results_frame, 
            columns=("frame", "objects"),
            show="headings"
        )
        self.results_tree.heading("frame", text="Frame")
        self.results_tree.heading("objects", text="Objects")
        self.results_tree.column("frame", width=100)
        self.results_tree.column("objects", width=300)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_tree.bind("<Double-1>", self._on_result_select)
    
    def set_video_id(self, video_id: ObjectId):
        self.current_video_id = video_id
        self._clear_results()
        
        video_info = self.db_manager.get_video_info(video_id)
        if video_info:
            self.end_frame_var.set(str(video_info["total_frames"] - 1))
    
    def set_video_player(self, video_player: VideoPlayer):
        self.video_player = video_player
    
    def _clear_results(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.current_results = {}
    
    def _run_query(self):
        if not self.current_video_id:
            messagebox.showerror("Error", "No video loaded")
            return
        
        try:
            start_frame = int(self.start_frame_var.get())
            end_frame = int(self.end_frame_var.get())
            
            video_info = self.db_manager.get_video_info(self.current_video_id)
            if not video_info:
                messagebox.showerror("Error", "Video not found")
                return
            
            total_frames = video_info["total_frames"]
            if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
                messagebox.showerror(
                    "Error", 
                    f"Invalid frame range. Must be between 0 and {total_frames-1}"
                )
                return
            
            class_str = self.class_var.get()
            class_id = None
            if class_str != "All":
                class_names = [name.capitalize() for name in self.config.get('classes', [])]
                if class_str in class_names:
                    class_id = class_names.index(class_str)
            
            self._clear_results()
            
            self.current_results = self.db_manager.query_frame_range(
                self.current_video_id, 
                start_frame, 
                end_frame, 
                class_id
            )
            
            self._update_results_tree()
        except Exception as e:
            messagebox.showerror("Error", f"Query failed: {str(e)}")
    
    def _update_results_tree(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if not self.current_results:
            self.results_tree.insert("", tk.END, values=("No results", ""))
            return
        
        total_objects = sum(len(annotations) for annotations in self.current_results.values())
        self.results_tree.insert(
            "", 
            tk.END, 
            values=(f"Found {len(self.current_results)} frames", f"{total_objects} objects")
        )
        
        for frame_number in sorted(self.current_results.keys()):
            annotations = self.current_results[frame_number]
            
            class_counts = {}
            for annotation in annotations:
                class_name = annotation["class_name"].capitalize()
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
            
            summary = ", ".join([f"{count} {class_name}" for class_name, count in class_counts.items()])
            
            self.results_tree.insert("", tk.END, values=(f"Frame {frame_number}", summary))
    
    def _on_result_select(self, event):
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = self.results_tree.item(selection[0])
        values = item["values"]
        
        if values and values[0].startswith("Frame "):
            try:
                frame_str = values[0].replace("Frame ", "")
                frame_number = int(frame_str)
                
                if self.video_player:
                    self.video_player.jump_to_frame(frame_number)
            except (ValueError, IndexError):
                pass