import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any

from app.database_manager import DatabaseManager
from app.gui.import_dialog import ImportDialog
from app.gui.video_import_dialog import VideoImportDialog 
from app.gui.video_player import VideoPlayer
from app.gui.query_panel import QueryPanel

class MainWindow:
    
    def __init__(self, root: tk.Tk, config: Dict[str, Any]):
        self.root = root
        self.config = config
        
        self.db_manager = DatabaseManager(config)
        self._setup_ui()        
        self.current_video_id = None
    
    def _setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)
        
        self._create_menu()
        
        # Left panel (controls)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        video_frame = ttk.LabelFrame(left_frame, text="Video Selection")
        video_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(video_frame, text="Select Video:").pack(anchor=tk.W, padx=5, pady=2)
        self.video_var = tk.StringVar()
        self.video_combo = ttk.Combobox(video_frame, textvariable=self.video_var)
        self.video_combo.pack(fill=tk.X, padx=5, pady=2)
        self.video_combo.bind("<<ComboboxSelected>>", self._on_video_select)
        
        video_button_frame = ttk.Frame(video_frame)
        video_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(video_button_frame, text="Refresh Videos", command=self._refresh_videos).pack(side=tk.LEFT, padx=2, pady=2)

        ttk.Button(
            video_button_frame, 
            text="Real-time Detection", 
            command=self._show_realtime_player
        ).pack(side=tk.RIGHT, padx=2, pady=2)
    
        self.query_panel = QueryPanel(left_frame, self.db_manager, self.config)
        self.query_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel (video player)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.video_player = VideoPlayer(right_frame, self.db_manager, self.config)
        self.video_player.pack(fill=tk.BOTH, expand=True)
        
        self.query_panel.set_video_player(self.video_player)
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky="ew")
        
        self._refresh_videos()
    
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Import Dataset", command=self._show_import_dialog)
                
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="YOLO Settings", command=self._show_yolo_settings)
        
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        self.root.config(menu=menubar)
    
    def _refresh_videos(self):
        try:
            videos = self.db_manager.get_all_videos()
            
            self.video_combo['values'] = [f"{video['name']} ({video['total_frames']} frames)" for video in videos]
            
            if videos and not self.video_var.get():
                self.video_var.set(f"{videos[0]['name']} ({videos[0]['total_frames']} frames)")
                self._on_video_select(None)
            
            self.status_var.set(f"Found {len(videos)} videos in database")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh videos: {str(e)}")
            self.status_var.set("Failed to refresh videos")
    
    def _on_video_select(self, event):
        video_selection = self.video_var.get()
        if video_selection:
            if "(" in video_selection:
                video_name = video_selection.split("(")[0].strip()
            else:
                video_name = video_selection
            
            video = self.db_manager.get_video_by_name(video_name)
            if video:
                self.current_video_id = video["_id"]
                
                self.video_player.load_video(self.current_video_id)
                
                self.query_panel.set_video_id(self.current_video_id)
                
                self.status_var.set(f"Loaded video: {video['name']} ({video['total_frames']} frames)")
            else:
                messagebox.showerror("Error", f"Video '{video_name}' not found in database")
    
    def _show_import_dialog(self):
        dialog = ImportDialog(self.root, self.db_manager, self.config)
        
        self.root.wait_window(dialog)
        self._refresh_videos()
    
    def _show_video_import_dialog(self):
        dialog = VideoImportDialog(self.root, self.db_manager, self.config)
        
        self.root.wait_window(dialog)
        self._refresh_videos()
    
    def _show_yolo_settings(self):
        YoloSettingsDialog(self.root, self.config)

    def _show_realtime_player(self):
        from app.realtime_video_window import RealTimeVideoWindow
        
        video_window = RealTimeVideoWindow(self.root, self.config)
        
        if self.current_video_id:
            video_info = self.db_manager.get_video_info(self.current_video_id)
            if video_info and 'file_path' in video_info:
                video_window.set_video_path(video_info['file_path'])

class YoloSettingsDialog(tk.Toplevel):
    
    def __init__(self, parent: tk.Tk, config: Dict[str, Any]):
        super().__init__(parent)
        self.parent = parent
        self.config = config
        
        self.title("YOLO Settings")
        self.geometry("500x400")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (parent.winfo_width() - width) // 2 + parent.winfo_x()
        y = (parent.winfo_height() - height) // 2 + parent.winfo_y()
        self.geometry(f"+{x}+{y}")
        
        self._setup_ui()
    
    def _setup_ui(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        yolo_frame = ttk.LabelFrame(main_frame, text="YOLO Detection Settings")
        yolo_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        yolo_config = self.config.get('yolo', {})
        ttk.Label(yolo_frame, text="Default Model:").grid(row=0, column=0, sticky=tk.W, pady=10, padx=10)
        
        yolo_models = ["yolov8s", "yolov8n", "yolov8m", "yolov5n", "yolov5m"]
        self.model_var = tk.StringVar(value=yolo_config.get('model', 'yolov5s'))
        model_combo = ttk.Combobox(yolo_frame, textvariable=self.model_var, values=yolo_models, width=15)
        model_combo.grid(row=0, column=1, sticky=tk.W, pady=10, padx=10)
        
        ttk.Label(yolo_frame, text="Confidence Threshold:").grid(row=1, column=0, sticky=tk.W, pady=10, padx=10)
        
        self.confidence_var = tk.DoubleVar(value=yolo_config.get('confidence_threshold', 0.5))
        confidence_scale = ttk.Scale(
            yolo_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.confidence_var,
            command=lambda val: self.conf_label_var.set(f"{float(val):.2f}")
        )
        confidence_scale.grid(row=1, column=1, sticky=tk.EW, pady=10, padx=10)
        
        self.conf_label_var = tk.StringVar(value=f"{self.confidence_var.get():.2f}")
        ttk.Label(yolo_frame, textvariable=self.conf_label_var, width=5).grid(row=1, column=2, sticky=tk.W, padx=10)
        
        ttk.Label(yolo_frame, text="IoU Threshold:").grid(row=2, column=0, sticky=tk.W, pady=10, padx=10)
        
        self.iou_var = tk.DoubleVar(value=yolo_config.get('iou_threshold', 0.45))
        iou_scale = ttk.Scale(
            yolo_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.iou_var,
            command=lambda val: self.iou_label_var.set(f"{float(val):.2f}")
        )
        iou_scale.grid(row=2, column=1, sticky=tk.EW, pady=10, padx=10)
        
        self.iou_label_var = tk.StringVar(value=f"{self.iou_var.get():.2f}")
        ttk.Label(yolo_frame, textvariable=self.iou_label_var, width=5).grid(row=2, column=2, sticky=tk.W, padx=10)
        
        ttk.Label(yolo_frame, text="Frame Skip:").grid(row=3, column=0, sticky=tk.W, pady=10, padx=10)
        
        self.frame_skip_var = tk.IntVar(value=yolo_config.get('frame_skip', 1))
        frame_skip_entry = ttk.Spinbox(
            yolo_frame,
            from_=1,
            to=30,
            textvariable=self.frame_skip_var,
            width=5
        )
        frame_skip_entry.grid(row=3, column=1, sticky=tk.W, pady=10, padx=10)
        
        ttk.Label(yolo_frame, text="Process every Nth frame").grid(row=3, column=2, sticky=tk.W, padx=10)
        
        mapping_frame = ttk.LabelFrame(main_frame, text="YOLO to VisDrone Class Mapping")
        mapping_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(mapping_frame, text="YOLO maps COCO classes to VisDrone classes automatically.").pack(anchor=tk.W, padx=10, pady=5)
        ttk.Label(mapping_frame, text="Examples: person -> pedestrian, bicycle -> bicycle, etc.").pack(anchor=tk.W, padx=10, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Save", command=self._save_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _save_settings(self):
        if 'yolo' not in self.config:
            self.config['yolo'] = {}
            
        self.config['yolo']['model'] = self.model_var.get()
        self.config['yolo']['confidence_threshold'] = self.confidence_var.get()
        self.config['yolo']['iou_threshold'] = self.iou_var.get()
        self.config['yolo']['frame_skip'] = self.frame_skip_var.get()
        
        from utils.config import save_config
        save_config(self.config)
        
        self.destroy()