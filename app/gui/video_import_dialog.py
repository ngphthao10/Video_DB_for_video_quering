import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any, Optional
import threading

from app.database_manager import DatabaseManager
from app.video_processor import VideoProcessor

class VideoImportDialog(tk.Toplevel):
    
    def __init__(self, parent: tk.Tk, db_manager: DatabaseManager, config: Dict[str, Any]):
        super().__init__(parent)
        self.parent = parent
        self.db_manager = db_manager
        self.config = config
        
        self.video_processor = VideoProcessor(db_manager, config)
        
        self.title("Import Video with YOLO Detection")
        self.geometry("600x400")
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

        ttk.Label(main_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        path_frame = ttk.Frame(main_frame)
        path_frame.grid(row=0, column=1, sticky=tk.EW, pady=5)
        path_frame.columnconfigure(0, weight=1)
        
        self.video_path_var = tk.StringVar(value=self.config.get('video_import', {}).get('default_video_path', ''))
        ttk.Entry(path_frame, textvariable=self.video_path_var).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(path_frame, text="Browse", command=self._browse_video).grid(row=0, column=1)

        yolo_frame = ttk.LabelFrame(main_frame, text="YOLO Detection Settings")
        yolo_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=10)
        yolo_frame.columnconfigure(1, weight=1)

        ttk.Label(yolo_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        
        yolo_models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        self.model_var = tk.StringVar(value=self.config.get('yolo', {}).get('model', 'yolov8s'))
        model_combo = ttk.Combobox(yolo_frame, textvariable=self.model_var, values=yolo_models)
        model_combo.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=5)

        ttk.Label(yolo_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        
        self.confidence_var = tk.DoubleVar(value=self.config.get('yolo', {}).get('confidence_threshold', 0.5))
        confidence_scale = ttk.Scale(
            yolo_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.confidence_var,
            command=lambda val: self.conf_label_var.set(f"{float(val):.2f}")
        )
        confidence_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=5)
        
        self.conf_label_var = tk.StringVar(value=f"{self.confidence_var.get():.2f}")
        ttk.Label(yolo_frame, textvariable=self.conf_label_var, width=5).grid(row=1, column=2, padx=5)

        ttk.Label(yolo_frame, text="Frame Skip:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        
        self.frame_skip_var = tk.IntVar(value=self.config.get('yolo', {}).get('frame_skip', 1))
        frame_skip_entry = ttk.Spinbox(
            yolo_frame,
            from_=1,
            to=30,
            textvariable=self.frame_skip_var,
            width=5
        )
        frame_skip_entry.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)

        progress_frame = ttk.LabelFrame(main_frame, text="Import Progress")
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=10)
        progress_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var, 
            maximum=100.0
        )
        self.progress_bar.grid(row=0, column=0, sticky=tk.EW, padx=5, pady=5)

        self.status_var = tk.StringVar(value="Ready to import")
        ttk.Label(progress_frame, textvariable=self.status_var).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, sticky=tk.E, pady=10)
        
        self.import_button = ttk.Button(button_frame, text="Import", command=self._import_video)
        self.import_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self._cancel_import)
        self.cancel_button.pack(side=tk.LEFT)
        
        main_frame.columnconfigure(1, weight=1)
    
    def _browse_video(self):
        video_file = filedialog.askopenfilename(
            title="Select Video File",
            initialdir=self.video_path_var.get(),
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )
        if video_file:
            self.video_path_var.set(video_file)
    
    def _import_video(self):
        video_path = self.video_path_var.get()
        
        if not video_path:
            messagebox.showerror("Error", "Please select a video file")
            return
        
        self.config['yolo']['model'] = self.model_var.get()
        self.config['yolo']['confidence_threshold'] = self.confidence_var.get()
        self.config['yolo']['frame_skip'] = self.frame_skip_var.get()
        
        self.import_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Initializing video import...")
        self.update_idletasks()
        
        self.import_thread = threading.Thread(target=self._run_import_thread, args=(video_path,))
        self.import_thread.daemon = True
        self.import_thread.start()
    
    def _run_import_thread(self, video_path: str):
        try:
            video_id = self.video_processor.import_video(
                video_path,
                callback=self._update_progress
            )
            
            self.after(0, lambda: self._import_completed(video_id))
            
        except Exception as error:
            error_message = str(error)
            self.after(0, lambda: self._import_failed(error_message))
    
    def _update_progress(self, progress: float, status: str):
        self.after(0, lambda: self._set_progress(progress, status))
    
    def _set_progress(self, progress: float, status: str):
        self.progress_var.set(progress)
        self.status_var.set(status)
        self.update_idletasks()
    
    def _import_completed(self, video_id):
        self.progress_var.set(100)
        self.status_var.set("Import completed successfully")
        
        video_info = self.db_manager.get_video_info(video_id)
        video_name = video_info["name"] if video_info else "Unknown"
        
        messagebox.showinfo(
            "Success", 
            f"Video '{video_name}' imported successfully with YOLO detection."
        )
        
        self.destroy()
    
    def _import_failed(self, error_message: str):
        self.progress_var.set(0)
        self.status_var.set(f"Error: {error_message}")
        self.import_button.config(state=tk.NORMAL)
        
        messagebox.showerror("Error", f"Failed to import video: {error_message}")
    
    def _cancel_import(self):
        if hasattr(self, 'import_thread') and self.import_thread.is_alive():
            if hasattr(self, 'video_processor'):
                self.video_processor.stop()
            
            self.status_var.set("Canceling import...")
            self.update_idletasks()
        else:
            self.destroy()