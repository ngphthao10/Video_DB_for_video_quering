import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any

from app.database_manager import DatabaseManager

class ImportDialog(tk.Toplevel):
    
    def __init__(self, parent: tk.Tk, db_manager: DatabaseManager, config: Dict[str, Any]):
        super().__init__(parent)
        self.parent = parent
        self.db_manager = db_manager
        self.config = config
        
        # Set dialog properties
        self.title("Import VisDrone Dataset")
        self.geometry("500x300")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Center dialog
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (parent.winfo_width() - width) // 2 + parent.winfo_x()
        y = (parent.winfo_height() - height) // 2 + parent.winfo_y()
        self.geometry(f"+{x}+{y}")
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dataset path
        ttk.Label(main_frame, text="Dataset Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        path_frame = ttk.Frame(main_frame)
        path_frame.grid(row=0, column=1, sticky=tk.EW, pady=5)
        path_frame.columnconfigure(0, weight=1)
        
        self.dataset_path_var = tk.StringVar(value=self.config.get('default_dataset_path', ''))
        ttk.Entry(path_frame, textvariable=self.dataset_path_var).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(path_frame, text="Browse", command=self._browse_dataset).grid(row=0, column=1)
        
        # Video name
        ttk.Label(main_frame, text="Video Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.video_name_var = tk.StringVar(value="VisDrone Video")
        ttk.Entry(main_frame, textvariable=self.video_name_var).grid(row=1, column=1, sticky=tk.EW, pady=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Import Progress")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=10)
        progress_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var, 
            maximum=100.0
        )
        self.progress_bar.grid(row=0, column=0, sticky=tk.EW, padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to import")
        ttk.Label(progress_frame, textvariable=self.status_var).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, sticky=tk.E, pady=10)
        
        ttk.Button(button_frame, text="Import", command=self._import_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT)
        
        main_frame.columnconfigure(1, weight=1)
    
    def _browse_dataset(self):
        directory = filedialog.askdirectory(
            title="Select VisDrone Dataset Directory",
            initialdir=self.dataset_path_var.get()
        )
        if directory:
            self.dataset_path_var.set(directory)
    
    def _import_dataset(self):
        dataset_path = self.dataset_path_var.get()
        fps = 30
        
        if not dataset_path:
            messagebox.showerror("Error", "Please select a dataset directory")
            return
        
        self.status_var.set("Importing dataset... This may take a while")
        self.progress_var.set(10)
        self.update_idletasks()
        
        try:
            video_ids = self.db_manager.import_visdrone_dataset(dataset_path, fps)
            
            self.progress_var.set(100)
            self.status_var.set(f"Import completed successfully. Created {len(video_ids)} videos.")
            
            messagebox.showinfo("Success", f"Dataset imported successfully! Created {len(video_ids)} videos.")
            
            self.destroy()
        except Exception as e:
            self.progress_var.set(0)
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to import dataset: {str(e)}")