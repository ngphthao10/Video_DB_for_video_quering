import tkinter as tk
from app.gui.main_window import MainWindow
from utils.config import load_config
from app.database_manager import DatabaseManager

def main():
    config = load_config()
    
    root = tk.Tk()
    root.title("VisDrone Video Database with YOLO Detection")
    root.geometry("1200x800")
    
    db_manager = DatabaseManager(config)
    db_manager.create_indices()
    
    app = MainWindow(root, config)
    
    root.mainloop()

if __name__ == "__main__":
    main()