import tkinter as tk

def get_screen_size():
    """
    Get the screen size using tkinter.
    :return: (width, height)
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height