from PIL import Image, ImageTk

class CanvasManager:
    def __init__(self, root):
        self.original_canvas = None
        self.enhanced_canvas = None
        self.setup_canvas(root)

    def setup_canvas(self, root):
        """Initialize canvases for original and enhanced videos"""
        self.original_canvas = tk.Canvas(root, bg="black")
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.enhanced_canvas = tk.Canvas(root, bg="black")
        self.enhanced_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def display_frame(self, canvas, frame):
        """Display a frame on the specified canvas"""
        if frame is None:
            return
        
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo