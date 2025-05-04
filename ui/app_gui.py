
import tkinter as tk
import threading
import cv2
import time
import os
import torch
import numpy as np
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from services.enhancer import *
from models.srcnn_model import *
from torchvision import transforms

class VideoEnhancementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Enhancement Application")

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.user_selection = None

        window_width = self.screen_width - 70
        window_height = self.screen_height - 70

        x_pos = (self.screen_width - window_width) // 2 - 10
        y_pos = (self.screen_height - window_height) // 2 - 30

        self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

        # Set to full screen
        # self.root.attributes('-fullscreen', True)
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Video processing variables
        self.cap = None
        self.video_path = None
        self.processing_thread = None
        self.export_thread = None
        self.is_running = False
        self.is_exporting = False
        self.frame_count = 0
        self.current_frame = None
        self.enhanced_frame = None
        self.current_frame_number = 0
        self.delay = 30  # ms between frames
        
        # Object tracking variables
        self.bg_subtractor = None
        self.object_ids = {}
        self.next_id = 1
        self.detec = []
        self.min_width = 10
        self.min_height = 10
        
        # AI model variables

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.ai_model_loaded = False
        self.my_font = ("Arial", 12, "bold")
        
        try:

            load_model("models\\srcnn_model.pth")
            self.ai_model_loaded = True
        except Exception as e:
            print(f"Error loading AI model: {str(e)}")
            self.ai_model_loaded = False
        
        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # Add exit button to top right corner
        # exit_button = tk.Button(root, text="X", command=self.exit_application,
        #                        bg="red", fg="white", font=("Arial", 12, "bold"), 
        #                        relief=tk.RAISED, height=1, width=3)
        # exit_button.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)
        
        # Create display frame for videos
        self.display_frame = tk.Frame(main_frame)
        self.display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for original video
        left_panel = tk.Frame(self.display_frame, bd=2, relief=tk.SUNKEN)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # video_label1 = tk.Label(left_panel, text="Source Video", font=self.my_font)
        # video_label1.pack(pady=5)
        
        # Canvas for original video
        self.original_canvas = tk.Canvas(left_panel, bg="black", width=self.screen_width//2 - 100, height=self.screen_height - 350)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar for original video
        original_progress_frame = tk.Frame(left_panel)
        original_progress_frame.pack(fill=tk.X, pady=5)
        
        original_label = tk.Label(original_progress_frame, text="Original Video", font=self.my_font)
        original_label.pack(side=tk.LEFT, pady=5)
        
        self.original_progress = ttk.Progressbar(original_progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.original_progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10, pady=5)
        
        # Right panel for enhanced video
        right_panel = tk.Frame(self.display_frame, bd=2, relief=tk.SUNKEN)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # video_label2 = tk.Label(right_panel, text="Enhanced Video", font=self.my_font)
        # video_label2.pack(pady=5)
        
        # Canvas for enhanced video
        self.enhanced_canvas = tk.Canvas(right_panel, bg="black", width=self.screen_width//2 - 100, height=self.screen_height - 350)
        self.enhanced_canvas.pack(fill=tk.BOTH, expand=True)
        self.enhanced_canvas.bind("<Button-1>", self.on_enhanced_canvas_click)
        
        # Progress bar for enhanced video
        enhanced_progress_frame = tk.Frame(right_panel)
        enhanced_progress_frame.pack(fill=tk.X, pady=5)
        
        enhanced_label = tk.Label(enhanced_progress_frame, text="Enhanced Video", font=self.my_font)
        enhanced_label.pack(side=tk.LEFT, pady=5)
        
        self.enhanced_progress = ttk.Progressbar(enhanced_progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.enhanced_progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10, pady=5)
        
        # Control frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Enhancement options frame
        options_frame = tk.Frame(control_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        # First row of checkboxes
        cb_frame1 = tk.Frame(options_frame)
        cb_frame1.pack(fill=tk.X)

        self.ai_model_var = tk.BooleanVar()
        self.ai_model_cb = tk.Checkbutton(cb_frame1, text="AI Model", variable=self.ai_model_var, font=self.my_font)
        self.ai_model_cb.pack(side=tk.LEFT, padx=25)
        if not self.ai_model_loaded:
            self.ai_model_cb.config(state=tk.DISABLED)

        self.color_enhancement_var = tk.BooleanVar()
        self.color_enhancement_cb = tk.Checkbutton(cb_frame1, text="Color Enhancement", variable=self.color_enhancement_var, font=self.my_font)
        self.color_enhancement_cb.pack(side=tk.LEFT, padx=25)

        # Add Histogram Equalization Checkbox
        self.histogram_equalization_var = tk.BooleanVar()
        self.histogram_equalization_cb = tk.Checkbutton(cb_frame1, text="Histogram Equ.",
                                                        variable=self.histogram_equalization_var, font=self.my_font)
        self.histogram_equalization_cb.pack(side=tk.LEFT, padx=20)
       
        # Add object tracking checkbox
        self.object_tracking_var = tk.BooleanVar()
        self.object_tracking_cb = tk.Checkbutton(cb_frame1, text="Track Objects", variable=self.object_tracking_var, font=self.my_font)
        self.object_tracking_cb.pack(side=tk.LEFT, padx=25)
        


        # Second row of checkboxes
        cb_frame2 = tk.Frame(options_frame)
        cb_frame2.pack(fill=tk.X, pady=5)
        
        self.sharpen_type1_var = tk.BooleanVar()
        self.sharpen_type1_cb = tk.Checkbutton(cb_frame2, text="Sharpen (Lablacian)", variable=self.sharpen_type1_var, font=self.my_font)
        self.sharpen_type1_cb.pack(side=tk.LEFT, padx=20)
        
        self.sharpen_type2_var = tk.BooleanVar()
        self.sharpen_type2_cb = tk.Checkbutton(cb_frame2, text="Sharpen (Sobel X)", variable=self.sharpen_type2_var, font=self.my_font)
        self.sharpen_type2_cb.pack(side=tk.LEFT, padx=20)

        self.sharpen_type3_var = tk.BooleanVar()
        self.sharpen_type3_cb = tk.Checkbutton(cb_frame2, text="Sharpen (Sobel Y)", variable=self.sharpen_type3_var, font=self.my_font)
        self.sharpen_type3_cb.pack(side=tk.LEFT, padx=20)
        
        # Add Blur Checkbox
        self.blur_var = tk.BooleanVar()
        self.blur_cb = tk.Checkbutton(cb_frame2, text="Blur", variable=self.blur_var, font=self.my_font)
        self.blur_cb.pack(side=tk.LEFT, padx=20)

        # Default Button
        default_button = tk.Button(cb_frame2, text="Reset", font=self.my_font, bg="green", fg="white",
                                   command=self.reset_to_default, height=1, width=10)
        default_button.pack(side=tk.LEFT, padx=10)

        # Sliders for Contrast, Brightness, and Gamma
        slider_frame = tk.Frame(control_frame)
        slider_frame.pack(fill=tk.X, pady=5)

        # Contrast Slider
        contrast_label = tk.Label(slider_frame, text="Contrast", font=self.my_font)
        contrast_label.pack(side=tk.LEFT, padx=10)
        self.contrast_slider = tk.Scale(slider_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, length=200, 
                                        command=self.update_enhanced_image)
        self.contrast_slider.set(1.0)  # Default value
        self.contrast_slider.pack(side=tk.LEFT, padx=10)

        # Brightness Slider
        brightness_label = tk.Label(slider_frame, text="Brightness", font=self.my_font)
        brightness_label.pack(side=tk.LEFT, padx=10)
        self.brightness_slider = tk.Scale(slider_frame, from_=-100, to=100, resolution=1, orient=tk.HORIZONTAL, length=200, 
                                          command=self.update_enhanced_image)
        self.brightness_slider.set(0)  # Default value
        self.brightness_slider.pack(side=tk.LEFT, padx=10)

        # Gamma Slider
        gamma_label = tk.Label(slider_frame, text="Gamma", font=self.my_font)
        gamma_label.pack(side=tk.LEFT, padx=10)
        self.gamma_slider = tk.Scale(slider_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, length=200, 
                                     command=self.update_enhanced_image)
        self.gamma_slider.set(1.0)  # Default value
        self.gamma_slider.pack(side=tk.LEFT, padx=10)

        # Add Slider for Distance Threshold
        dist_label = tk.Label(slider_frame, text="Tracking Dist", font=self.my_font)
        dist_label.pack(side=tk.LEFT, padx=10)
        self.distance_slider = tk.Scale(slider_frame, from_=2, to=50, resolution=1, orient=tk.HORIZONTAL, length=200, font=("Arial", 10))
        self.distance_slider.set(10)  # Default value
        self.distance_slider.pack(side=tk.LEFT, padx=10)

        # Bind the slider movement to an event
        self.distance_slider.bind("<ButtonRelease-1>", self.update_distance_threshold)
        
        # Button frame
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Create taller 3D-looking buttons
        load_button = tk.Button(button_frame, text="Load", command=self.load_video,
                               bg="#d0d0d0", relief=tk.RAISED, height=3, width=12, font=self.my_font)
        load_button.pack(side=tk.LEFT, padx=10)
        
        self.start_button = tk.Button(button_frame, text="Start", command=self.start_video,
                                    bg="#d0d0d0", relief=tk.RAISED, height=3, width=12, font=self.my_font)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.start_button.config(state=tk.DISABLED)
        
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_video,
                                   bg="#d0d0d0", relief=tk.RAISED, height=3, width=12, font=self.my_font)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        self.stop_button.config(state=tk.DISABLED)
        
        # Add Export button
        self.export_button = tk.Button(button_frame, text="Export Video", command=self.export_video,
                                     bg="brown", fg="black", relief=tk.RAISED, height=3, width=12, font=self.my_font)
        self.export_button.pack(side=tk.LEFT, padx=10)
        self.export_button.config(state=tk.DISABLED)
        
        # Export progress variable and label
        self.export_progress_var = tk.StringVar()
        self.export_progress_var.set("")
        self.export_progress_label = tk.Label(button_frame, textvariable=self.export_progress_var, 
                                             font=self.my_font, fg="#4CAF50")
        self.export_progress_label.pack(side=tk.LEFT, padx=5)

        self.metrics_button = tk.Button(button_frame, text="Metrics OFF", 
                                    command=self.toggle_metrics,
                                    bg="gray", fg="white", relief=tk.RAISED, 
                                    height=3, width=12, font=self.my_font)
        self.metrics_button.pack(side=tk.LEFT, padx=10)
        self.show_metrics = False  # Track metrics display state

        # Add the life stream key 
        self.live_stream_button = tk.Button(button_frame, text="Life Stream", command=self.live_stream_video,
                                     bg="red", fg="black", relief=tk.RAISED, height=3, width=12, font=self.my_font)
        self.live_stream_button.pack(side=tk.LEFT, padx=15)

        self.rtsp_entry = tk.Entry(button_frame, font=self.my_font, width=40)  # You can adjust width as needed
        self.rtsp_entry.pack(side=tk.LEFT, padx=5)
        self.rtsp_entry.insert(0, "rtsp://")

        # Bind the escape key to exit full screen
        self.root.bind("<Escape>", lambda event: self.exit_application())
        
    def on_enhanced_canvas_click(self, event):
        """Handle mouse click on the enhanced canvas and print coordinates."""
        # Get the x and y coordinates of the click
        x = event.x
        y = event.y
        self.user_selection = (x, y)
        print(f"User selection: {self.user_selection}")

    def load_video(self):
        """Open file dialog to choose a video or image file"""
        self.video_path = filedialog.askopenfilename(
            title="Select Video or Image File",
            filetypes=[
                ("Video and Image files", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        self.video_path = os.path.abspath(self.video_path)

        if self.video_path:
            # Check if the selected file is an image
            if self.video_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Load and display the image
                # image = cv2.imread(self.video_path)
                image = cv2.imdecode(np.fromfile(self.video_path, dtype=np.uint8), cv2.IMREAD_COLOR)

                if image is not None:
                    self.display_frame_on_canvas(image, self.original_canvas)
                    enhanced_image = self.enhance_frame(image)
                    self.display_frame_on_canvas(enhanced_image, self.enhanced_canvas, is_enhanced=True)
                    self.start_button.config(state=tk.NORMAL)
                    self.export_button.config(state=tk.NORMAL)
                    self.object_tracking_cb.config(state=tk.DISABLED)
                    self.current_frame = image
                else:
                    tk.messagebox.showerror("Error", "Could not load the image file.")
            else:
                # Assume it's a video and handle it as before
                self.cap = cv2.VideoCapture(self.video_path)
                if self.cap.isOpened():
                    self.start_button.config(state=tk.NORMAL)
                    self.export_button.config(state=tk.NORMAL)
                    self.object_tracking_cb.config(state=tk.NORMAL)
                    # Read the first frame and display it
                    ret, frame = self.cap.read()
                    if ret:
                        self.display_frame_on_canvas(frame, self.original_canvas)
                        enhanced = self.enhance_frame(frame)
                        self.display_frame_on_canvas(enhanced, self.enhanced_canvas, is_enhanced=True)
                        # Get total frames
                        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        # Reset position
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # Initialize progress bars
                        self.original_progress["maximum"] = self.frame_count
                        self.enhanced_progress["maximum"] = self.frame_count
                        self.original_progress["value"] = 0
                        self.enhanced_progress["value"] = 0
                        # Reset object tracking variables
                        self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
                        self.object_ids = {}
                        self.next_id = 1
                        self.detec = []
                else:
                    tk.messagebox.showerror("Error", "Could not open the video file.")

    def start_video(self):
        """Start processing the selected file (video or image)"""

        if self.video_path:
            # Check if the selected file is an image
            if self.video_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Load the image
                # image = cv2.imread(self.video_path)
                image = cv2.imdecode(np.fromfile(self.video_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    # Apply enhancements

                    enhanced_image = self.enhance_frame(image)
                    
                    # Display original and enhanced images
                    self.display_frame_on_canvas(image, self.original_canvas)
                    self.display_frame_on_canvas(enhanced_image, self.enhanced_canvas, is_enhanced=True)
                else:
                    tk.messagebox.showerror("Error", "Could not load the image file.")
            else:
                # Proceed with video processing as before
                if not self.is_running:
                    self.is_running = True
                    self.start_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.NORMAL)
                    
                    # Reset video to the beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_number = 0
                    
                    # Reset progress bars
                    self.original_progress["value"] = 0
                    self.enhanced_progress["value"] = 0
                    
                    # Reset object tracking variables
                    self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
                    self.object_ids = {}
                    self.next_id = 1
                    self.detec = []
                    
                    # Start processing in a separate thread
                    self.processing_thread = threading.Thread(target=self.process_video)
                    self.processing_thread.daemon = True
                    self.processing_thread.start()
                        
    def stop_video(self):
        """Stop playing the videos"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.NORMAL if self.video_path else tk.DISABLED)
    
        # Release the capture if it's a live stream
        if hasattr(self, 'is_live_stream') and self.is_live_stream:
            if self.cap and self.cap.isOpened():
                self.cap.release()
    
    def process_video(self):
        """Process and display video frames"""
        if not self.cap:
            return
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                # End of video - restart
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_number = 0
                self.original_progress["value"] = 0
                self.enhanced_progress["value"] = 0
                # Reset object tracking
                self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
                self.object_ids = {}
                self.next_id = 1
                self.detec = []
                continue
            
            # Store current frame
            self.current_frame = frame
            self.current_frame_number += 1
            
            # Process the frame for enhancement
            self.enhanced_frame = self.enhance_frame(frame)
            
            # Update both canvases
            self.root.after(0, lambda: self.display_frame_on_canvas(self.current_frame, self.original_canvas))
            self.root.after(0, lambda: self.display_frame_on_canvas(self.enhanced_frame, self.enhanced_canvas, is_enhanced=True))
            
            # Update progress bars
            self.root.after(0, lambda: self.update_progress())
            
            # Control frame rate
            self.root.after(self.delay)
    
    def update_progress(self):
        """Update the progress bars based on current frame"""
        if self.frame_count > 0:
            progress_value = self.current_frame_number
            self.original_progress["value"] = progress_value
            self.enhanced_progress["value"] = progress_value
  
    def track_objects(self, frame):
        """Apply object tracking to the frame"""
        # Create a copy to avoid modifying the input
        result = frame.copy()
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        
        # Apply background subtraction
        if self.bg_subtractor is None:
            self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        
        img_sub = self.bg_subtractor.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Apply morphological operations
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Track current centers for this frame
        current_centers = []
        
        # Process all contours that meet size requirements
        for (_, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            valid_contour = (w >= self.min_width) and (h >= self.min_height)
            if not valid_contour:
                continue
            
            # Get center of the object
            center = get_center(x, y, w, h)
            current_centers.append((center, (x, y, w, h)))
            # print(center)
            # Store detection for line crossing check
            if center not in self.detec:
                self.detec.append(center)
        
        # Update object IDs based on proximity to previous objects
        new_object_ids = {}
        # print(current_centers)
        for center, (x, y, w, h) in current_centers:
            obj_id = None
            # Try to match with existing objects
            min_dist = float('inf')
            min_id = None
            
            for known_center, known_id in self.object_ids.items():
                # Calculate distance between centers
                dist = np.sqrt((center[0] - known_center[0])**2 + (center[1] - known_center[1])**2)
                if dist < min_dist and dist < 100  :  # 50 pixel threshold for matching
                    min_dist = dist
                    min_id = known_id
            
            if min_id is not None:
                obj_id = min_id
            else:
                # New object detected
                obj_id = self.next_id
                self.next_id += 1
            
            # Store the new center with its ID
            new_object_ids[center] = obj_id
            
            # print(f"Object ID: {obj_id}, Center: {center}")
            # Draw rectangle around the object
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add label above the object
            cv2.putText(result, f"ID: {obj_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Update the object_ids dictionary for the next frame
        self.object_ids = new_object_ids
        
        # Find the nearest center to the user selection
        if self.user_selection:
            min_distance = float('inf')
            nearest_center = None
            for center, _ in current_centers:
                distance = np.sqrt((center[0] - self.user_selection[0])**2 + (center[1] - self.user_selection[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_center = center
            
            if nearest_center:
                print(f"Nearest center to user selection {self.user_selection}: {nearest_center} (Distance: {min_distance:.2f})")
                self.user_selection = nearest_center

        return result
        
    def enhance_frame(self, frame):
        """Apply selected enhancements to the frame"""
        # Create a copy to avoid modifying the original
        enhanced = frame.copy()

        # Lighting Enhancement
        enhanced = self.update_enhanced_contrast(frame=enhanced.copy())

        # Color Enhancement (Auto Tone)
        if self.color_enhancement_var.get():
            if len(enhanced.shape) == 3:  # Only apply to color images
                enhanced = auto_tone_image(enhanced)

        # Sharpen Enhancement
        enhanced = apply_sharpening(enhanced,self.sharpen_type1_var.get(),self.sharpen_type2_var.get(),self.sharpen_type3_var.get())

        # Blur Enhancement
        if self.blur_var.get():
            enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
        # AI model enhancement
        if self.ai_model_var.get() and self.ai_model_loaded:
            enhanced = apply_ai_model(enhanced)
        
        # Histogram Equalization
        if self.histogram_equalization_var.get():
            enhanced = self.histogram_equalization(enhanced)       

        # Object tracking - apply this last so tracking is visible
        if self.object_tracking_var.get():
            enhanced = self.track_objects(enhanced)
        
        if self.user_selection:
            x, y = self.user_selection
            cv2.circle(enhanced, (x, y), 4, (0, 0, 255), -1)  # Draw a red circle with radius 10
    
        return enhanced
    
    def display_frame_on_canvas(self, frame, canvas, is_enhanced=False):
        
        if frame is None:
            return
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # If canvas hasn't been drawn yet, use default dimensions
        if canvas_width <= 1:
            canvas_width = self.screen_width // 2 - 100
        if canvas_height <= 1:
            canvas_height = self.screen_height - 350
        
        # Resize frame to fit canvas while maintaining aspect ratio
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        
        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to RGB for PIL
        if len(resized_frame.shape) == 3:
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        else:
            # Convert grayscale to RGB
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
        
        if is_enhanced and self.current_frame is not None and self.show_metrics:
            try:
                psnr_val, ssim_val = compute_psnr_ssim(self.current_frame, self.enhanced_frame)
                text = f"PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}"
                cv2.putText(resized_frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            except Exception as e:
                print(f"Metric calc error: {e}")

        # Convert to PhotoImage
        image = Image.fromarray(resized_frame)
        photo = ImageTk.PhotoImage(image=image)
        
        # Clear canvas
        canvas.delete("all")
        
        # Update canvas
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep a reference to prevent garbage collection

    def export_video(self):
        """Export the enhanced video or image with selected enhancements"""
        if not self.video_path:
            tk.messagebox.showerror("Error", "No file loaded to export.")
            return

        # Check if the selected file is an image
        if self.video_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Ask user for save location
            output_path = filedialog.asksaveasfilename(
                title="Save Enhanced Image",
                defaultextension=".png",
                filetypes=[
                    ("PNG image", "*.png"),
                    ("JPEG image", "*.jpg *.jpeg"),
                    ("BMP image", "*.bmp"),
                    ("All files", "*.*")
                ]
            )
            
            if not output_path:
                return
            
            # Load the image
            image = cv2.imdecode(np.fromfile(self.video_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            # image = cv2.imread(self.video_path)
            if image is not None:
                # Apply enhancements
                enhanced_image = self.enhance_frame(image)
                
                # Save the enhanced image
                # cv2.imwrite(output_path, enhanced_image)
                cv2.imencode('.png', enhanced_image)[1].tofile(output_path)
                tk.messagebox.showinfo("Export Complete", f"Enhanced image saved to:\n{output_path}")
            else:
                tk.messagebox.showerror("Error", "Failed to load the image for export.")
        
        else:
            # Proceed with exporting video as before
            if self.is_exporting:
                tk.messagebox.showinfo("Export in Progress", "An export is already in progress.")
                return
            
            # Ask user for save location
            output_path = filedialog.asksaveasfilename(
                title="Save Enhanced Video",
                defaultextension=".mp4",
                filetypes=[
                    ("MP4 video", "*.mp4"),
                    ("AVI video", "*.avi"),
                    ("MKV video", "*.mkv"),
                    ("All files", "*.*")
                ]
            )
            
            if not output_path:
                return
            
            # Disable buttons during export
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.DISABLED)
            
            # Stop playback if running
            if self.is_running:
                self.stop_video()
            
            # Start export in separate thread
            self.is_exporting = True
            self.export_thread = threading.Thread(target=self.process_export, args=(output_path,))
            self.export_thread.daemon = True
            self.export_thread.start()
                
    def toggle_metrics(self):
        """Toggle metrics display on/off"""
        self.show_metrics = not self.show_metrics
        if self.show_metrics:
            self.metrics_button.config(text="Metrics ON", bg="green")
        else:
            self.metrics_button.config(text="Metrics OFF", bg="gray")
        
        # Force redisplay of current frame to update metrics visibility
        if self.current_frame is not None and self.enhanced_frame is not None:
            self.display_frame_on_canvas(self.current_frame, self.original_canvas, is_enhanced=False)
            self.display_frame_on_canvas(self.enhanced_frame, self.enhanced_canvas, is_enhanced=True)

    def live_stream_video(self):
        
        # Get RTSP URL from entry widget
        rtsp_url = self.rtsp_entry.get()
        
        if not rtsp_url or not rtsp_url.startswith("rtsp://"):
            messagebox.showerror("Error", "Please enter a valid RTSP URL starting with 'rtsp://'")
            return
        
        # Stop any current video processing
        if self.is_running:
            self.stop_video()
        
        # Release any existing video capture
        if self.cap:
            self.cap.release()
        
        try:
            # Open the RTSP stream
            self.cap = cv2.VideoCapture(rtsp_url)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open RTSP stream. Check the URL and connection.")
                return
            
            # Enable/disable buttons
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.DISABLED)
            
            # Set flags
            self.is_running = True
            self.frame_count = 0  # Not applicable for live stream
            self.current_frame_number = 0
            
            # Reset progress bars
            self.original_progress["value"] = 0
            self.enhanced_progress["value"] = 0
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.process_live_stream)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start live stream: {str(e)}")


    def process_live_stream(self):
        """Process frames from the live RTSP stream"""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                # If we lose connection, try to reconnect
                print("Lost connection, attempting to reconnect...")
                self.cap.release()
                time.sleep(1)  # Wait before reconnecting
                self.cap = cv2.VideoCapture(self.rtsp_entry.get())
                if not self.cap.isOpened():
                    self.root.after(0, lambda: messagebox.showerror("Error", "Lost connection to RTSP stream"))
                    self.is_running = False
                    break
                continue
            
            # Store current frame
            self.current_frame = frame
            self.current_frame_number += 1
            
            if self.current_frame_number % 20 == 0:  # Update every 10 frames for performance
                # Process the frame for enhancement
                self.enhanced_frame = self.enhance_frame(frame)            
                # Update both canvases
                self.root.after(0, lambda: self.display_frame_on_canvas(self.current_frame, self.original_canvas))
                self.root.after(0, lambda: self.display_frame_on_canvas(self.enhanced_frame, self.enhanced_canvas, is_enhanced=True))
            
            # Control frame rate
            time.sleep(0.01)  # ~30fps
        
    def process_export(self, output_path):
        """Process the video and save it with enhancements"""
        try:
            # Reset the video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            start_time = time.time()
            
            # Process each frame
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Apply enhancements
                enhanced = self.enhance_frame(frame)
                
                # Write the frame
                out.write(enhanced)
                
                # Update progress
                frame_count += 1
                progress_percent = int((frame_count / total_frames) * 100)
                elapsed = time.time() - start_time
                remaining = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
                
                # Update progress label
                self.export_progress_var.set(f"Exporting: {progress_percent}% (Est. {int(remaining)}s remaining)")
                
                # Process GUI events to keep UI responsive
                self.root.update_idletasks()
            
            # Release writer
            out.release()
            
            # Show completion message
            self.root.after(0, lambda: messagebox.showinfo("Export Complete", 
                                                        f"Enhanced video saved to:\n{output_path}"))
            self.export_progress_var.set("Export complete")
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Export Error", 
                                                         f"Error during export: {str(e)}"))
            self.export_progress_var.set("Export failed")
            
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.export_button.config(state=tk.NORMAL))
            self.is_exporting = False
    
    def exit_application(self):
        """Clean up and exit the application"""
        if self.is_running:
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
        
        if self.is_exporting:
            if messagebox.askyesno("Export in Progress", "An export is in progress. Do you want to cancel and exit?"):
                self.is_exporting = False
                if self.export_thread and self.export_thread.is_alive():
                    self.export_thread.join(timeout=1.0)
            else:
                return
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.root.destroy()

    def update_enhanced_contrast(self, event=None, frame=None):
            
        if (frame is None):
            frame = self.current_frame
        # Get slider values
        try:
            contrast = float(self.contrast_slider.get())
            brightness = float(self.brightness_slider.get())
            gamma = float(self.gamma_slider.get())
        except Exception as e:
            print(f"Error updating enhanced image: {e}")
            return frame
            
        if frame is not None:     
            adjusted = self.apply_adjustments(frame, contrast, brightness, gamma)
            # Apply enhancements
            return adjusted
        else:
            return frame

    def update_enhanced_image(self,event=None):
        """Update the enhanced image based on slider values"""
        enhanced_image = self.update_enhanced_contrast()
        # Update the enhanced canvas
        self.display_frame_on_canvas(enhanced_image, self.enhanced_canvas, is_enhanced=True)

    def apply_adjustments(self, frame, contrast, brightness, gamma):
        """Apply contrast, brightness, and gamma adjustments to a frame"""
        # Adjust contrast and brightness
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

        # Adjust gamma
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
            adjusted = cv2.LUT(adjusted, table)

        return adjusted
    
    def reset_to_default(self):
        """Reset all sliders and checkboxes to default values and update the enhanced image"""
        # Reset sliders to default values
        self.contrast_slider.set(1.0)  # Reset contrast to default
        self.brightness_slider.set(0)  # Reset brightness to default
        self.gamma_slider.set(1.0)  # Reset gamma to default

        # Uncheck all checkboxes
        self.ai_model_var.set(False)
        self.color_enhancement_var.set(False)
        self.object_tracking_var.set(False)
        self.sharpen_type1_var.set(False)
        self.sharpen_type2_var.set(False)
        self.sharpen_type3_var.set(False)
        self.histogram_equalization_var.set(False)
        self.user_selection = None  # Reset user selection

        # Reset the enhanced image to the original frame with no enhancements
        if self.current_frame is not None:
            # Display the original frame on the enhanced canvas
            self.display_frame_on_canvas(self.current_frame, self.enhanced_canvas, is_enhanced=False)

    def histogram_equalization(self, frame):
        """Perform histogram equalization on the input frame"""
        # Check if the image is grayscale or color
        if len(frame.shape) == 2:  # Grayscale image
            equalized = cv2.equalizeHist(frame)
        else:  # Color image
            # Convert to YUV color space
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # Equalize the Y (luminance) channel
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            # Convert back to BGR color space
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return equalized
    
    def update_distance_threshold(self, event):
        """Update the distance threshold when the slider value changes."""
        self.min_width = self.distance_slider.get()
        self.min_height = self.distance_slider.get()
        # print(f"Distance threshold updated to: {new_threshold}")
        # Additional logic can be added here if needed
    