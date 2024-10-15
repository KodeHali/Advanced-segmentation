import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import messagebox
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import math

class ActiveContourApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Active Contour Segmentation GUI")
        self.root.geometry("800x600")

        # Upload image button
        self.upload_btn = Button(self.root, text="Upload Image", command=self.upload_image, padx=20, pady=10)
        self.upload_btn.pack(pady=10)

        # Manual Contour button for user-defined contour
        self.manual_contour_btn = Button(self.root, text="Manual Contour (Click & Drag)", command=self.enable_manual_contour, padx=20, pady=10)
        self.manual_contour_btn.pack(pady=10)

        # Automatic Contour button for auto-initialized contour (just placing it, not running)
        self.auto_contour_btn = Button(self.root, text="Auto Contour", command=self.place_auto_contour, padx=20, pady=10)
        self.auto_contour_btn.pack(pady=10)

        # Apply Active Contour button
        self.active_contour_btn = Button(self.root, text="Apply Active Contour", command=self.apply_active_contour, padx=20, pady=10)
        self.active_contour_btn.pack(pady=10)

        # Reset button
        self.reset_btn = Button(self.root, text="Reset", command=self.reset_image, padx=20, pady=10)
        self.reset_btn.pack(pady=10)

        # Create a Canvas to display the image in the GUI
        self.image_canvas = Canvas(self.root, bg="white")
        self.image_canvas.pack(fill=BOTH, expand=YES)

        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.initial_contour = []
        self.is_manual_mode = False
        self.scaling_factor = 1.0

        # Variables for click-and-drag circle drawing
        self.start_x = None
        self.start_y = None
        self.radius = None

        # Bind the mouse events for drawing the circle
        self.image_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def upload_image(self):
        """Upload and display the image, and adjust the window size to match the image dimensions."""
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            image_height, image_width = self.original_image.shape[:2]

            # Adjust the size of the GUI to match the size of the image
            self.root.geometry(f"{image_width}x{image_height}")
            self.display_image(self.original_image)

    def display_image(self, image):
        """Display the image in the Tkinter canvas (GUI)"""
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        image_height, image_width = image.shape[:2]

        # Match the canvas size to the image size (no scaling)
        self.image_canvas.config(width=image_width, height=image_height)

        # Display the image as is, without scaling
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=NW, image=self.tk_image)

    def enable_manual_contour(self):
        """Enable manual contour mode (click and drag)."""
        if self.image_path is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return
        self.is_manual_mode = True
        self.initial_contour = []  # Clear any previously placed contours
        messagebox.showinfo("Manual Contour Mode", "Click and drag to create a circle.")

    def on_mouse_down(self, event):
        """Capture the starting point when the mouse button is pressed."""
        if self.is_manual_mode:
            self.start_x = event.x
            self.start_y = event.y

    def on_mouse_drag(self, event):
        """Update the circle radius dynamically as the mouse is dragged."""
        if self.is_manual_mode and self.start_x is not None and self.start_y is not None:
            # Calculate the radius based on the distance from the start point to the current mouse position
            current_x = event.x
            current_y = event.y
            self.radius = int(math.sqrt((current_x - self.start_x) ** 2 + (current_y - self.start_y) ** 2))

            # Redraw the image with the current circle
            self.display_image_with_circle(self.start_x, self.start_y, self.radius)

    def on_mouse_up(self, event):
        """Finalize the circle when the mouse button is released."""
        if self.is_manual_mode and self.start_x is not None and self.start_y is not None:
            # Convert the canvas coordinates to image coordinates and store the final contour
            image_start_x = int(self.start_x)
            image_start_y = int(self.start_y)
            image_radius = int(self.radius)

            # Generate circle points as the initial contour (using parametric equation of a circle)
            self.initial_contour = [(image_start_x + image_radius * np.cos(t), image_start_y + image_radius * np.sin(t)) 
                                    for t in np.linspace(0, 2 * np.pi, 100)]
            
            # Redraw the final image with the contour
            self.display_image_with_circle(self.start_x, self.start_y, self.radius)

    def display_image_with_circle(self, center_x, center_y, radius):
        """Display the image with the current circle being drawn."""
        result_image = self.original_image.copy()
        cv2.circle(result_image, (center_x, center_y), radius, (0, 0, 255), 2)

        # Display the image in the GUI
        self.display_image(result_image)

    def place_auto_contour(self):
        """Automatically place an initial contour (without running the algorithm)."""
        if self.image_path is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Automatically define a circular contour around the center of the image
        h, w = self.original_image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(h, w) // 4
        self.initial_contour = [(center[0] + int(radius * np.cos(t)), center[1] + int(radius * np.sin(t))) for t in np.linspace(0, 2 * np.pi, 100)]
        self.is_manual_mode = False  # Disable manual mode

        # Update the displayed image in the GUI with the auto contour points
        result_image = self.original_image.copy()
        for point in self.initial_contour:
            cv2.circle(result_image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        self.display_image(result_image)


    def apply_active_contour(self):
        """Apply the active contour segmentation after placing either manual or auto contour."""
        if self.image_path is None or not self.initial_contour:
            messagebox.showerror("Error", "Please upload an image and define an initial contour.")
            return

        # Convert the initial contour into a numpy array
        initial_contour_np = np.array(self.initial_contour, dtype=np.float32)

        # Convert image to grayscale for processing
        img_gray = rgb2gray(cv2.imread(self.image_path))

        # Define a refined set of 12 parameter combinations based on the best-performing ones
        param_sets = [
            (0.008, 0.015, 0.03, 3),  # Slight decrease in `alpha` and `beta` for more flexibility
            (0.008, 0.015, 0.03, 5),  
            (0.01, 0.015, 0.03, 3),   # Keep `alpha` same, decrease `beta` slightly
            (0.01, 0.015, 0.03, 5),   
            (0.01, 0.02, 0.025, 3),   # Decrease `gamma` slightly for less sensitivity
            (0.01, 0.02, 0.025, 5),   
            (0.012, 0.02, 0.03, 3),   # Increase `alpha` slightly to test more responsiveness
            (0.012, 0.02, 0.03, 5),   
            (0.01, 0.025, 0.03, 3),   # Increase `beta` slightly for smoother deformation
            (0.01, 0.025, 0.03, 5),   
            (0.012, 0.025, 0.03, 3),  # Balance higher `alpha` with slightly higher `beta`
            (0.012, 0.025, 0.03, 5)
        ]



        # Prepare a 3x4 subplot grid to display all 12 combinations
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))

        # Flatten the axes array to avoid index issues
        axes = axes.flatten()

        # Process each parameter set and print them in the terminal
        for i, (alpha, beta, gamma, sigma) in enumerate(param_sets):
            print(f"Processing set {i + 1}: alpha={alpha}, beta={beta}, gamma={gamma}, sigma={sigma}")

            # Apply Gaussian smoothing with the specified sigma
            img_smoothed = gaussian(img_gray, sigma=sigma, preserve_range=False)

            # Apply active contour algorithm
            snake = active_contour(img_smoothed, initial_contour_np, alpha=alpha, beta=beta, gamma=gamma)

            # Plot the result in the corresponding subplot
            ax = axes[i]
            ax.imshow(img_gray, cmap='gray')
            ax.plot(initial_contour_np[:, 0], initial_contour_np[:, 1], '--r', lw=2)
            ax.plot(snake[:, 0], snake[:, 1], '-b', lw=2)
            ax.set_xticks([]), ax.set_yticks([])

        plt.tight_layout()
        plt.show()
            
    def reset_image(self):
        """Reset the image and the initial contour."""
        if self.original_image is not None:
            self.initial_contour = []  # Clear the contour points
            self.is_manual_mode = False  # Reset to non-manual mode
            self.display_image(self.original_image)


if __name__ == "__main__":
    root = Tk()
    app = ActiveContourApp(root)
    root.mainloop()
