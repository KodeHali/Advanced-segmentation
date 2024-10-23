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

        # Upload button
        self.upload_btn = Button(self.root, text="Upload Image", command=self.upload_image, padx=20, pady=10)
        self.upload_btn.pack(pady=10)

        # Manual button
        self.manual_contour_btn = Button(self.root, text="Manual Contour (Click & Drag)", command=self.enable_manual_contour, padx=20, pady=10)
        self.manual_contour_btn.pack(pady=10)

        # Auto button
        self.auto_contour_btn = Button(self.root, text="Auto Contour", command=self.place_auto_contour, padx=20, pady=10)
        self.auto_contour_btn.pack(pady=10)

        # Apply button
        self.active_contour_btn = Button(self.root, text="Apply Active Contour", command=self.apply_active_contour, padx=20, pady=10)
        self.active_contour_btn.pack(pady=10)

        # Reset button
        self.reset_btn = Button(self.root, text="Reset", command=self.reset_image, padx=20, pady=10)
        self.reset_btn.pack(pady=10)

        # Using a canvas to display the processed image
        self.image_canvas = Canvas(self.root, bg="white")
        self.image_canvas.pack(fill=BOTH, expand=YES)

        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.initial_contour = []
        self.is_manual_mode = False
        self.scaling_factor = 1.0

        # For the contour circles. 
        self.start_x = None
        self.start_y = None
        self.radius = None

        self.image_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def upload_image(self):
        """Upload and display image."""
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            image_height, image_width = self.original_image.shape[:2]

            self.root.geometry(f"{image_width}x{image_height}")
            self.display_image(self.original_image)

    def display_image(self, image):
        """Display the image in canvas"""
        image_height, image_width = image.shape[:2]

        # I put the canvas in the bottom left due to scaling / dimension issues 
        self.image_canvas.config(width=image_width, height=image_height)

        # Displaying image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=NW, image=self.tk_image)

    def enable_manual_contour(self):
        """Enable manual contour mode"""
        if self.image_path is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return
        self.is_manual_mode = True
        self.initial_contour = []  # Clearing the previous contours if there are any
        messagebox.showinfo("Manual Contour Mode", "Click and drag to create a circle.")

    def on_mouse_down(self, event):
        """Capturing the first click"""
        if self.is_manual_mode:
            self.start_x = event.x
            self.start_y = event.y

    def on_mouse_drag(self, event):
        """Updating circle radius on the cavnas"""
        if self.is_manual_mode and self.start_x is not None and self.start_y is not None:
            # Calculating radius based on start -> mouse location
            current_x = event.x
            current_y = event.y
            self.radius = int(math.sqrt((current_x - self.start_x) ** 2 + (current_y - self.start_y) ** 2))

            # Redrawing image with the circle
            self.display_image_with_circle(self.start_x, self.start_y, self.radius)

    def on_mouse_up(self, event):
        """Finalize the circle when the mouse button is released"""
        if self.is_manual_mode and self.start_x is not None and self.start_y is not None:
            # converting canvas to image contours and storing coordinates
            image_start_x = int(self.start_x)
            image_start_y = int(self.start_y)
            image_radius = int(self.radius)

            # Generate circle points as the initial contour
            self.initial_contour = [(image_start_x + image_radius * np.cos(t), image_start_y + image_radius * np.sin(t)) 
                                    for t in np.linspace(0, 2 * np.pi, 100)]
            
            # Redrawing the finak image 
            self.display_image_with_circle(self.start_x, self.start_y, self.radius)

    def display_image_with_circle(self, center_x, center_y, radius):
        """Display the image with the current circle being drawn."""
        result_image = self.original_image.copy()
        cv2.circle(result_image, (center_x, center_y), radius, (0, 0, 255), 2)
        self.display_image(result_image)

    def place_auto_contour(self):
        """Automatically place initial contours."""
        if self.image_path is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # automatically definr a circular contour around the center of the image
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
        """Apply algorithm with either contour (mnaual or auto)."""
        if self.image_path is None or not self.initial_contour:
            messagebox.showerror("Error", "Please upload an image and define an initial contour.")
            return

        # Convert the initial contour into a numpy array
        initial_contour_np = np.array(self.initial_contour, dtype=np.float32)

        # Convert image to grayscale for processing
        img_gray = rgb2gray(cv2.imread(self.image_path))

        # Define a refined set of 12 parameter combinations based on the best-performing ones
        param_sets = [
            (0.008, 0.015, 0.03, 3),  # We've narrowed down the most efficient parameters to this set
            (0.008, 0.015, 0.03, 5),  
            (0.01, 0.015, 0.03, 3),  
            (0.01, 0.015, 0.03, 5),   
            (0.01, 0.02, 0.025, 3), 
            (0.01, 0.02, 0.025, 5),   
            (0.012, 0.02, 0.03, 3), 
            (0.012, 0.02, 0.03, 5),   
            (0.01, 0.025, 0.03, 3), 
            (0.01, 0.025, 0.03, 5),   
            (0.012, 0.025, 0.03, 3),  
            (0.012, 0.025, 0.03, 5)
        ]



        # Display all the figures
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))

        # Avpoiding index issues
        axes = axes.flatten()

        # Running for all sets (and printing them due to some confusion during tests)
        for i, (alpha, beta, gamma, sigma) in enumerate(param_sets):
            print(f"Processing set {i + 1}: alpha={alpha}, beta={beta}, gamma={gamma}, sigma={sigma}")

            # Applying Gaussian smoothing
            img_smoothed = gaussian(img_gray, sigma=sigma, preserve_range=False)

            # algorithm
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
            self.initial_contour = []  
            self.is_manual_mode = False # resetting
            self.display_image(self.original_image)


if __name__ == "__main__":
    root = Tk()
    app = ActiveContourApp(root)
    root.mainloop()
