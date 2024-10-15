import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import messagebox

def resize_for_debugging(image, width=600):
    """Resize the image to a fixed width, keeping the aspect ratio."""
    height, original_width = image.shape[:2]
    scale_factor = width / original_width
    new_height = int(height * scale_factor)
    resized_image = cv2.resize(image, (width, new_height))
    return resized_image

class WatershedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Watershed Segmentation GUI")

        # Set a larger window size for the GUI
        self.root.geometry("800x600")

        # Upload image button with padding
        self.upload_btn = Button(self.root, text="Upload Image", command=self.upload_image, padx=20, pady=10)
        self.upload_btn.pack(pady=10)

        # Apply Watershed button with padding
        self.watershed_btn = Button(self.root, text="Apply Watershed", command=self.apply_watershed, padx=20, pady=10)
        self.watershed_btn.pack(pady=10)

        # Add Reset button
        self.reset_btn = Button(self.root, text="Reset", command=self.reset_image, padx=20, pady=10)
        self.reset_btn.pack(pady=10)

        # Create a Canvas to display the image
        self.image_canvas = Canvas(self.root, bg="white")
        self.image_canvas.pack(fill=BOTH, expand=YES)

        self.image_path = None
        self.original_image = None
        self.processed_image = None

        # Bind the resize event to dynamically adjust the image canvas
        self.root.bind("<Configure>", self.on_resize)

    def upload_image(self):
        # Use file dialog to upload image
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            # Read the image
            self.original_image = cv2.imread(self.image_path)
            # Display the original image without rescaling
            self.display_image(self.original_image)

    def display_image(self, image):
        """Display the image in the canvas, scaling it to fit the canvas while maintaining aspect ratio."""
        
        # Get the current size of the canvas (this is the space we have to fit the image)
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        # Get the actual size of the image (full resolution)
        image_height, image_width = image.shape[:2]

        # Calculate the scaling factor to fit the image within the canvas while maintaining aspect ratio
        scale_factor = min(canvas_width / image_width, canvas_height / image_height)

        # Calculate new dimensions of the image based on the scaling factor
        display_width = int(image_width * scale_factor)
        display_height = int(image_height * scale_factor)

        # Resize the image for display (this doesn't affect the original image resolution)
        resized_image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)

        # Convert the OpenCV image (BGR format) to RGB format for Tkinter display
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Convert the PIL image to a format that Tkinter can use
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # Clear the canvas and add the new image
        self.image_canvas.delete("all")

        # Center the image in the canvas
        self.image_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=CENTER, image=self.tk_image)

        # Optionally, you can also ensure the canvas resizes properly (in case the canvas needs adjusting)
        self.image_canvas.config(width=display_width, height=display_height)

    def apply_watershed(self):
        if self.image_path is None:
            return

        # Step 1: Work with a copy of the original image
        image = self.original_image.copy()

        # Step 2: Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply Gaussian blur to reduce noise
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 4: Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 
        enhanced_image = clahe.apply(gray_blurred)

        # Step 5: Thresholding to isolate bright regions (tumors)
        # Lower the threshold value to include more regions
        _, binary_image = cv2.threshold(
            enhanced_image, 120, 255, cv2.THRESH_BINARY)

        # Alternative: Use Otsu's method
        # _, binary_image = cv2.threshold(
        #     enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Alternative: Use adaptive thresholding
        # binary_image = cv2.adaptiveThreshold(
        #     enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY, 11, 2)

        # Debug: Show the binary image after thresholding
        resized_binary_image = resize_for_debugging(binary_image)
        cv2.imshow("Binary Image", resized_binary_image)
        cv2.waitKey(0)

        # Step 6: Remove small noise with morphological opening
        kernel = np.ones((3, 3), np.uint8)
        opened_image = cv2.morphologyEx(
            binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

        # Step 7: Use morphological closing to close small holes inside the tumors
        closed_image = cv2.morphologyEx(
            opened_image, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Debug: Show the image after morphological operations
        resized_morph_image = resize_for_debugging(closed_image)
        cv2.imshow("Morphological Operations Result", resized_morph_image)
        cv2.waitKey(0)

        # Step 8: Sure background area (using dilation)
        sure_bg = cv2.dilate(closed_image, kernel, iterations=1)

        # Step 9: Compute the distance transform
        dist_transform = cv2.distanceTransform(closed_image, cv2.DIST_L2, 5)

        # Step 10: Threshold the distance transform to get sure foreground
        # Decrease the threshold multiplier to include more regions
        _, sure_fg = cv2.threshold(
            dist_transform, 0.3 * dist_transform.max(), 255, 0)

        # Debug: Show the sure foreground
        resized_sure_fg = resize_for_debugging(sure_fg)
        cv2.imshow("Sure Foreground", resized_sure_fg)
        cv2.waitKey(0)

        # Step 11: Find unknown region (neither sure background nor sure foreground)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Step 12: Marker labelling using connected components
        num_labels, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0 but 1
        markers = markers + 1

        # Mark the unknown region with zero
        markers[unknown == 255] = 0

        # Debug: Show the markers before watershed
        markers_visual = np.uint8(markers * (255 / markers.max()))
        resized_markers = resize_for_debugging(markers_visual)
        cv2.imshow("Markers before Watershed", resized_markers)
        cv2.waitKey(0)

        # Step 13: Apply the watershed algorithm
        markers = cv2.watershed(image, markers)

        # Step 14: Mark watershed boundaries
        image[markers == -1] = [0, 0, 255]  # Red color in BGR

        # Step 15: Remove small regions from the markers
        # Reduce min_area to include smaller regions
        min_area = 300  # Adjust this threshold as needed
        markers_8u = np.uint8(markers)
        contours, _ = cv2.findContours(
            markers_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                cv2.drawContours(markers, [cnt], -1, 1, -1)  # Set region to background

        # Re-apply watershed after removing small regions
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [0, 0, 255]  # Red color in BGR

        # Debug: Display the final processed image with watershed boundaries
        cv2.imshow("Final Image with Watershed Boundaries", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def reset_image(self):
        """Reset the image to its original form before any processing."""
        if self.original_image is not None:
            self.display_image(self.original_image)

    def on_resize(self, event):
        """Handle window resizing and update the displayed image accordingly."""
        if self.original_image is not None:
            self.display_image(self.original_image)

if __name__ == "__main__":
    root = Tk()
    app = WatershedApp(root)
    root.mainloop()
