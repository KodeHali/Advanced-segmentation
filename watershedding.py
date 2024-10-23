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

        self.root.geometry("800x600")

        # Upload button
        self.upload_btn = Button(self.root, text="Upload Image", command=self.upload_image, padx=20, pady=10)
        self.upload_btn.pack(pady=10)

        # Watershed button
        self.watershed_btn = Button(self.root, text="Apply Watershed", command=self.apply_watershed, padx=20, pady=10)
        self.watershed_btn.pack(pady=10)

        # Reset button
        self.reset_btn = Button(self.root, text="Reset", command=self.reset_image, padx=20, pady=10)
        self.reset_btn.pack(pady=10)

        # Canvas to display image
        self.image_canvas = Canvas(self.root, bg="white")
        self.image_canvas.pack(fill=BOTH, expand=YES)

        self.image_path = None
        self.original_image = None
        self.processed_image = None

        # Dyanmically adjusting the image canvas
        self.root.bind("<Configure>", self.on_resize)

    def upload_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.original_image)

    def display_image(self, image):
        """Display the image in the canvas."""
        
        # getting current size of image
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        # Get the full resolution of the image
        image_height, image_width = image.shape[:2]

        # Ccalculating scaling factor
        scale_factor = min(canvas_width / image_width, canvas_height / image_height)

        # Calculate new dimensions of the image based on the scaling factor
        display_width = int(image_width * scale_factor)
        display_height = int(image_height * scale_factor)

        resized_image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)

        # Converting to rgb to display in tkinter
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        self.tk_image = ImageTk.PhotoImage(pil_image)

        self.image_canvas.delete("all")

        # centering image
        self.image_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=CENTER, image=self.tk_image)

        self.image_canvas.config(width=display_width, height=display_height)

    def apply_watershed(self):
        """Applying the algorithm, with a lot of pre-processing for the algoruthm to work on the low contrast scans"""
        if self.image_path is None:
            return

        # using a copy so I can display the original later
        image = self.original_image.copy()

        # Converting to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Using CLAHE to enhance contrasts. Contrasts were consistantly too weak for the algorithm
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 
        enhanced_image = clahe.apply(gray_blurred)

        # Tuning the thresholds to isolate bright regions. These thresholds seem the best for low-contrast scans
        _, binary_image = cv2.threshold(
            enhanced_image, 120, 255, cv2.THRESH_BINARY)

        # Debug: Show the binary image after thresholding
        resized_binary_image = resize_for_debugging(binary_image)
        cv2.imshow("Binary Image", resized_binary_image)
        cv2.waitKey(0)

        # Using morphological opening
        kernel = np.ones((3, 3), np.uint8)
        opened_image = cv2.morphologyEx(
            binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

        # Closing small holes int he tumors
        closed_image = cv2.morphologyEx(
            opened_image, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Debug: Show the image after morphological operations
        resized_morph_image = resize_for_debugging(closed_image)
        cv2.imshow("Morphological Operations Result", resized_morph_image)
        cv2.waitKey(0)

        sure_bg = cv2.dilate(closed_image, kernel, iterations=1)

        dist_transform = cv2.distanceTransform(closed_image, cv2.DIST_L2, 5)

        _, sure_fg = cv2.threshold(
            dist_transform, 0.3 * dist_transform.max(), 255, 0)

        # Debug: Show the sure foreground
        resized_sure_fg = resize_for_debugging(sure_fg)
        cv2.imshow("Sure Foreground", resized_sure_fg)
        cv2.waitKey(0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # labelling markers
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Debug: Show the markers before watershed
        markers_visual = np.uint8(markers * (255 / markers.max()))
        resized_markers = resize_for_debugging(markers_visual)
        cv2.imshow("Markers before Watershed", resized_markers)
        cv2.waitKey(0)

        # algorithm
        markers = cv2.watershed(image, markers)

        # marking the boundaries
        image[markers == -1] = [0, 0, 255]  # Red 

        # Reducing small regions from markers
        # Trying to incldue smaller regions
        min_area = 300
        markers_8u = np.uint8(markers)
        contours, _ = cv2.findContours(
            markers_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                cv2.drawContours(markers, [cnt], -1, 1, -1)

        # reapplying algorithm
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [0, 0, 255]  # Red

        # Debug: Display the final processed image with watershed boundaries
        cv2.imshow("Final Image with Watershed Boundaries", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def reset_image(self):
        """Reset the image to its original form before any processing."""
        if self.original_image is not None:
            self.display_image(self.original_image)

    def on_resize(self, event):
        """Handle window resizing"""
        if self.original_image is not None:
            self.display_image(self.original_image)

if __name__ == "__main__":
    root = Tk()
    app = WatershedApp(root)
    root.mainloop()
