from ultralytics import YOLO
import cv2
import numpy as np


class PuckDetector:
    def __init__(self, model_path='models/best.pt'):
        """
        Initializes the PuckDetector with a YOLOv8 model.

        Args:
            model_path (str): The file path to your custom-trained YOLOv8n model weights (.pt file).
        """
        try:
            # Load the custom-trained YOLOv8 model
            self.model = YOLO(model_path)
            print("PuckDetector initialized successfully.")
        except Exception as e:
            print(f"Error loading the YOLO model from {model_path}: {e}")
            self.model = None

    def detect_pucks(self, image: np.ndarray):
        """
        Performs inference on an image to detect pucks.

        Args:
            image (np.ndarray): The input image as a NumPy array (e.g., from OpenCV).

        Returns:
            list: A list of detection results, where each result contains
                  bounding box coordinates, confidence scores, and class labels.
        """
        if self.model is None:
            print("Model is not loaded. Cannot perform detection.")
            return []

        # Perform inference on the image
        results = self.model(image)
        return results

    def draw_detections(self, image: np.ndarray, results: list):
        """
        Draws the detected bounding boxes and labels on the image.

        Args:
            image (np.ndarray): The original image.
            results (list): The list of detection results from detect_pucks().

        Returns:
            np.ndarray: The image with detections drawn on it.
        """
        if not results:
            return image

        # The ultralytics results object has a built-in plotting function
        annotated_image = results[0].plot()
        return annotated_image

    def get_puck_centers(self, results: list):
        """
        Extracts the center coordinates of detected pucks and categorizes them by color.

        Args:
            results (list): The list of detection results from detect_pucks().

        Returns:
            dict: A dictionary with keys 'blue_pucks' and 'red_pucks', each
                containing a list of (x, y) coordinates for the puck centers.
        """
        puck_centers = {
            'blue_pucks': [],
            'red_pucks': []
        }

        if not results or not results[0].boxes:
            return puck_centers

        for box in results[0].boxes:
            class_id = box.cls.item()
            class_name = self.model.names[class_id]

            x_min, y_min, x_max, y_max = [int(coord) for coord in box.xyxy[0]]
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)

            if class_name == 'Blue_Pucks':
                puck_centers['blue_pucks'].append((center_x, center_y))
            elif class_name == 'Red_Pucks':
                puck_centers['red_pucks'].append((center_x, center_y))

        return puck_centers
