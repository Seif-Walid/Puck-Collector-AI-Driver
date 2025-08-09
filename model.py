from ultralytics import YOLO
import cv2
import numpy as np
from camera import Camera


class ObjectDetector:
    """
    A class for detecting various objects (pucks and bases) using a YOLOv8 model.
    """

    def __init__(self, model_path='models/best.pt'):
        """
        Initializes the ObjectDetector with a YOLOv8 model.

        Args:
            model_path (str): The file path to your custom-trained YOLOv8n model weights (.pt file).
        """
        try:
            self.model = YOLO(model_path)
            print("ObjectDetector initialized successfully.")
        except Exception as e:
            print(f"Error loading the YOLO model from {model_path}: {e}")
            self.model = None

    def detect_objects(self, image: np.ndarray):
        """
        Performs inference on an image to detect all specified objects.

        Args:
            image (np.ndarray): The input image as a NumPy array (e.g., from OpenCV).

        Returns:
            list: A list of detection results.
        """
        if self.model is None:
            print("Model is not loaded. Cannot perform detection.")
            return []

        results = self.model(image)
        return results

    def capture_and_detect(self, camera: Camera):
        """
        Captures a photo from the camera and performs object detection.

        Args:
            camera (Camera): An instance of the Camera class.

        Returns:
            tuple: A tuple containing the raw frame and the detection results, or (None, None) on failure.
        """
        frame = camera.read_frame()
        if frame is None:
            return None, None

        results = self.detect_objects(frame)
        return frame, results

    def get_object_info(self, results: list):
        """
        Extracts information (class, center, and bounding box) for all detected objects.

        Args:
            results (list): The list of detection results from detect_objects().

        Returns:
            dict: A dictionary mapping class names to lists of object info.
                  Each list contains dictionaries with 'center' and 'bbox' keys.
        """
        all_objects = {
            'Red_Pucks': [],
            'Blue_Pucks': [],
            'Red_Base': [],
            'Blue_Base': []
        }

        if not results or not results[0].boxes:
            return all_objects

        for box in results[0].boxes:
            class_id = box.cls.item()
            class_name = self.model.names[class_id]

            x_min, y_min, x_max, y_max = [int(coord) for coord in box.xyxy[0]]
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)

            object_info = {
                'center': (center_x, center_y),
                'bbox': (x_min, y_min, x_max, y_max)
            }

            # Use more robust string checking to categorize objects
            lower_class_name = class_name.lower()
            if 'blue' in lower_class_name and 'puck' in lower_class_name:
                all_objects['Blue_Pucks'].append(object_info)
            elif 'red' in lower_class_name and 'puck' in lower_class_name:
                all_objects['Red_Pucks'].append(object_info)
            elif 'blue' in lower_class_name and 'base' in lower_class_name:
                all_objects['Blue_Base'].append(object_info)
            elif 'red' in lower_class_name and 'base' in lower_class_name:
                all_objects['Red_Base'].append(object_info)

        return all_objects

    def draw_detections(self, image: np.ndarray, results: list):
        """
        Draws the detected bounding boxes and labels on the image.

        Args:
            image (np.ndarray): The original image.
            results (list): The list of detection results from detect_objects().

        Returns:
            np.ndarray: The image with detections drawn on it.
        """
        if not results:
            return image

        annotated_image = results[0].plot()
        return annotated_image

    def live_demo(self, camera: Camera, delay: int = 10):
        """
        Captures photos continuously from a camera, performs detection,
        and displays the results in a live window.
        Press 'q' to exit the live demo.

        Args:
            camera (Camera): An instance of the Camera class.
            delay (int): The delay in milliseconds between each frame capture.
        """
        try:
            while True:
                frame = camera.read_frame()
                if frame is None:
                    break

                results = self.detect_objects(frame)
                object_info = self.get_object_info(results)

                # Annotate image with bounding boxes
                annotated_image = self.draw_detections(frame, results)

                # Draw center points and print info for all objects
                for class_name, objects in object_info.items():
                    color = (0, 0, 0)
                    if 'Blue' in class_name:
                        color = (255, 0, 0)  # Blue
                    elif 'Red' in class_name:
                        color = (0, 0, 255)  # Red

                    for obj in objects:
                        x, y = obj['center']
                        cv2.circle(annotated_image, (x, y), 5, color, -1)

                # Print summary to console
                summary_str = " | ".join(
                    [f"{key}: {[obj['center'] for obj in value]}" for key, value in object_info.items()]
                )
                print(summary_str)

                # Display the annotated image
                cv2.imshow("Object Detector Live Demo", annotated_image)

                # Wait for key press; 'q' will break the loop
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"An error occurred during the live demo: {e}")
        finally:
            cv2.destroyAllWindows()
            camera.release()