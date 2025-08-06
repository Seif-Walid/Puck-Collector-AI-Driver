import cv2
import numpy as np


class Camera:
    def __init__(self, camera_index=0):
        """
        Initializes the camera object using OpenCV.

        Args:
            camera_index (int): The index of the camera to use (e.g., 0 for the default camera).
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError("Could not open video device")

    def read_frame(self) -> np.ndarray | None:
        """
        Captures a single frame from the camera.

        Returns:
            np.ndarray | None: The captured frame as a NumPy array or None if the frame could not be read.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            return None
        return frame

    def release(self):
        """Releases the camera."""
        self.cap.release()
        print("Camera released.")

    def __del__(self):
        """Ensures the camera is released when the object is destroyed."""
        if self.cap.isOpened():
            self.release()
