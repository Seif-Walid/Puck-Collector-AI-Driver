import cv2
from model import ObjectDetector
from camera import Camera

if __name__ == '__main__':
    # Initialize the ObjectDetector class.
    object_detector = ObjectDetector(model_path='models/best.pt')

    # Check if the model was loaded successfully.
    if object_detector.model is None:
        print("Exiting due to model loading error.")
    else:
        try:
            # Create a camera object using the default webcam (index 0).
            camera = Camera(camera_index=0)
            print("Starting live object detection demo. Press 'q' to quit.")

            # Run the live demo. The delay is set to 10ms.
            # This method will handle the entire loop, including display and exiting on 'q'.
            object_detector.live_demo(camera, delay=10)

        except IOError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            # Ensure all resources are released, even if an error occurs.
            if 'camera' in locals():
                camera.release()