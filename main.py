import cv2
from model import PuckDetector
from camera import Camera

if __name__ == '__main__':
    # Initialize the PuckDetector class.
    puck_detector = PuckDetector(model_path='models/best.pt')

    # Check if the model was loaded successfully.
    if puck_detector.model is None:
        print("Exiting due to model loading error.")
    else:
        try:
            # Create a camera object using the default webcam (index 0).
            camera = Camera(camera_index=0)
            print("Starting live puck detection demo. Press 'q' to quit.")

            # Run the live demo. The delay is set to 10ms.
            # This method will handle the entire loop, including display and exiting on 'q'.
            puck_detector.live_demo(camera, delay=1000)

        except IOError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            # Ensure all resources are released, even if an error occurs.
            if 'camera' in locals():
                camera.release()
