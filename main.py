import cv2
from model import PuckDetector
from camera import Camera

if __name__ == '__main__':
    # Initialize the PuckDetector
    puck_detector = PuckDetector(model_path='models/best.pt')

    try:
        # Create a camera object
        camera = Camera(camera_index=0)  # Change index if needed

        print("Capturing photo and detecting pucks...")
        frame, detections = puck_detector.capture_and_detect(camera)

        if frame is not None and detections and detections[0].boxes:
            puck_centers = puck_detector.get_puck_centers(detections)

            print("\nPuck Centers:")
            print(f"Blue pucks: {puck_centers['blue_pucks']}")
            print(f"Red pucks: {puck_centers['red_pucks']}")

            # Draw detections and centers on the image
            annotated_image = puck_detector.draw_detections(frame, detections)
            for x, y in puck_centers['blue_pucks']:
                cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
            for x, y in puck_centers['red_pucks']:
                cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)

            cv2.imshow("Puck Detections", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No pucks detected or failed to capture a frame.")

    except IOError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # It's good practice to ensure the camera is released
        if 'camera' in locals():
            camera.release()
