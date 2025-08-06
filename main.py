import cv2
from model import PuckDetector

if __name__ == '__main__':
    # Initialize the PuckDetector class.
    # The model path is relative to the project root.
    puck_detector = PuckDetector(model_path='models/best.pt')

    # Load a sample image for testing.
    # Make sure you have a file named 'puck_images_resized/sample_test_1_red_puck.jpg'.
    image_path = 'puck_images_resized/sample_test_2_blue_2_red.jpg'
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Error: Image not found at {image_path}")

        print("Performing inference on the test image...")
        detections = puck_detector.detect_pucks(image)

        if detections and detections[0].boxes:
            # Get the puck centers using the new method
            puck_centers = puck_detector.get_puck_centers(detections)

            print("\nPuck Centers:")
            print(f"Blue pucks: {puck_centers['blue_pucks']}")
            print(f"Red pucks: {puck_centers['red_pucks']}")

            # Draw the detections on the image
            annotated_image = puck_detector.draw_detections(image, detections)

            # Draw circles on the center of each puck for visual confirmation
            for x, y in puck_centers['blue_pucks']:
                # Draw a small blue circle at the center
                cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
            for x, y in puck_centers['red_pucks']:
                # Draw a small red circle at the center
                cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)

            # Display the result
            cv2.imshow("Puck Detections", annotated_image)
            cv2.waitKey(0)  # Wait indefinitely for a key press
            cv2.destroyAllWindows()
        else:
            print("No pucks detected in the image.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
