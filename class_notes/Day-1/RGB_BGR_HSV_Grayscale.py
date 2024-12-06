import cv2
import numpy as np

def main(grid):
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a larger image to hold all frames horizontally
        height, width, _ = frame.shape
        if grid == "horizontal":
            combined_image = np.zeros((height, width * 4, 3), dtype=np.uint8)

            # Place the frames horizontally
            combined_image[:, :width] = rgb_frame
            combined_image[:, width:2*width] = frame
            combined_image[:, 2*width:3*width] = hsv_frame
            combined_image[:, 3*width:] = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for display

        if grid == "square":
            combined_image = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

            # Place the frames in a 2x2 grid
            combined_image[:height, :width] = rgb_frame
            combined_image[height:, :width] = frame
            combined_image[:height, width:] = hsv_frame
            combined_image[height:, width:] = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Display the combined image
        cv2.imshow("Combined Frame", combined_image)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    grid = "square" #square # horizontal
    main(grid)