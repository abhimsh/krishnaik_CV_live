import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Split the image into B, G, and R channels
        b, g, r = cv2.split(frame)

        # Create a larger image to hold all channels horizontally
        height, width, _ = frame.shape
        combined_image = np.zeros((height, width * 3, 3), dtype=np.uint8)

        # Place the channels horizontally
        combined_image[:, :width] = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
        combined_image[:, width:2*width] = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
        combined_image[:, 2*width:] = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])

        # Display the combined image
        cv2.imshow("Combined Frame", combined_image)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()