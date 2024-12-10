import cv2
import numpy as np

def apply_sobel_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Sobel gradients in x and y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Combine the gradients to get the magnitude
    sobelxy = cv2.magnitude(sobelx, sobely)

    # Convert the result to 8-bit for display
    sobelxy = cv2.convertScaleAbs(sobelxy)

    return sobelxy

def apply_canny_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def apply_gaussian_blur(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    return blurred

def apply_laplacian_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.absolute(laplacian)

def apply_median_filter(frame):
    median = cv2.medianBlur(frame, 5)
    return median

def apply_bilateral_filter(frame):
    bilateral = cv2.bilateralFilter(frame, 9, 75, 75)
    return bilateral

def apply_unsharp_masking(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    unsharp = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
    return unsharp

def apply_histogram_equalization(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    return equ

def apply_histogram_equalization_rgb(frame):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Equalize the histogram of the V channel
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    # Convert back to BGR color space
    equ_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return equ_rgb

def apply_grayscale_conversion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def apply_hsv_conversion(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv

def display_grid(frame, filtered_frames, filter_names):
    num_filters = len(filtered_frames)
    cols = 3
    rows = num_filters // cols + 1

    # Ensure all filtered frames have the same shape and data type as the original frame
    for i in range(len(filtered_frames)):
        if filtered_frames[i].dtype == np.float64:  # 64-bit float
            filtered_frames[i] = cv2.convertScaleAbs(filtered_frames[i])  # Convert to 8-bit unsigned integer
        if len(filtered_frames[i].shape) == 2:  # Grayscale image
            filtered_frames[i] = cv2.cvtColor(filtered_frames[i], cv2.COLOR_GRAY2BGR)
        elif filtered_frames[i].shape != frame.shape:
            filtered_frames[i] = cv2.resize(filtered_frames[i], (frame.shape[1], frame.shape[0]))

    # Create a blank image to hold the grid
    grid_img = np.zeros((rows * frame.shape[0], cols * frame.shape[1], 3), dtype=np.uint8)

    # Fill the grid with filtered frames
    for i, (filtered_frame, filter_name) in enumerate(zip(filtered_frames, filter_names)):
        row = i // cols
        col = i % cols
        grid_img[row * frame.shape[0]:(row + 1) * frame.shape[0],
                 col * frame.shape[1]:(col + 1) * frame.shape[1]] = filtered_frame
        cv2.putText(grid_img, filter_name, (col * frame.shape[1], row * frame.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Resize the grid image if it's too large
    scale_factor = 0.5  # Adjust this factor as needed
    grid_img = cv2.resize(grid_img, None, fx=scale_factor, fy=scale_factor)

    cv2.imshow('Filtered Frames', grid_img)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Apply filters
        filtered_frames = [
            frame,
            apply_sobel_filter(frame),
            apply_canny_edge_detection(frame),
            apply_gaussian_blur(frame),
            apply_laplacian_filter(frame),
            apply_median_filter(frame),
            apply_bilateral_filter(frame),
            apply_unsharp_masking(frame),
            apply_histogram_equalization(frame),
            apply_histogram_equalization_rgb(frame),
            apply_grayscale_conversion(frame),
            apply_hsv_conversion(frame),
        ]

        filter_names = [
            "Original",
            "Sobel Filter",
            "Canny Edge",
            "Gaussian Blur",
            "Laplacian",
            "Median",
            "Bilateral",
            "Unsharp Masking",
            "Histogram Equalization",
            "Histogram Equalization RGB",
            "GrayScale",
            "HSV"
        ]

        # Display the filtered frames in a grid
        display_grid(frame, filtered_frames, filter_names)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()