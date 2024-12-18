import argparse
import os
import cv2
import numpy as np

def read_image_bgr(img_path: str):
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    img = cv2.imread(img_path)
    return img


def get_median_values_from_list(list_indicies: int) -> list[int]:
    final_indecies = []
    for all_index in list_indicies:
        medain_val = np.median(all_index)
        final_indecies.append(int(medain_val))

    return final_indecies

def get_column_index(processed_gray_img, width: int) -> list[int]: 
    col_idices = []
    set_of_cols = []
    get_index = False
    for i in range(width):
        output = np.all(processed_gray_img[:, i] == 255)
        
        if output:
            get_index = True
            set_of_cols.append(i)
        else:
            if get_index:
                col_idices.append(set_of_cols)
                set_of_cols = []
                get_index = False

    return get_median_values_from_list(col_idices)


def get_row_index(processed_gray_img, height: int) -> list[int]:
    row_idices = []
    set_of_rows = []
    get_index = False
    for i in range(height):
        output = np.all(processed_gray_img[i, :] == 255)
        if output:
            get_index = True
            set_of_rows.append(i)
        else:
            if get_index:
                row_idices.append(set_of_rows)
                set_of_rows = []
                get_index = False\
                
    return get_median_values_from_list(row_idices)


def cut_image(image, row_indicies: list[int], column_indicies:list[int]):
    prev_row_index = 0
    prev_col_index = 0
    cut_images = []
    for row_index in row_indicies:
        img_row = image[prev_row_index: row_index, :, :]
        prev_row_index = row_index
        for col_index in column_indicies:
            new_image = img_row[: , prev_col_index: col_index, :]
            cut_images.append(new_image)
            prev_col_index = col_index

        prev_col_index = 0
    return cut_images


def perform_threshold_on_img(image_path: str, output_dir_path:str):
    bgr_img = read_image_bgr(image_path)
    gray_img = cv2.cvtColor(bgr_img, 
                            cv2.COLOR_BGR2GRAY)
    
    processed_gray_img = gray_img.copy()
    processed_gray_img = np.where(processed_gray_img > threshold, 
                                  255, 
                                  0)
    processed_img_path = os.path.join(output_dir_path, 
                                       "processed_img.jpg")
    cv2.imwrite(processed_img_path,
                processed_gray_img)
    
    return processed_gray_img


if __name__ == "__main__":
    try:
        # Create the argument parser
        parser = argparse.ArgumentParser(description="Script to Cut a source image to separate price tag images")
        
        # Add the `src_image_path` argument
        parser.add_argument(
            'src_image_path',  # Positional argument
            type=str,
            help='Path to the source image (required)'
        )
        
        # Parse the arguments
        args = parser.parse_args()
        
        # Read the provided argument
        src_image_path = args.src_image_path
        print(f"Source Image Path: {src_image_path}")

        # threshold for pixel values
        threshold = 240
        output_folder_path = os.path.join(os.getcwd(), "output")
        output_image_name_format = "image_{}.jpg"

        # Create a output folder
        os.makedirs(output_folder_path, exist_ok=True)

        # Read the image and convert to bgr for processsing
        bgr_img = read_image_bgr(src_image_path)
        # Covert all the values in the image as 0 / 255
        gray_img = perform_threshold_on_img(src_image_path, output_folder_path)
        height, width = gray_img.shape

        # get the columns and rows there are no text in the entire row/columns of pixels
        # (All the pixel values are 0)
        col_indices = get_column_index(gray_img, width=width)
        row_indices = get_row_index(gray_img, height=height)

        # Remove the initial column index and row index as its not needed since the 
        # image boundary itself is sufficient
        col_indices = col_indices[1: ]
        row_indices = row_indices[1: ]

        # Add width and height for slicing purpose
        col_indices.append(width)
        row_indices.append(height)

        # Cut the image based on the columns and rows
        images_list = cut_image(bgr_img, row_indices, col_indices)
        print(f"{len(images_list)} Price tag/s found from {src_image_path}")

        # Save the images to respective_files
        for index, img in enumerate(images_list, start=1):
            image_path = os.path.join(output_folder_path, 
                                    output_image_name_format.format(index))
            cv2.imwrite(image_path,
                        img)
    
    except Exception as error:
        print(f"Error Encountered while processing the image : {error}")
    else:
        print(f"Please find all the images in \"{output_folder_path}\" Folder")
    finally:
        print("Processing Complete!!!")