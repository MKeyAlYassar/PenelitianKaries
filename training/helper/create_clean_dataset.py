import os
import glob
import cv2
import numpy as np

folder_path = "../../dataset_lengkap"
output_path = "../../clean_dataset"
input_size = (224, 224)

def load_and_save_images(folder_path, input_size, output_path):
    """
    Load images from the specified folder, remove duplicates and images with incorrect shapes,
    then save valid images to a new path.

    Parameters:
        folder_path (str): Path to the folder containing "Normal" and "Karies" subfolders.
        input_size (tuple): Expected size of the images (height, width).
        output_path (str): Path to save the filtered images.

    Returns:
        int: Number of valid images saved.
    """
    seen_files = set()  # Track seen files to detect duplicates
    duplicate_files = set()  # Track duplicate files to skip both

    # Identify duplicates
    for subfolder_name in ["Normal", "Karies"]:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        for img_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
            file_name = os.path.basename(img_path).lower()
            if file_name in seen_files:
                duplicate_files.add(file_name)
            seen_files.add(file_name)

    # Process and save valid images
    valid_image_count = 0
    for subfolder_name in ["Normal", "Karies"]:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        for img_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
            file_name = os.path.basename(img_path)

            # Skip duplicates
            if file_name in duplicate_files:
                continue

            # Load and validate image
            img = cv2.imread(img_path)
            if img is None or img.shape[:2] != input_size:
                continue

            # Save the valid image to the new path
            save_subfolder = os.path.join(output_path, subfolder_name)
            os.makedirs(save_subfolder, exist_ok=True)
            save_path = os.path.join(save_subfolder, file_name)
            cv2.imwrite(save_path, img)

            valid_image_count += 1

    return valid_image_count

valid_images = load_and_save_images(folder_path, input_size, output_path)
print(f"Saved {valid_images} valid images to '{output_path}'.")