import os
import cv2
import numpy as np
import glob


def horizontal_flip(images):
    """
    Augment the images by applying a horizontal flip.
    Returns the augmented images.
    """
    # Apply horizontal flip to all images
    flipped_images = [cv2.flip(image, 1) for image in images]
    return flipped_images


def augment_and_save_images(dataset_path, output_path):
    """
    Load images from the dataset, augment them with a horizontal flip,
    and save the augmented images to a new directory.
    """
    # Ensure output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process subfolders (Normal and Karies)
    for subfolder_name in ["Normal", "Karies"]:
        subfolder_path = os.path.join(dataset_path, subfolder_name)
        output_subfolder_path = os.path.join(output_path, subfolder_name)

        # Create subfolder in output path if it doesn't exist
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)

        # Process all images in the subfolder
        for img_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
            # Load image
            image = cv2.imread(img_path)

            # Apply horizontal flip
            flipped_image = horizontal_flip([image])[0]

            # Save augmented image to the new directory
            # Use the same file name but in the new directory
            file_name = os.path.basename(img_path)
            save_path = os.path.join(output_subfolder_path, file_name)
            cv2.imwrite(save_path, flipped_image)


# Paths
DATASET_PATH = "../../clean_dataset"
OUTPUT_PATH = "../../augmented_dataset"

# Run the augmentation and saving process
augment_and_save_images(DATASET_PATH, OUTPUT_PATH)

print("Augmentation and saving completed!")
