import os
import json

def list_images_in_json(file_paths):
    """
    Lists image paths specified in multiple JSON files.

    Parameters:
    - file_paths (list of str): List of JSON file paths to be processed.

    Returns:
    - image_paths (set): A set containing image paths listed in the JSON files.
    """
    image_paths = set()

    # Iterate over each JSON file to collect image paths
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)

        for entry in data:
            # Extract image path from the JSON entry
            image_path = entry['image'].split("-")[-1].lower()
            image_paths.add(image_path)

    return image_paths


def remove_listed_images_from_directory(image_paths, directory):
    """
    Removes images from the specified directory if they are in the provided image list.

    Parameters:
    - image_paths (set): A set of image file names to be removed.
    - directory (str): Path to the directory where images are located.

    Returns:
    - removed_images (list): List of images that were successfully removed.
    """
    removed_images = []

    # Iterate over each file in the directory
    for file_name in os.listdir(directory):
        if file_name in image_paths:
            file_path = os.path.join(directory, file_name)
            os.remove(file_path)
            removed_images.append(file_name)

    return removed_images


# Example usage:
file_paths = [
    "../project-1-at-2024-11-15-11-14-9ed6a173.json",
]

directory = "../../dataset_share/Karies"

# Step 1: List images specified in JSON files
image_paths = list_images_in_json(file_paths)

print(len(image_paths))

# Step 2: Remove listed images from the specified directory
removed_images = remove_listed_images_from_directory(image_paths, directory)

print("Removed images:", removed_images)
print(len(removed_images))
