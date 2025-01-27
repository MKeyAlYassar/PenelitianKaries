import os
import shutil

def organize_files(target_folder, folders_to_check, folder_a, folder_b):
    """
    Organize files by checking if they exist in the folders to check and moving them
    to either folder A (if they exist in the folders to check) or folder B (if they don't).

    Args:
        target_folder (str): Path to the folder where files will be checked.
        folders_to_check (list): List of folder paths to verify the existence of files.
        folder_a (str): Path to folder A where existing files will be moved.
        folder_b (str): Path to folder B where non-existing files will be moved.
    """
    # Create folders A and B if they don't exist
    os.makedirs(folder_a, exist_ok=True)
    os.makedirs(folder_b, exist_ok=True)

    # Gather all normalized filenames from folders_to_check
    reference_files = set()
    for folder in folders_to_check:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                normalized_filename = filename.split("\\")[-1].replace(" ", "_").replace("(", "").replace(")", "").lower()
                reference_files.add(normalized_filename)

    # Check files in the target folder
    for filename in os.listdir(target_folder):
        file_path = os.path.join(target_folder, filename)

        if os.path.isfile(file_path):
            normalized_filename = filename.split("\\")[-1].replace(" ", "_").replace("(", "").replace(")", "").lower()

            # Move the file to folder A or B based on its existence in the reference set
            if normalized_filename in reference_files:
                shutil.move(file_path, os.path.join(folder_a, normalized_filename))
                print(f"Moved {filename} to {folder_a}")
            else:
                shutil.move(file_path, os.path.join(folder_b, normalized_filename))
                print(f"Moved {filename} to {folder_b}")

# Example usage
if __name__ == "__main__":
    target_folder = "../../clean_dataset-copy/Normal"  # Replace with the path to the target folder

    folders_to_check = ["../../dmft_dataset/Missing",
                        "../../dmft_dataset/Filling",
                        "../../dmft_dataset/Decay"]

    folder_a = "../../clean_datasetv2/Karies"  # Replace with the path to folder A
    folder_b = "../../clean_datasetv2/Normal"  # Replace with the path to folder B

    organize_files(target_folder, folders_to_check, folder_a, folder_b)
