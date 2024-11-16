import os

def rename_files_in_directory(directory):
    """
    Renames each file in the specified directory by:
    - Replacing spaces with underscores.
    - Removing parentheses.
    - Converting the filename to lowercase.

    Parameters:
    - directory (str): Path to the directory containing files to be renamed.
    """
    for filename in os.listdir(directory):
        # Create the new filename
        new_filename = filename.replace(" ", "_").replace("(", "").replace(")", "").lower()

        # Construct full file paths
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)

        # Rename the file if the new name is different
        if old_file_path != new_file_path:
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: "{filename}" to "{new_filename}"')

# Usage
directory = "../../dataset_share/Normal"
rename_files_in_directory(directory)
