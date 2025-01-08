import os
import cv2
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tqdm import tqdm
import glob

# Ensure GPU memory growth is enabled
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set backbone
BACKBONE = 'vgg19'
MODEL_PATH = f"h5_models/checkpoints/unet_{BACKBONE}/epoch_100.h5"
IMAGE_INPUT_SIZE = (224, 224)

# Paths
DATASET_PATH = "../augmented_dataset"
OUTPUT_PATH = f"../augmented_masked_dataset/{BACKBONE}"

# Load trained model
model = tf.keras.models.load_model(f'h5_models/checkpoints/unet_{BACKBONE}/epoch_100.h5', compile=False)

# Preprocessing function for the selected backbone
preprocess_input = sm.get_preprocessing(BACKBONE)

# Function to load images from a folder
def load_images(folder_path, input_size):
    images = []
    filenames = []
    seen_files = set()  # Track seen files
    duplicate_files = set()  # Track duplicate files to skip them completely

    for subfolder_name in ["Normal", "Karies"]:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        for img_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
            # Extract filename
            file_name = os.path.basename(img_path).lower()

            if file_name in seen_files:
                # Mark as duplicate and continue
                duplicate_files.add(file_name)
                print(f"Duplicate detected: {file_name}")
                continue
            seen_files.add(file_name)

    # Reload and process files, skipping duplicates and invalid files
    for subfolder_name in ["Normal", "Karies"]:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        for img_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
            file_name = os.path.basename(img_path)

            # Skip duplicates
            if file_name in duplicate_files:
                print(f"Skipping duplicate file: {file_name}")
                continue

            # Load and validate image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Invalid image skipped: {img_path}")
                continue

            if img.shape[:2] != input_size:  # Validate shape
                print(f"Image with incorrect shape skipped: {file_name} (shape: {img.shape[:2]})")
                continue

            images.append(img)
            filenames.append((subfolder_name, file_name))  # Store subfolder and file name

    return np.array(images), filenames

# Load images
print("Loading images...")
images, filenames = load_images(DATASET_PATH, IMAGE_INPUT_SIZE)

# Preprocess images
print("Preprocessing images...")
images_preprocessed = preprocess_input(images)

# Predict masks
print("Segmenting images...")
predicted_masks = model.predict(images_preprocessed, batch_size=8)

# Post-process masks: Extract masks and apply them
def apply_mask_and_save(images, masks, filenames, output_folder):
    for i, (image, mask, (subfolder, filename)) in tqdm(enumerate(zip(images, masks, filenames))):
        # Convert mask to binary
        mask_binary = np.argmax(mask, axis=-1).astype(np.uint8)  # Shape: (H, W)

        # Apply mask to image
        mask_applied = cv2.bitwise_and(image, image, mask=mask_binary)

        # Ensure output folder structure
        output_subfolder = os.path.join(output_folder, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        # Save the masked image
        output_path = os.path.join(output_subfolder, filename)
        cv2.imwrite(output_path, mask_applied)

# Apply the predicted mask and save images
print("Applying masks and saving images...")
apply_mask_and_save(images, predicted_masks, filenames, OUTPUT_PATH)

print("Segmentation complete. Results saved to:", OUTPUT_PATH)