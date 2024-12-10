import json

from helper_function import rle_to_mask
import os
import glob
import cv2
import segmentation_models as sm

import numpy as np
import cv2
import tensorflow as tf
import sklearn
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.callbacks import *
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
# np.set_printoptions(threshold=sys.maxsize)

# Set GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def process_json_one_hot(file_paths):
    masks = []

    # Iterate over each file path in the provided list
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)

        for entry in data:
            temp = {}
            # Extract image path and RLE data
            image_path = entry['image'].split("-")[-1].lower()
            rle = entry['tag'][0]['rle']
            original_width = 224
            original_height = 224

            # Convert RLE to mask (binary mask with values 0 and 1)
            try:
                mask = rle_to_mask(
                    rle,
                    original_width,
                    original_height
                )
            except ValueError:
                continue

            # Convert binary mask to one-hot encoding
            # Assuming two classes: background (0) and foreground (1)
            one_hot_mask = np.stack([1 - mask, mask], axis=-1)

            temp["image_path"] = image_path
            temp["mask"] = one_hot_mask
            masks.append(temp)

    return masks


file_paths = [
              "../labeling/1-100.json",
              "../labeling/101-200.json",
              "../labeling/resmi_menamatkan_gambar.json",
              "../labeling/project-1-at-2024-12-10-15-09-d3d41871.json",
              "../labeling/project-1-at-2024-12-10-15-09-46e9d17f.json",
              "../labeling/project-1-at-2024-12-07-14-44-e1dc025b.json",
              "../labeling/project-3-at-2024-12-10-15-23-68e63351.json",
              "../labeling/project-3-at-2024-12-07-15-20-00f9eaeb.json"
              ]

masks = process_json_one_hot(file_paths)

dataset_path = "../dataset_lengkap"

train_images = []

for subfolder_name in ["Normal", "Karies"]:
    subfolder_path = os.path.join(dataset_path, subfolder_name)

    for img_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
        temp = {}
        img_path_save = img_path.split("\\")[-1].replace(" ", "_").replace("(", "").replace(")", "").lower()

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))

        temp["image_path"] = img_path_save
        temp["image"] = img
        train_images.append(temp)

print(f"Banyak Dataset: {len(train_images)}")
print(f"Banyak Label: {len(masks)}")

def combine_images_and_masks(train_images, masks):
    combined_list = []
    seen_paths = set()  # Keep track of processed image paths
    duplicate_count = 0  # Counter for duplicates

    for img_dict in train_images:
        # Find the corresponding mask dictionary where image_path matches
        matched = False
        for mask_dict in masks:
            if img_dict["image_path"] == mask_dict["image_path"]:
                if img_dict["image_path"] in seen_paths:
                    # print(f"Warning: Duplicate detected for image_path '{img_dict['image_path']}'")
                    duplicate_count += 1
                else:
                    # Combine the two dictionaries
                    combined_dict = {
                        "image_path": img_dict["image_path"],
                        "image": img_dict["image"],
                        "mask": mask_dict["mask"]
                    }
                    combined_list.append(combined_dict)
                    seen_paths.add(img_dict["image_path"])
                matched = True
                break
        # if not matched:
        #     print(f"Warning: No matching mask found for image_path '{img_dict['image_path']}'")

    print(f"Total duplicates detected: {duplicate_count}")
    return combined_list


combined_list = combine_images_and_masks(train_images, masks)
print(f"Banyak Combined List: {len(combined_list)}")

def separate_images_and_masks(combined_list):
    train_images = []
    train_masks = []

    for entry in combined_list:
        # Append image to train_images and mask to train_masks
        train_images.append(entry["image"])
        train_masks.append(entry["mask"])

    return train_images, train_masks

images, masks = separate_images_and_masks(combined_list)

images = np.array(images).astype('float32')
masks = np.array(masks).astype('float32')

print("All Images", images.shape)
print("All Masks:", masks.shape)

train_images, temp_images, train_masks, temp_masks = train_test_split(images, masks, test_size=0.3, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(temp_images, temp_masks, test_size=0.5, random_state=42)

# # Augment Training Images
# from augment_functions import horizontal_flip
#
# train_images, train_masks = horizontal_flip(train_images, train_masks)

# Check Data Shape
print("Train Images", train_images.shape)
print("Train Masks:", train_masks.shape)
print("Validation Images", val_images.shape)
print("Validation Masks:", val_masks.shape)
print("Test Images", test_images.shape)
print("Test Masks:", test_masks.shape)

# Initialize Backbone
BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Preprocess input
train_images = preprocess_input(train_images)
val_images = preprocess_input(val_images)

# Define model
model = sm.Unet(BACKBONE, classes=2, encoder_weights='imagenet', activation='softmax')
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# Directory to save the model checkpoints
if not os.path.exists(f"h5_models/checkpoints/unet_{BACKBONE}"):
        os.makedirs(f"h5_models/checkpoints/unet_{BACKBONE}")

checkpoint_dir = f'h5_models/checkpoints/unet_{BACKBONE}'

# Callback to save the model every 10 epochs
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, f'epoch_{{epoch:02d}}.h5'),
    save_freq='epoch',
    save_best_only=False,
    period=10  # Save every 10 epochs
)

# Training parameters
EPOCHS = 100

# Fit model with the callback
history = model.fit(
    x=train_images,
    y=train_masks,
    batch_size=8,
    epochs=EPOCHS,
    validation_data=(val_images, val_masks),
    callbacks=[checkpoint_callback],
)

print("Training Finished")

def plot_training_history(history):
    """
    Plot the training and validation loss and accuracy over epochs.
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['iou_score'], label='Training IoU')
    plt.plot(history.history['val_iou_score'], label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('IoU Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)