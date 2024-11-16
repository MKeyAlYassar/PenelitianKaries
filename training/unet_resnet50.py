import json

from helper_function import rle_to_mask
import os
import glob
import cv2

import numpy as np
from tensorflow.keras.utils import Sequence
import cv2
import tensorflow as tf
import pickle
import sklearn
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.callbacks import *
from tensorflow.keras.applications import ResNet50
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Set GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def process_json(file_paths):
    """
    Reads multiple JSON files and processes each image and its RLE mask.

    Parameters:
    - file_paths (list of str): List of JSON file paths to be processed.

    Returns:
    - masks (list of dict): A list of dictionaries, each containing 'image_path' and 'mask'.
    """
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

            # Convert RLE to mask
            mask = rle_to_mask(
                rle,
                original_width,
                original_height
            )

            temp["image_path"] = image_path
            temp["mask"] = mask
            masks.append(temp)

    return masks


file_paths = ["../labeling/project-1-at-2024-11-07-22-50-7c64a4fb.json",
              "../labeling/normal_100.json",
              "../labeling/normal_50.json"]

masks = process_json(file_paths)

dataset_path = "../dataset_660"

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


def combine_images_and_masks(train_images, masks):
    combined_list = []

    for img_dict in train_images:
        # Find the corresponding mask dictionary where image_path matches
        for mask_dict in masks:
            if img_dict["image_path"] == mask_dict["image_path"]:
                # Combine the two dictionaries
                combined_dict = {
                    "image_path": img_dict["image_path"],
                    "image": img_dict["image"],
                    "mask": mask_dict["mask"]
                }
                combined_list.append(combined_dict)
                break  # No need to continue searching once matched

    return combined_list


combined_list = combine_images_and_masks(train_images, masks)

def separate_images_and_masks(combined_list):
    train_images = []
    train_masks = []

    for entry in combined_list:
        # Append image to train_images and mask to train_masks
        train_images.append(entry["image"])
        train_masks.append(entry["mask"])

    return train_images, train_masks

images, masks = separate_images_and_masks(combined_list)

images = np.array(images)
masks = np.array(masks)

train_images, temp_images, train_masks, temp_masks = train_test_split(images, masks, test_size=0.3, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(temp_images, temp_masks, test_size=0.5, random_state=42)

print("Train Images", train_images.shape)
print("Train Masks:", train_masks.shape)
print("Validation Images", val_images.shape)
print("Validation Masks:", val_masks.shape)
print("Test Images", test_images.shape)
print("Test Masks:", test_masks.shape)

def decoder_block(x, y, filters):
    x = UpSampling2D()(x)
    x = Concatenate(axis = 3)([x,y])
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x

def resnet50_unet(input_shape, *, classes, dropout):
    """ Input """
    inputs = Input(input_shape)

    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output
    s2 = resnet50.get_layer("conv1_relu").output
    s3 = resnet50.get_layer("conv2_block3_out").output
    s4 = resnet50.get_layer("conv3_block4_out").output

    x = resnet50.get_layer("conv4_block6_out").output

    """ Decoder """
    x = decoder_block(x, s4, 512)
    x = decoder_block(x, s3, 256)
    x = decoder_block(x, s2, 128)
    x = decoder_block(x, s1, 64)

    x = Dropout(dropout)(x)
    outputs = Conv2D(classes, 1, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ResNet50_U-Net")
    return model


inp_size = (224, 224, 3)
model = resnet50_unet(inp_size, classes=2, dropout=0.3)

model.compile(loss= tf.keras.losses.SparseCategoricalCrossentropy(),
             optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
             metrics= ['accuracy'],
             run_eagerly= True)

# Fit Model
history = model.fit(train_images,
                    train_masks,
                    validation_data=(val_images, val_masks),
                    batch_size=8,
                    epochs=100,
                    verbose=1)

model.save(f'h5_models/segmentasi_gigi_unet_resnet50.h5')

print("training_finish")

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
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

def plot_image_mask_prediction(images, real_masks, predicted_masks, num_examples=3):
    """
    Plot a few samples of the input image, real mask, and predicted mask.
    """
    plt.figure(figsize=(6, num_examples * 3))

    for i in range(num_examples):
        # Input image
        plt.subplot(num_examples, 3, i * 3 + 1)
        plt.imshow(images[i])
        plt.title("Input Image")
        plt.axis('off')

        # Real mask
        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(real_masks[i], cmap='gray')
        plt.title("Real Mask")
        plt.axis('off')

        # Predicted mask
        predicted_mask = np.argmax(predicted_masks[i], axis=-1)  # Get class indices
        predicted_mask = (predicted_mask * 255).astype(np.uint8)  # Scale to 0-255

        plt.subplot(num_examples, 3, i * 3 + 3)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Predict masks for the training set
predicted_masks = model.predict(train_images)

# Plot 3 pairs of image, real mask, and predicted mask
plot_image_mask_prediction(train_images, train_masks, predicted_masks, num_examples=3)