import json
import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from helper_function import rle_to_mask

# Set GPU Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Function to process JSON files into masks
def process_json_one_hot(file_paths):
    masks = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
        for entry in data:
            temp = {}
            image_path = entry['image'].split("-")[-1].lower()
            rle = entry['tag'][0]['rle']
            original_width = 224
            original_height = 224
            try:
                mask = rle_to_mask(rle, original_width, original_height)
            except ValueError:
                continue
            one_hot_mask = np.stack([1 - mask, mask], axis=-1)
            temp["image_path"] = image_path
            temp["mask"] = one_hot_mask
            masks.append(temp)
    return masks

# File paths for the JSON files
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

# Load dataset images
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

# Combine images and masks
def combine_images_and_masks(train_images, masks):
    combined_list = []
    seen_paths = set()
    for img_dict in train_images:
        for mask_dict in masks:
            if img_dict["image_path"] == mask_dict["image_path"]:
                if img_dict["image_path"] not in seen_paths:
                    combined_dict = {
                        "image_path": img_dict["image_path"],
                        "image": img_dict["image"],
                        "mask": mask_dict["mask"]
                    }
                    combined_list.append(combined_dict)
                    seen_paths.add(img_dict["image_path"])
                break
    return combined_list

combined_list = combine_images_and_masks(train_images, masks)

# Separate images and masks
def separate_images_and_masks(combined_list):
    train_images = []
    train_masks = []
    for entry in combined_list:
        train_images.append(entry["image"])
        train_masks.append(entry["mask"])
    return train_images, train_masks

images, masks = separate_images_and_masks(combined_list)
images = np.array(images).astype('float32')
masks = np.array(masks).astype('float32')

# Train-test split
train_images, temp_images, train_masks, temp_masks = train_test_split(
    images, masks, test_size=0.3, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(
    temp_images, temp_masks, test_size=0.5, random_state=42)

# Preprocess input
BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)
train_images = preprocess_input(train_images)
val_images = preprocess_input(val_images)
test_images = preprocess_input(test_images)

# Load the trained model
model = tf.keras.models.load_model('h5_models/checkpoints/unet_resnet50/epoch_100.h5', compile=False)

# Evaluate predictions
# Function to calculate Jaccard loss
def calculate_jaccard_loss(true_masks, pred_masks):
    # Use Jaccard loss formula: 1 - IoU
    intersection = np.sum(true_masks * pred_masks, axis=(1, 2))
    union = np.sum(true_masks + pred_masks, axis=(1, 2)) - intersection
    iou_scores = intersection / (union + 1e-7)
    return np.mean(1 - iou_scores)


# Function to evaluate predictions
def evaluate_predictions(images, masks, dataset_name):
    # Predict on the dataset
    predictions = model.predict(images, batch_size=8)

    # Convert predictions to one-hot encoded masks
    predicted_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(masks, axis=-1)

    # Reshape masks to match (batch_size, height, width, channels)
    true_classes = np.expand_dims(true_classes, axis=-1)
    predicted_classes = np.expand_dims(predicted_classes, axis=-1)

    # Calculate IoU using the segmentation_models metric
    iou_metric = sm.metrics.IOUScore()
    iou_score = iou_metric(masks, predictions).numpy()

    # Calculate Jaccard loss
    jaccard_loss = calculate_jaccard_loss(true_classes, predicted_classes)

    print(f"{dataset_name} Loss: {jaccard_loss:.4f}")
    print(f"{dataset_name} IoU: {iou_score:.4f}")
    return jaccard_loss, iou_score


# Evaluate the datasets
print("Evaluating Training Data:")
train_jaccard_loss, train_iou = evaluate_predictions(train_images, train_masks, "Training")

print("\nEvaluating Validation Data:")
val_jaccard_loss, val_iou = evaluate_predictions(val_images, val_masks, "Validation")

print("\nEvaluating Testing Data:")
test_jaccard_loss, test_iou = evaluate_predictions(test_images, test_masks, "Testing")