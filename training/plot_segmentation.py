import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os
import glob
from helper_function import rle_to_mask
import segmentation_models as sm
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('h5_models/checkpoints/unet_resnet50/epoch_100.h5', compile=False)
BACKBONE = "resnet50"

# Specify image paths to plot
image_paths_to_plot = [
    "124_4.jpg",
    "056_1.jpg",
    "245_2.jpg"
    # "001_2.jpg",
    # "001_3.jpg",
    # "001_4.jpg"
]

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
                mask = rle_to_mask(
                    rle,
                    original_width,
                    original_height
                )
            except ValueError:
                continue

            one_hot_mask = np.stack([1 - mask, mask], axis=-1)

            temp["image_path"] = image_path
            temp["mask"] = one_hot_mask
            masks.append(temp)

    return masks

def combine_images_and_masks(train_images, masks):
    combined_list = []
    seen_paths = set()
    duplicate_count = 0

    for img_dict in train_images:
        matched = False
        for mask_dict in masks:
            if img_dict["image_path"] == mask_dict["image_path"]:
                if img_dict["image_path"] in seen_paths:
                    duplicate_count += 1
                else:
                    combined_dict = {
                        "image_path": img_dict["image_path"],
                        "image": img_dict["image"],
                        "mask": mask_dict["mask"]
                    }
                    combined_list.append(combined_dict)
                    seen_paths.add(img_dict["image_path"])
                matched = True
                break

    print(f"Total duplicates detected: {duplicate_count}")
    return combined_list

def apply_mask_on_image(image, mask):
    """
    Apply the mask to the image such that only the regions corresponding to the mask
    remain visible (e.g., teeth), and all other areas are black.
    """
    # Ensure the mask is binary (0 or 1)
    mask_binary = (mask > 0).astype(np.uint8)

    # Scale the mask to match the image intensity
    mask_scaled = mask_binary * 255

    # Convert the mask to 3 channels to match the image's RGB channels
    mask_rgb = cv2.cvtColor(mask_scaled, cv2.COLOR_GRAY2RGB)

    # Apply the mask by multiplying the image with the mask
    masked_image = cv2.bitwise_and(image, mask_rgb)

    return masked_image

def plot_results_by_paths(data, image_paths):
    available_paths = {item['image_path'] for item in data}
    valid_paths = [img_path for img_path in image_paths if img_path in available_paths]

    if not valid_paths:
        print("No valid image paths found in the dataset.")
        return

    fig, axes = plt.subplots(len(valid_paths), 4, figsize=(10, 5 * len(valid_paths)))

    for i, img_path in enumerate(valid_paths):
        entry = next(item for item in data if item['image_path'] == img_path)
        image = entry['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ground_truth = entry['mask'][:, :, 1]
        pred_mask = entry['prediction']

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Real Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(ground_truth, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title("Predicted Label")
        axes[i, 2].axis("off")

        blended_image = apply_mask_on_image(image, pred_mask)
        axes[i, 3].imshow(blended_image)
        axes[i, 3].set_title("Masked Real Image")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()

    # Notify user about missing paths
    missing_paths = set(image_paths) - available_paths
    if missing_paths:
        print(f"Warning: The following image paths were not found in the dataset: {missing_paths}")

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

dataset_path = "../dataset_lengkap"
masks = process_json_one_hot(file_paths)

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

data = combine_images_and_masks(train_images, masks)

# Make Prediction
# Make image array
image2predict = [data["image"] for data in data]
image2predict = np.array(image2predict)

# preprocess input
preprocess_input = sm.get_preprocessing(BACKBONE)
image2predict = preprocess_input(image2predict)

# Make Prediction
predicted_masks = model.predict(image2predict)
binary_masks = np.argmax(predicted_masks, axis=-1)
# print(binary_masks)

# Combine data with prediction
data_plot =[]
for data, prediction in zip(data, binary_masks):
    data["prediction"] = prediction
    data_plot.append(data)

plot_results_by_paths(data_plot, image_paths_to_plot)
