import tensorflow as tf
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import segmentation_models as sm
import sys
# np.set_printoptions(threshold=sys.maxsize)

# Load the saved model
model = tf.keras.models.load_model('h5_models/checkpoints/unet_resnet50/epoch_100.h5', compile=False)

# Directory for the Karies folder
karies_folder = "../dataset_lengkap/Karies"

# Load all the images from the Karies folder and resize them
def load_images_from_folder(folder, img_size=(224, 224)):
    images = []
    image_paths = []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display
                images.append(img)
                image_paths.append(img_path)

    return np.array(images), image_paths


# Load images
karies_images, image_paths = load_images_from_folder(karies_folder)
karies_images = np.array(karies_images)

BACKBONE = 'vgg19'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
train_images = preprocess_input(karies_images)

# Predict masks for all images in the Karies folder
predicted_masks = model.predict(karies_images)
binary_masks = np.argmax(predicted_masks, axis=-1)

# for mask in binary_masks:
#     print(np.unique(mask))
# print(binary_masks[-1])
# print(np.unique(binary_masks[-1]))

# print(predicted_masks.shape)
# print(predicted_masks[32])

def apply_mask_on_image(image, mask):
    mask = (mask * 255).astype(np.uint8)  # Scale to 0-255

    # Convert the mask to 3 channels to blend with RGB image
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Apply mask (with transparency) onto the image
    blended = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)  # 0.7 for original image, 0.3 for mask transparency

    return blended

def display_random_predictions_loop(images, predicted_masks, num_examples=3):
    try:
        while True:
            # Randomly pick `num_examples` indices
            indices = random.sample(range(len(images)), num_examples)

            plt.figure(figsize=(6, num_examples * 2))  # Adjust size for better display

            for i, idx in enumerate(indices):
                # Plot the real image
                plt.subplot(num_examples, 3, i * 3 + 1)
                plt.imshow(images[idx])
                plt.title(f"Input Image {idx}")
                plt.axis('off')

                # Plot the binary mask
                plt.subplot(num_examples, 3, i * 3 + 2)
                plt.imshow(predicted_masks[idx], cmap='gray')
                plt.title(f"Predicted Mask {idx}")
                plt.axis('off')

                # Plot the image with the mask applied
                blended_image = apply_mask_on_image(images[idx], predicted_masks[idx])
                plt.subplot(num_examples, 3, i * 3 + 3)
                plt.imshow(blended_image)
                plt.title("Image + Mask")
                plt.axis('off')

            plt.tight_layout()
            plt.show()

    except KeyboardInterrupt:
        print("Exited the loop.")

# Usage
# Assuming `karies_images` is your dataset of images and `predicted_masks` is your predictions
display_random_predictions_loop(karies_images, binary_masks, num_examples=3)