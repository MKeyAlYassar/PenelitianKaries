import tensorflow as tf
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('h5_models/segmentasi_gigi_unet_vgg19_augment.h5')

# Directory for the Karies folder
karies_folder = "../dataset_660/Karies"


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

# Predict masks for all images in the Karies folder
predicted_masks = model.predict(karies_images)


# Function to apply mask on the image (blending)
def apply_mask_on_image(image, mask):
    mask = np.argmax(mask, axis=-1)  # Get class indices
    mask = (mask * 255).astype(np.uint8)  # Scale to 0-255

    # Convert the mask to 3 channels to blend with RGB image
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Apply mask (with transparency) onto the image
    blended = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)  # 0.7 for original image, 0.3 for mask transparency

    return blended


# Function to plot image and predicted mask pairs, along with image + mask overlay
def plot_predictions(images, predicted_masks, num_examples=3):
    # Randomly pick 3 indices
    indices = random.sample(range(len(images)), num_examples)

    plt.figure(figsize=(7, num_examples * 3))  # Adjusted for smaller plot window

    for i, idx in enumerate(indices):
        # Input image
        plt.subplot(num_examples, 3, i * 3 + 1)
        plt.imshow(images[idx])
        plt.title(f"Input Image {os.path.basename(image_paths[idx])}")
        plt.axis('off')

        # Predicted mask
        predicted_mask = np.argmax(predicted_masks[idx], axis=-1)  # Get class indices
        predicted_mask = (predicted_mask * 255).astype(np.uint8)  # Scale to 0-255
        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        # Image with mask applied
        blended_image = apply_mask_on_image(images[idx], predicted_masks[idx])
        plt.subplot(num_examples, 3, i * 3 + 3)
        plt.imshow(blended_image)
        plt.title("Image + Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Main loop to continuously display 3 random pairs of images and masks until interrupted
def display_predictions_with_loop():
    try:
        while True:
            # Plot 3 random predictions
            plot_predictions(karies_images, predicted_masks, num_examples=3)
    except KeyboardInterrupt:
        print("Loop interrupted, exiting.")


# Run the display loop
display_predictions_with_loop()
