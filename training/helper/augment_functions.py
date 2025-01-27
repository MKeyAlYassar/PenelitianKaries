import numpy as np
import cv2
import matplotlib.pyplot as plt

def horizontal_flip(images, masks=None):
    """
    Augment the images and masks by applying a horizontal flip.
    Returns the augmented images and masks, effectively doubling the dataset size.
    """
    # Apply horizontal flip to all images and masks
    flipped_images = [cv2.flip(image, 1) for image in images]
    # flipped_masks = [cv2.flip(mask, 1) for mask in masks]

    # Combine original and flipped data
    augmented_images = np.concatenate((images, flipped_images), axis=0)
    # augmented_masks = np.concatenate((masks, flipped_masks), axis=0)

    return augmented_images

def rotate_180(images, masks=None):
    """
    Augment the images and masks by applying a 180-degree rotation.
    Returns the augmented images and masks, effectively doubling the dataset size.
    """
    # Apply 180-degree rotation to all images and masks
    rotated_images = [cv2.rotate(image, cv2.ROTATE_180) for image in images]
    rotated_masks = [cv2.rotate(mask, cv2.ROTATE_180) for mask in masks] if masks is not None else None

    # Combine original and rotated data
    augmented_images = np.concatenate((images, rotated_images), axis=0)
    augmented_masks = np.concatenate((masks, rotated_masks), axis=0) if masks is not None else None

    if masks is not None:
        return augmented_images, augmented_masks
    elif masks is None:
        return augmented_images

def plot_augmentation_examples(original_images, original_masks, augmented_images, augmented_masks, num_examples=3):
    """
    Plot examples of original and augmented (flipped) images and masks side by side.
    """
    plt.figure(figsize=(6, num_examples * 4))

    for i in range(num_examples):
        # Original image and mask
        plt.subplot(num_examples, 4, i * 4 + 1)
        plt.imshow(original_images[i])
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(num_examples, 4, i * 4 + 2)
        plt.imshow(original_masks[i], cmap='gray')
        plt.title("Original Mask")
        plt.axis('off')

        # Augmented (flipped) image and mask
        plt.subplot(num_examples, 4, i * 4 + 3)
        plt.imshow(augmented_images[i + len(original_images)])
        plt.title("Flipped Image")
        plt.axis('off')

        plt.subplot(num_examples, 4, i * 4 + 4)
        plt.imshow(augmented_masks[i + len(original_masks)], cmap='gray')
        plt.title("Flipped Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.show()