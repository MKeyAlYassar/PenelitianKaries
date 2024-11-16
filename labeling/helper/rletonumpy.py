from typing import List
import numpy as np
import json
from PIL import Image

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    treshold = 0

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    binary_mask = np.where(image > treshold, 255, image)
    binary_mask = np.where(binary_mask <= treshold, 0, binary_mask)
    return binary_mask

def apply_mask(image_path: str, mask: np.array):
    """
    Opens an image, applies a mask to it, and displays the masked image.

    Args:
        image_path: Path to the image file.
        mask: A NumPy array representing the mask (same dimensions as the image).
    """
    # Open the image
    image = Image.open(image_path)
    image = image.convert("RGBA")  # Convert to RGBA to handle transparency

    # Convert image to NumPy array
    image_np = np.array(image)

    # Ensure mask has the same shape as the image
    if mask.shape != image_np.shape[:2]:
        raise ValueError("Mask dimensions do not match image dimensions!")

    # Apply the mask: mask out the areas where the mask is 0 (set those areas to transparent)
    masked_image_np = image_np.copy()
    masked_image_np[:, :, 3] = np.where(mask == 0, 0, masked_image_np[:, :, 3])  # Modify alpha channel

    # Convert the NumPy array back to an image
    masked_image = Image.fromarray(masked_image_np)

    # Display the masked image
    masked_image.show()

def process_json(file_path):
    """Reads the JSON file and processes each image and its RLE mask."""
    with open(file_path, 'r') as file:
        data = json.load(file)

    for entry in data:
        # Extract image path and RLE data
        image_path = entry['image'].split("-")[-1]
        rle = entry['tag'][0]['rle']  # Assuming there's at least one tag
        original_width = 224  # Set this according to your mask's dimensions
        original_height = 224  # Set this according to your mask's dimensions

        mask = rle_to_mask(
            rle,
            original_width,
            original_height
        )

        print(image_path)
        image_path = "../../dataset_660/Karies/061 (4).jpg"

        apply_mask(image_path, mask)
        break


process_json("../project-1-at-2024-10-23-11-50-316b0a6b.json")