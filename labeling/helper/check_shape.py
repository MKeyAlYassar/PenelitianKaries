import os
import cv2
import glob

dataset_path = "../../dataset_lengkap"
train_images = []

for subfolder_name in ["Normal", "Karies"]:
    subfolder_path = os.path.join(dataset_path, subfolder_name)

    for img_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
        temp = {}
        img_path_save = img_path.split("\\")[-1].replace(" ", "_").replace("(", "").replace(")", "").lower()

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Check if the image has the correct shape
        if img.shape != (224, 224, 3):
            print(f"Image at path '{img_path}' has an unexpected shape: {img.shape}")
        else:
            temp["image_path"] = img_path_save
            temp["image"] = img
            train_images.append(temp)
