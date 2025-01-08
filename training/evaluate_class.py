import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from classification_models.tfkeras import Classifiers
import matplotlib.pyplot as plt
import glob

# Parameters
BACKBONE = "vgg19"
SEGMENT_BACKBONE = "no"
MODEL_TYPE = "epoch200" # epoch200 or best
RANDOM_STATE = 42  # Same random state as used in training
DATASET_PATH = "../clean_dataset" if SEGMENT_BACKBONE == "no" else f"../masked_dataset/{SEGMENT_BACKBONE}"
MODEL_PATH = f"h5_models/classification/{BACKBONE}/{MODEL_TYPE}_{BACKBONE}_{SEGMENT_BACKBONE}seg_sgd.h5"

# Load pretrained model for preprocessing
_, preprocess_input = Classifiers.get(BACKBONE)

# Load and preprocess data
dataset = []
for subfolder_name in ["Normal", "Karies"]:
    subfolder_path = os.path.join(DATASET_PATH, subfolder_name)
    for img_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
        temp = {}
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        temp["image"] = img
        temp["class"] = [1, 0] if subfolder_name == "Normal" else [0, 1]
        dataset.append(temp)

images = [data["image"] for data in dataset]
labels = [data["class"] for data in dataset]

images = np.array(images)
labels = np.array(labels)

# Split dataset
train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.3, random_state=RANDOM_STATE)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=RANDOM_STATE)

train_images = preprocess_input(train_images)
val_images = preprocess_input(val_images)
test_images = preprocess_input(test_images)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Evaluate the model on train, validation, and test datasets
print("Evaluating Train Dataset")
train_loss, train_accuracy = model.evaluate(train_images, train_labels, verbose=0)
print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

print("\nEvaluating Validation Dataset")
val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

print("\nEvaluating Test Dataset")
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Predict on test dataset
test_predictions = model.predict(test_images)
test_pred_classes = np.argmax(test_predictions, axis=1)
test_true_classes = np.argmax(test_labels, axis=1)

# Plot confusion matrix
conf_matrix = confusion_matrix(test_true_classes, test_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Normal", "Karies"])
disp.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix for Test Dataset")
plt.show()

# Calculate precision, recall, and F1-score
print("\nClassification Report:")
report = classification_report(test_true_classes, test_pred_classes, target_names=["Normal", "Karies"])
print(report)
