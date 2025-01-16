import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from classification_models.tfkeras import Classifiers
import matplotlib.pyplot as plt
import glob
from tabulate import tabulate

# Parameters
BACKBONES = ["densenet121", "resnet50", "mobilenetv2", "inceptionv3", "resnext50"]  # Add more backbones as needed
SEGMENT_BACKBONES = ["no", "vgg19"]  # Add more segmentation backbones as needed
RANDOM_STATE = 42  # Consistent random state

# Initialize results storage
results = []

# Loop through all backbone combinations
for backbone in BACKBONES:
    for seg_backbone in SEGMENT_BACKBONES:
        print(f"Processing Backbone: {backbone}, Segmentation Backbone: {seg_backbone}...")

        # Paths
        DATASET_PATH = "../clean_dataset" if seg_backbone == "no" else f"../masked_dataset/{seg_backbone}"
        MODEL_PATH = f"h5_models/classification/{backbone}/best_val_loss_{backbone}_{seg_backbone}_lr5eneg05_SGD.h5"

        # Load pretrained model for preprocessing
        try:
            _, preprocess_input = Classifiers.get(backbone)
        except Exception as e:
            print(f"Error loading backbone {backbone}: {e}")
            continue

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

        images = np.array([data["image"] for data in dataset])
        labels = np.array([data["class"] for data in dataset])

        # Split dataset
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            images, labels, test_size=0.3, random_state=RANDOM_STATE)
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=0.5, random_state=RANDOM_STATE)

        # Preprocess data
        train_images = preprocess_input(train_images)
        val_images = preprocess_input(val_images)
        test_images = preprocess_input(test_images)

        # Load the trained model
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=True)
        except Exception as e:
            print(f"Error loading model {MODEL_PATH}: {e}")
            continue

        # Evaluate the model
        train_loss, train_accuracy = model.evaluate(train_images, train_labels, verbose=0)
        val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

        # Predict on test dataset
        test_predictions = model.predict(test_images)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_true_classes = np.argmax(test_labels, axis=1)

        # Generate metrics
        precision = precision_score(test_true_classes, test_pred_classes, average='weighted')
        recall = recall_score(test_true_classes, test_pred_classes, average='weighted')
        f1 = f1_score(test_true_classes, test_pred_classes, average='weighted')

        # Save confusion matrix
        conf_matrix = confusion_matrix(test_true_classes, test_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Normal", "Karies"])
        output_dir = "confusion_matrices"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{MODEL_PATH.split('/')[-1].split('.')[0]}"
        save_path = os.path.join(output_dir, f"{filename}.png")
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Oranges, ax=ax)
        plt.title(f"Confusion Matrix: {backbone}-{seg_backbone}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Store results
        results.append({
            "Backbone": backbone,
            "Segment Backbone": seg_backbone,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        })

# Display results in tabular format
headers = ["Backbone", "Segment", "Train Loss", "Val Loss", "Test Loss",
           "Train Acc", "Val Acc", "Test Acc", "Recall", "Precision", "F1-Score"]

rows = [
    [r["Backbone"], r["Segment Backbone"], f"{r['Train Loss']:.4f}", f"{r['Validation Loss']:.4f}",
     f"{r['Test Loss']:.4f}", f"{r['Train Accuracy']:.4f}", f"{r['Validation Accuracy']:.4f}",
     f"{r['Test Accuracy']:.4f}", f"{r['Recall']:.4f}", f"{r['Precision']:.4f}", f"{r['F1-Score']:.4f}"]
    for r in results
]
print("\nResults Summary:")
print(tabulate(rows, headers=headers, tablefmt="grid"))
