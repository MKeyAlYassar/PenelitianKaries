import os
import cv2
import numpy as np
import tensorflow as tf
from classification_models.tfkeras import Classifiers
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import time


# Ensure GPU memory growth is enabled
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Hyperparameter configurations
hyperparams = [
    {"BACKBONE": "resnet50", "SEGMENT_BACKBONE": "no", "LEARNING_RATE": 0.0003, "OPTIMIZER": "SGD"},
    {"BACKBONE": "resnet50", "SEGMENT_BACKBONE": "resnet50", "LEARNING_RATE": 0.0003, "OPTIMIZER": "SGD"},

    {"BACKBONE": "inceptionv3", "SEGMENT_BACKBONE": "no", "LEARNING_RATE": 0.0003, "OPTIMIZER": "SGD"},
    {"BACKBONE": "inceptionv3", "SEGMENT_BACKBONE": "resnet50", "LEARNING_RATE": 0.0003, "OPTIMIZER": "SGD"},

    {"BACKBONE": "mobilenetv2", "SEGMENT_BACKBONE": "no", "LEARNING_RATE": 0.0003, "OPTIMIZER": "SGD"},
    {"BACKBONE": "mobilenetv2", "SEGMENT_BACKBONE": "resnet50", "LEARNING_RATE": 0.0003, "OPTIMIZER": "SGD"},

    {"BACKBONE": "densenet121", "SEGMENT_BACKBONE": "no", "LEARNING_RATE": 0.0003, "OPTIMIZER": "SGD"},
    {"BACKBONE": "densenet121", "SEGMENT_BACKBONE": "resnet50", "LEARNING_RATE": 0.0003, "OPTIMIZER": "SGD"},

    {"BACKBONE": "resnext50", "SEGMENT_BACKBONE": "no", "LEARNING_RATE": 0.0003, "OPTIMIZER": "SGD"},
    {"BACKBONE": "resnext50", "SEGMENT_BACKBONE": "resnet50", "LEARNING_RATE": 0.0003, "OPTIMIZER": "SGD"},
]

# Dataset loading and preprocessing
def load_and_preprocess_data(DATASET_PATH, SEGMENT_BACKBONE, preprocess_input):
    dataset = []

    for path in DATASET_PATH:
        for subfolder_name in ["Normal", "Karies"]:
            subfolder_path = os.path.join(path, subfolder_name)
            for img_path in glob.glob(os.path.join(subfolder_path, "*.jpg")):
                temp = {}
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))

                temp["image"] = img
                temp["class"] = [1, 0] if subfolder_name == "Normal" else [0, 1]
                dataset.append(temp)

    images = np.array([data["image"] for data in dataset])
    labels = np.array([data["class"] for data in dataset])

    train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.3, random_state=42)
    val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=42)

    train_images = preprocess_input(train_images)
    val_images = preprocess_input(val_images)
    test_images = preprocess_input(test_images)

    # Check Data Shape
    print("Train Images:", train_images.shape)
    print("Train Labels:", train_labels.shape)
    print("Validation Images:", val_images.shape)
    print("Validation Labels:", val_labels.shape)
    print("Test Images:", test_images.shape)
    print("Test Labels:", test_labels.shape)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

# Augment Training Images
def augment_data(train_images, train_labels):
    from helper.augment_functions import rotate_180
    augmented_images = rotate_180(train_images)
    augmented_labels = np.concatenate((train_labels, train_labels), axis=0)
    return augmented_images, augmented_labels

# Plot Training History
def plot_training_history(history, save_path):
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
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

# Main training loop
for config in hyperparams:
    BACKBONE = config["BACKBONE"]
    SEGMENT_BACKBONE = config["SEGMENT_BACKBONE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    OPTIMIZER_NAME = config["OPTIMIZER"]

    if SEGMENT_BACKBONE == "no":
        DATASET_PATH = ["../clean_dataset"]
    else:
        DATASET_PATH = [f"../masked_dataset/{SEGMENT_BACKBONE}"]

    model_base, preprocess_input = Classifiers.get(BACKBONE)
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_and_preprocess_data(DATASET_PATH, SEGMENT_BACKBONE, preprocess_input)
    train_images, train_labels = augment_data(train_images, train_labels)

    if OPTIMIZER_NAME == "SGD":
        optimizer = SGD(learning_rate=LEARNING_RATE)
    elif OPTIMIZER_NAME == "Adam":
        optimizer = Adam(learning_rate=LEARNING_RATE)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")

    # Build Model
    base_model = model_base(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])

    # Compile Model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Save paths with adjusted learning rate format
    checkpoint_dir = f"h5_models/classification/{BACKBONE}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Replace dot with underscore in learning rate or use scientific notation
    lr_formatted = f"{LEARNING_RATE:.0e}".replace("-", "neg")  # Scientific notation, e.g., "1e-4" for 0.0001
    model_name = f"{BACKBONE}_{SEGMENT_BACKBONE}_lr{lr_formatted}_{OPTIMIZER_NAME}.h5"
    plot_name = f"plot/{BACKBONE}_{SEGMENT_BACKBONE}_lr{lr_formatted}_{OPTIMIZER_NAME}.png"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f'best_{model_name}'),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
    )

    # Train Model
    history = model.fit(
        x=train_images,
        y=train_labels,
        batch_size=16,
        epochs=200,
        validation_data=(val_images, val_labels),
        callbacks=[checkpoint_callback],
    )

    # Save final model and plot
    model.save(os.path.join(checkpoint_dir, model_name))
    plot_training_history(history, plot_name)

    # Pause GPU to rest for 15 minutes
    print(f"Configuration completed: {config}. Pausing GPU for 15 minutes to rest.")
    time.sleep(15 * 60)  # Pause for 15 minutes
