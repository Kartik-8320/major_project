import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "external_dataset/sign_images"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# CREATE TRAIN DATA FIRST
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# 🔥 SAVE CLASS ORDER HERE (IMPORTANT)
os.makedirs("models", exist_ok=True)
with open("models/class_names.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ Class indices saved:", train_data.class_indices)

# BUILD MODEL
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# SAVE MODEL IN COMPATIBLE FORMAT
model.save("models/sign_language_cnn_tf213.keras")

print("✅ Model trained and saved successfully")
