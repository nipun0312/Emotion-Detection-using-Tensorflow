import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, GlobalAveragePooling2D, Input, Multiply)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np
import json

# Define constants
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0003

# Dataset paths
train_dir = "./dataset/train"
val_dir = "./dataset/test"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Load dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)

# Squeeze-and-Excitation Block (Attention Mechanism)
def se_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // reduction, activation="relu")(se)
    se = Dense(channels, activation="sigmoid")(se)
    return Multiply()([input_tensor, se])

# Build Model using EfficientNetB0 as Feature Extractor
base_model = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet")
base_model.trainable = False  # Freeze pretrained layers

input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = tf.image.grayscale_to_rgb(input_layer)  # Convert grayscale to 3-channel RGB

x = base_model(x, training=False)
x = se_block(x)  # Add SE Block for Attention

x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)

output_layer = Dense(7, activation='softmax')(x)  # 7 emotion classes

# Create final model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with Adam optimizer and Categorical Crossentropy
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

# Callbacks for better training
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_emotion_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Convert history values to JSON-serializable format
history_dict = {key: [float(i) for i in np.array(value)] for key, value in history.history.items()}

# Save training history
with open("training_history.json", "w") as f:
    json.dump(history_dict, f)

# Save the final model
model.save("improved_emotion_model.h5")

print("Training complete! Best model saved as improved_emotion_model.h5")
