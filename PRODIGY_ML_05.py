import tf as tf
from tf.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tf.keras.applications import ResNet50 # type: ignore
from tf.keras import layers, models
from tf.keras.optimizers import Adam
import numpy as np
import os

# Set image size and paths
IMAGE_SIZE = (224, 224)  # Size that ResNet50 expects
TRAIN_DIR = '/path/to/train'  # Directory containing the training images
VALIDATION_DIR = '/path/to/validation'  # Directory containing validation images

# Load the ResNet50 model with pre-trained weights (no top layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
base_model.trainable = False

# Add custom layers on top for food classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(101, activation='softmax')  # 101 classes (Food-101 dataset)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the trained model
model.save('food_classification_model.h5')
