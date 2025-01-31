import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from glob import glob

# Define dataset path
dataset_path = "path_to_dataset/train"  # Update this path

# Image preprocessing settings
img_size = 64  # Resize images to 64x64

# Initialize lists for images and labels
X = []
y = []

# Load images
image_paths = glob(os.path.join(dataset_path, "*.jpg"))  # Load all .jpg files

for img_path in image_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        continue  # Skip corrupted images
    img = cv2.resize(img, (img_size, img_size))  # Resize
    X.append(img.flatten())  # Flatten image into 1D array
    label = 0 if "cat" in img_path else 1  # Extract label from filename
    y.append(label)

# Convert lists to NumPy arrays
X = np.array(X, dtype=np.float32) / 255.0  # Normalize pixel values
y = np.array(y, dtype=np.int32)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm = SVC(kernel="rbf", C=1.0, gamma="scale")  # RBF kernel often works better for images
svm.fit(X_train, y_train)

# Predict on test set
y_pred = svm.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
