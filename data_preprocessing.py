import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # Fixed import

# Dataset path (use raw string or double backslashes)
dataset_path = r"C:\Users\dell\Downloads\fingerprint_blood_group_detection\dataset"

image_size = 128
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

data = []
labels = []

for label, group in enumerate(blood_groups):
    group_path = os.path.join(dataset_path, group)
    print(f"Loading images from: {group_path}")

    # Check if there are any files in the folder
    files_in_group = os.listdir(group_path)  

    if not files_in_group:
        print(f"No images found in: {group_path}")
        continue  # Skip to next blood group

    for img_file in files_in_group:
        img_path = os.path.join(group_path, img_file)
        print(f"Processing image: {img_path}")

        # Read and process the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        else:
            print(f"Successfully loaded image: {img_path}")

        img = cv2.resize(img, (image_size, image_size))
        img = np.array(img) / 255.0  

        data.append(img)
        labels.append(label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# One-hot encode the labels
labels = to_categorical(labels, num_classes=len(blood_groups))  # Fixed typo from `lables`

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Print sizes
print(f"Total images: {len(data)}")
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
