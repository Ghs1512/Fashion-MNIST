from keras.preprocessing import image
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.models import Model
import numpy as np
import os
import tqdm

# Load pre-trained EfficientNetB0 model without top classification layers
base_model = EfficientNetB0(weights='imagenet', include_top=False)

# Function to extract features using EfficientNetB0 model
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # EfficientNetB0 requires 224x224 input
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = base_model.predict(x)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Path to the main dataset folder
dataset_path = "D:\\ASEB\\AI\\AI_image_data_pls_work"

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    raise ValueError(f"Dataset path {dataset_path} does not exist.")

# Initialize empty lists to store features and labels
features = []
labels = []

# Loop through each class folder
for class_label, class_name in enumerate(sorted(os.listdir(dataset_path))):
    class_folder = os.path.join(dataset_path, class_name)

    # Check if the class folder path exists and is a directory
    if not os.path.isdir(class_folder):
        print(f"Skipping {class_folder}, not a directory.")
        continue

    # Loop through each image in the class folder
    for img_file in tqdm.tqdm(os.listdir(class_folder), desc=f"Processing {class_name}"):
        img_path = os.path.join(class_folder, img_file)

        # Check if the img_path is a file before processing
        if not os.path.isfile(img_path):
            continue

        # Extract features and append to the list
        img_features = extract_features(img_path)
        if img_features is not None:
            features.append(img_features)
            labels.append(class_label)

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Save the extracted features and labels
np.save("D:\\ASEB\\AI\\ENb0_features_pls_work.npy", X)
np.save("D:\\ASEB\\AI\\ENb0_labels_pls_work.npy", y)
