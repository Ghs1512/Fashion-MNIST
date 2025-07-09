from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import numpy as np
import os
import tqdm

# Load pre-trained InceptionV3 model without top classification layers
base_model = InceptionV3(weights='imagenet', include_top=False)

# Function to extract features using InceptionV3 model
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(299, 299))  # InceptionV3 requires 299x299 input
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

# File paths for saving features and labels incrementally
features_file = "D:\\ASEB\\AI\\Iv3_features_pls_work.npy"
labels_file = "D:\\ASEB\\AI\\Iv3_labels_pls_work.npy"

# Initialize files for incremental saving
with open(features_file, 'wb') as f_feat, open(labels_file, 'wb') as f_labels:
    memory_limit = 10 * 1024**3  # 10 GiB in bytes
    batch_features = []
    batch_labels = []
    current_memory = 0

    # Loop through each class folder
    for class_label, class_name in enumerate(sorted(os.listdir(dataset_path))):
        class_folder = os.path.join(dataset_path, class_name)

        # Check if the class folder is a directory
        if not os.path.isdir(class_folder):
            print(f"Skipping {class_folder}, not a directory.")
            continue

        # Loop through each image in the class folder
        for img_file in tqdm.tqdm(os.listdir(class_folder), desc=f"Processing {class_name}"):
            img_path = os.path.join(class_folder, img_file)

            # Check if the image path is a file before processing
            if not os.path.isfile(img_path):
                continue

            # Extract features
            img_features = extract_features(img_path)
            if img_features is not None:
                batch_features.append(img_features)
                batch_labels.append(class_label)
                current_memory += img_features.nbytes + np.array(class_label).nbytes

                # Check memory usage and save if limit is reached
                if current_memory >= memory_limit:
                    # Save current batch to disk
                    np.save(f_feat, np.array(batch_features))
                    np.save(f_labels, np.array(batch_labels))

                    # Clear memory
                    batch_features = []
                    batch_labels = []
                    current_memory = 0

    # Save any remaining data after processing all images
    if batch_features:
        np.save(f_feat, np.array(batch_features))
        np.save(f_labels, np.array(batch_labels))

print("Feature extraction and saving completed.")

