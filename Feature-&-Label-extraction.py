import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input

# Load pre-trained VGG16 model without top classification layers
base_model = VGG16(weights='imagenet', include_top=False)

# Function to extract features using VGG16 model
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features.flatten()

# Path to the main dataset folder
dataset_path = "C:\\Users\\gokul\\Desktop\\Python\\test"

# Initialize empty lists to store features and labels
features = []
labels = []

# Loop through each class folder
for class_label, class_name in enumerate(sorted(os.listdir(dataset_path))):
    class_folder = os.path.join(dataset_path, class_name)

    # Loop through each image in the class folder
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)

        # Extract features and append to the list
        img_features = extract_features(img_path)
        features.append(img_features)
        labels.append(class_label)
    print("Features and labels saved successfully for the folder.")

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Save the extracted features and labels
np.save("C:\\Users\\gokul\\Desktop\\Python\\features_og.npy", X)
np.save("C:\\Users\\gokul\\Desktop\\Python\\labels_og.npy", y)

print("Features and labels saved successfully.")
