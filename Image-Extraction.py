import os
import pandas as pd
import numpy as np
from PIL import Image

# Read the CSV file
df = pd.read_csv('C:\\Users\\gokul\\Desktop\\Python\\fashion-mnist_train.csv')

# Create a directory to save the images
output_dir = 'fashion_mnist_images'
os.makedirs(output_dir, exist_ok=True)

# Process the data
label_column = df.columns[0]  # Assuming the first column is the label
pixels = df.drop(label_column, axis=1).values
labels = df[label_column].values
num_images = len(df)
img_height = 28
img_width = 28

# Convert and save images
for i in range(num_images):
    # Reshape pixel values to 28x28 image
    img_array = np.reshape(pixels[i], (img_height, img_width))

    # Convert to image
    img = Image.fromarray(img_array.astype('uint8'), 'L')  # 'L' mode for grayscale

    # Save image with label as filename
    filename = f"{labels[i]}_image{i + 1}.jpg"
    img.save(os.path.join(output_dir, filename))

    if (i + 1) % 1000 == 0:
        print(f'Saved {i + 1} images.')

print(f'Conversion complete. Total {num_images} images saved.')
