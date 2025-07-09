import os
import shutil


source_dir = "C:\\Users\\gokul\\Desktop\\Python\\fashion_mnist_images"
dest_dir = "C:\\Users\\gokul\\Desktop\\Python\\Seg_images"
images_per_dir = 1000



def create_subdirectories(root_dir, total_images, images_per_dir):
    num_dirs = total_images // images_per_dir
    if total_images % images_per_dir != 0:
        num_dirs += 1

    for i in range(num_dirs):
        os.makedirs(os.path.join(root_dir, f'batch_{i+1}'), exist_ok=True)

def move_images_to_subdirectories(source_dir, dest_dir, images_per_dir):
    image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    total_images = len(image_files)


    create_subdirectories(dest_dir, total_images, images_per_dir)
    current_dir = 1
    image_count = 0

    for image_file in image_files:
        if image_count == images_per_dir:
            current_dir += 1
            image_count = 0

        src_path = os.path.join(source_dir, image_file)
        dest_path = os.path.join(dest_dir, f'batch_{current_dir}', image_file)

        shutil.move(src_path, dest_path)

        image_count += 1




#Ensure destination directory is clean
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
os.makedirs(dest_dir, exist_ok=True)

move_images_to_subdirectories(source_dir, dest_dir,images_per_dir)
