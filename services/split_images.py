import os
import shutil
import random

def split_images(
    src_folder, 
    output_folder,
    train_ratio=0.7, 
    valid_ratio=0.2, 
    test_ratio=0.1,
    image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")
):
    
    # Create output subfolders
    train_dir = os.path.join(output_folder, 'train')
    valid_dir = os.path.join(output_folder, 'valid')
    test_dir  = os.path.join(output_folder, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all image files
    all_images = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith(image_extensions)
    ]
    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(train_ratio * total)
    valid_end = train_end + int(valid_ratio * total)

    train_images = all_images[:train_end]
    valid_images = all_images[train_end:valid_end]
    test_images  = all_images[valid_end:]

    # Copy images
    for img in train_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(train_dir, img))
    for img in valid_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(valid_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(test_dir, img))

    print(f"Total images: {total}")
    print(f"Train: {len(train_images)} | Valid: {len(valid_images)} | Test: {len(test_images)}")
    print(f"Images copied to: {output_folder}")

# Example usage:
split_images(
    src_folder="D:\\_research\\pedestrian_datasets\\++far_human\\_cropped_far_human",
    output_folder="D:\\_research\\pedestrian_datasets\\++far_human\\_dataset",
    train_ratio=0.7,
    valid_ratio=0.2,
    test_ratio=0.1
)
