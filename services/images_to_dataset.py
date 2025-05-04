import os
import shutil
import random
from pathlib import Path

# Configuration
input_root = 'SpaceTimeShapes_images'
output_root = '_dataset'
ratios = (0.7, 0.15, 0.15)  # train, val, test

# Create output folders
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_root, split), exist_ok=True)

# Get sorted class names and assign indices
class_names = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])
class_to_index = {cls: idx for idx, cls in enumerate(class_names)}

# Save class_index.txt
with open(os.path.join(output_root, 'class_index.txt'), 'w') as f:
    for cls, idx in class_to_index.items():
        f.write(f"{idx} {cls}\n")

# Gather all image paths and labels
image_label_list = []
for class_name in class_names:
    class_dir = os.path.join(input_root, class_name)
    for video_folder in os.listdir(class_dir):
        video_dir = os.path.join(class_dir, video_folder)
        if not os.path.isdir(video_dir):
            continue
        for img_name in os.listdir(video_dir):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(video_dir, img_name)
                image_label_list.append((img_path, class_to_index[class_name]))

# Shuffle and split
random.shuffle(image_label_list)
total = len(image_label_list)
train_end = int(total * ratios[0])
val_end = train_end + int(total * ratios[1])

splits_data = {
    'train': image_label_list[:train_end],
    'val': image_label_list[train_end:val_end],
    'test': image_label_list[val_end:]
}

# Save images and labels
for split in splits:
    split_dir = os.path.join(output_root, split)
    label_file = open(os.path.join(output_root, f"{split}_labels.txt"), "w")

    for img_path, class_idx in splits_data[split]:
        img_filename = f"{Path(img_path).stem}_{random.randint(0, 99999)}.jpg"
        dest_path = os.path.join(split_dir, img_filename)
        shutil.copy(img_path, dest_path)
        label_file.write(f"{img_filename} {class_idx}\n")

    label_file.close()
    print(f"{split.upper()}: {len(splits_data[split])} images")
