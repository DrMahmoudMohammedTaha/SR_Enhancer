import os

# ✅ Set your folder paths here
images_dir = "C:\\Users\\Cloud Tech\\Downloads\\++unified_dataset\\+far_drone\\valid\\images"
labels_dir = "C:\\Users\\Cloud Tech\\Downloads\\++unified_dataset\\+far_drone\\valid\\labels"

def clean_orphan_labels(images_dir, labels_dir):
    image_basenames = {os.path.splitext(f)[0] for f in os.listdir(images_dir)
                       if os.path.isfile(os.path.join(images_dir, f))}

    deleted_files = 0

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        label_name = os.path.splitext(label_file)[0]
        if label_name not in image_basenames:
            os.remove(os.path.join(labels_dir, label_file))
            print(f"Deleted orphan label: {label_file}")
            deleted_files += 1

    print(f"\nDone. Deleted {deleted_files} orphan label files.")

# 🔧 Run the cleanup
clean_orphan_labels(images_dir, labels_dir)
