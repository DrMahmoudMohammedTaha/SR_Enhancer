import os

# ✅ Set your labels folder path here
labels_folder = "C:\\Users\\Cloud Tech\\Downloads\\++unified_dataset\\+far_human\\valid\\labels"

def replace_class_id_with_2(labels_folder, new_class_id):
    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_folder, filename)

            # Read and process lines
            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = new_class_id  # Change class ID to new_class_id
                    new_lines.append(" ".join(parts) + "\n")

            # Overwrite file with modified lines
            with open(file_path, "w") as f:
                f.writelines(new_lines)

            print(f"Updated: {filename}")

    print(f"\nAll class IDs have been set to {new_class_id}.")

# 🔧 Run the replacement
replace_class_id_with_2(labels_folder , "1")
