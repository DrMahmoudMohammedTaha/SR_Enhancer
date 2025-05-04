import os
import cv2

# Paths
input_root = 'SpaceTimeShapes_videos'
output_root = 'SpaceTimeShapes_images'

# Create output root directory if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# Walk through each class folder
for class_name in os.listdir(input_root):
    class_path = os.path.join(input_root, class_name)
    if not os.path.isdir(class_path):
        continue

    for video_file in os.listdir(class_path):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_path = os.path.join(class_path, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_root, class_name, video_name)
        os.makedirs(output_dir, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames from {video_file} to {output_dir}")
