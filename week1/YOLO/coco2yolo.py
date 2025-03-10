import json
import os

json_file = "kitti_mots_coco.json"
labels_dir = "KITTI-MOTS/training/labels"

# Create folder
os.makedirs(labels_dir, exist_ok=True)

# Mapping
category_map = {1: 0, 2: 1, 10: 2}  # "car" -> 0, "pedestrian" -> 1

# Load JSON
with open(json_file, "r") as f:
    annotations = json.load(f)

# Process each image
for ann in annotations:
    image_name = ann["image"].split("/")[-1].replace(".png", ".txt")
    
    folder_name = ann["image"].split("/")[0]
    folder_path = os.path.join(labels_dir, folder_name)
    
    # Create subfolder
    os.makedirs(folder_path, exist_ok=True)

    label_path = os.path.join(folder_path, image_name)

    with open(label_path, "w") as label_file:
        for obj_id, bbox, category in zip(ann["objects"]["id"], ann["objects"]["bbox"], ann["objects"]["category"]):
            # Category 10 - ignore
            if category == 10:
                continue

            x, y, w, h = bbox
            x_center = (x + w / 2) / ann["width"]
            y_center = (y + h / 2) / ann["height"]
            w_norm = w / ann["width"]
            h_norm = h / ann["height"]

            label_file.write(f"{category_map[category]} {x_center} {y_center} {w_norm} {h_norm}\n")

print("Completed. Labels saved in:", labels_dir)
