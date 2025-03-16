import json
import random
from collections import defaultdict
out1="C:/Users/User/Documents/MASTER/c5/ImageClassesCombinedWithCOCOAnnotations/train_bags_coco.json"
out2="C:/Users/User/Documents/MASTER/c5/ImageClassesCombinedWithCOCOAnnotations/val_bags_coco.json"

import json
import random

# Load the original JSON file
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Save JSON data to a file
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Split dataset into train and test

def split_dataset(json_data, train_ratio=0.8):
    images = json_data['images']
    annotations = json_data['annotations']
    categories = json_data['categories']
    
    # Shuffle images randomly
    random.shuffle(images)
    
    # Split images into train and test
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]
    
    # Get image IDs for train and test
    train_image_ids = {img['id'] for img in train_images}
    test_image_ids = {img['id'] for img in test_images}
    
    # Split annotations accordingly
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    test_annotations = [ann for ann in annotations if ann['image_id'] in test_image_ids]
    
    # Create train and test datasets
    train_data = {'images': train_images, 'annotations': train_annotations, 'categories': categories}
    test_data = {'images': test_images, 'annotations': test_annotations, 'categories': categories}
    
    return train_data, test_data

# Load dataset
json_data = load_json(file_path)

# Split dataset
train_data, test_data = split_dataset(json_data)

# Save the split datasets
save_json(train_data, out1)
save_json(test_data, out2)

print("Dataset split completed. Train and test JSON files are saved.")