import os
import json
import random
from datasets import Dataset, DatasetDict, Features, Value, Sequence, ClassLabel


# MAPPING CLASSES
def load_annotations(annotation_file):
    """Load COCO-style annotations from a JSON file."""
    with open(annotation_file, "r") as f:
        return json.load(f)

def extract_class_mapping(coco_annotations):
    """
    Extract class mappings for Garbage, Paper, and Plastic Bags.
    """
    fixed_mapping = {1: "Garbage Bag", 2: "Paper Bag", 3: "Plastic Bag"}  # Asegurar nombres correctos

    class_ids = sorted(set(
        obj["category_id"]
        for obj in coco_annotations["annotations"]
        if obj["category_id"] in fixed_mapping
    ))
    
    class_mapping = {cat: i for i, cat in enumerate(class_ids)}
    class_names = [fixed_mapping[cat] for cat in class_ids]
    return class_mapping, class_names

def split_dataset(coco_annotations, train_ratio=0.83, val_ratio=0.15, test_ratio=0.02, seed=1337):
    """Split the dataset into train, validation, and test sets."""
    assert train_ratio + val_ratio + test_ratio == 1, "Splits must sum to 1."
    random.seed(seed)
    random.shuffle(coco_annotations["images"])
    
    total = len(coco_annotations["images"])
    train_idx = int(total * train_ratio)
    val_idx = train_idx + int(total * val_ratio)
    
    return {
        "train": coco_annotations["images"][:train_idx],
        "validation": coco_annotations["images"][train_idx:val_idx],
        "test": coco_annotations["images"][val_idx:]
    }

def create_hf_dataset(coco_data, annotations, image_root, class_mapping, class_names):
    """Converts COCO-style annotations to a Hugging Face Dataset."""
    formatted_data = []
    ann_map = {ann["image_id"]: [] for ann in annotations}
    for ann in annotations:
        if ann["category_id"] not in class_mapping:
            continue  # Ignora categorías no mapeadas
        ann_map[ann["image_id"]].append({
            "id": ann["id"],
            "bbox": ann["bbox"],
            "category": class_mapping[ann["category_id"]]
        })
    
    for sample in coco_data:
        image_path = os.path.join(image_root, sample["file_name"])
        formatted_data.append({
            "image_id": sample["id"],
            "image": image_path,
            "width": sample["width"],
            "height": sample["height"],
            "objects": ann_map.get(sample["id"], [])
        })
    
    features = Features({
        "image_id": Value("int64"),
        "image": Value("string"),
        "width": Value("int32"),
        "height": Value("int32"),
        "objects": Sequence({
            "id": Value("int64"),
            "bbox": Sequence(Value("float32"), length=4),
            "category": ClassLabel(names=class_names)
        })
    })
    
    return Dataset.from_list(formatted_data, features=features)

# Paths for annotations and images
annotation_file = "BagImages/coco_instances.json"
image_root = "BagImages/images_raw"

# Load annotations
coco_annotations = load_annotations(annotation_file)

# Extract class mapping
class_mapping, class_names = extract_class_mapping(coco_annotations)
print("Class mapping:", class_mapping)
print("Class names:", class_names)

# Split dataset
dataset_splits = split_dataset(coco_annotations)

# Convert to Hugging Face DatasetDict
hf_dataset = DatasetDict({
    "train": create_hf_dataset(dataset_splits["train"], coco_annotations["annotations"], image_root, class_mapping, class_names),
    "test": create_hf_dataset(dataset_splits["test"], coco_annotations["annotations"], image_root, class_mapping, class_names),
    "validation": create_hf_dataset(dataset_splits["validation"], coco_annotations["annotations"], image_root, class_mapping, class_names),
})

# Save dataset to disk
hf_dataset.save_to_disk("plastic_paper_garbage_hf")

annotations = hf_dataset["train"][1]["objects"]
categories = hf_dataset["train"].features["objects"].feature["category"].names

id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

# Print dataset structure
print(hf_dataset["train"].features)

# PREPROCESS DATA
from transformers import AutoImageProcessor, ConditionalDetrImageProcessor, ConditionalDetrFeatureExtractor
from PIL import Image  # <-- Importar Image de PIL
import numpy as np
IMAGE_SIZE = 480
MODEL_NAME = "microsoft/conditional-detr-resnet-50"
MAX_SIZE = IMAGE_SIZE

image_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    do_resize=True,
    size = {"max_height": MAX_SIZE, "max_width": MAX_SIZE},
)

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForObjectDetection
from tqdm import tqdm

# Initialize the feature extractor and DETR model
# (Assumes that 'image_processor', 'id2label', and 'label2id' are defined)
feature_extractor = image_processor
model = AutoModelForObjectDetection.from_pretrained(
    "toukapy/detr_domain_shift",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Define base paths for input images and for output images (with drawn detections)
output_base = "runs/inference/detr2"

# Load a default font for drawing text
font = ImageFont.load_default()

# Colors for each class
class_colors = {
    "Garbage Bag": "red",
    "Paper Bag": "blue",
    "Plastic Bag": "green"
}

for sample in tqdm(hf_dataset["test"]):
    input_path = sample["image"] 
    out_file = os.path.join(output_base, os.path.basename(input_path))

    os.makedirs(output_base, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    orig_width, orig_height = image.size

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([[orig_height, orig_width]])
    results = feature_extractor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]

    draw = ImageDraw.Draw(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.tolist()
        xmin, ymin, xmax, ymax = map(int, box)
        class_name = model.config.id2label[label.item()]
        score_val = round(score.item(), 3)
        text = f"{class_name}: {score_val}"

        box_color = class_colors.get(class_name, "white") 

        draw.rectangle((xmin, ymin, xmax, ymax), outline=box_color, width=2)
        
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        draw.rectangle((xmin, ymin, text_bbox[2], text_bbox[3]), fill=box_color)

        draw.text((xmin, ymin), text, fill="white", font=font)

    image.save(out_file)
