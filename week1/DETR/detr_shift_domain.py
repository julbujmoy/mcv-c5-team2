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

import albumentations as A

train_augment_and_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
)

validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
)

def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }



def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    result = None  # Inicializamos result

    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        try:
            image = np.array(Image.open(image).convert("RGB"))

            # Calcular 'area' si no está presente
            if "area" not in objects:
                objects["area"] = [w * h for _, _, w, h in objects["bbox"]]  # Calcular área a partir de bbox

            # Aplicar augmentaciones
            output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
            images.append(output["image"])

            # Formatear anotaciones en COCO format
            formatted_annotations = format_image_annotations_as_coco(
                image_id, output["category"], objects["area"], output["bboxes"]
            )
            annotations.append(formatted_annotations)

        except Exception as e:
            print(f"Error procesando la imagen {image_id}: {e}")  # Mensaje de depuración
            continue

    if images:
        result = image_processor(images=images, annotations=annotations, return_tensors="pt")

        if not return_pixel_mask:
            result.pop("pixel_mask", None)
    else:
        print("No se procesaron imágenes correctamente, devolviendo None.")

    return result




from functools import partial

# Make transform functions for batch and apply for dataset splits
train_transform_batch = partial(
    augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
)
validation_transform_batch = partial(
    augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
)

#print(hf_dataset["train"][0])

hf_dataset["train"] = hf_dataset["train"].with_transform(train_transform_batch)
hf_dataset["validation"] = hf_dataset["validation"].with_transform(validation_transform_batch)
hf_dataset["test"] = hf_dataset["test"].with_transform(validation_transform_batch)

#print(hf_dataset["train"][0])

import torch

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data

# FUNCTION TO COMPUTE mAP
from transformers.image_transforms import center_to_corners_format

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """

    print(boxes)
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])


    return boxes
    
import numpy as np
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


eval_compute_metrics_fn = partial(
    compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
)
# TRAINING MODEL
from transformers import AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained(
    MODEL_NAME,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="detr_finetuned_kitti_mots-horizon",
    num_train_epochs=50,
    fp16=False,
    per_device_train_batch_size=8,
    dataloader_num_workers=4,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    weight_decay=1e-3,
    max_grad_norm=0.01,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    push_to_hub=True,
    hub_token="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"
)

from transformers import Trainer

# 2) Create the Trainer (the frozen parameters won't update during training)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset["train"],
    eval_dataset=hf_dataset["validation"],
    processing_class=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

# 3) Train, updating only the head
trainer.train()

trainer.push_to_hub()

from pprint import pprint

metrics = trainer.evaluate(eval_dataset=hf_dataset["validation"], metric_key_prefix="val")
pprint(metrics)