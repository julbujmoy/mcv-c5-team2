import json
import os

import torch
import numpy as np
from PIL import Image
from torchvision.ops import masks_to_boxes
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection import MeanAveragePrecision
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerModel, AutoModelForUniversalSegmentation
from tqdm import tqdm
from torchvision import transforms
import pycocotools.mask as maskUtils
from torchvision.transforms.functional import resize
import torch
import numpy as np
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from functools import partial
from transformers.image_transforms import center_to_corners_format
from transformers import Trainer
from transformers import TrainingArguments, Trainer


# Load Mask2Former trained on COCO instance segmentation dataset
image_processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-coco-instance",
    use_fast=True
)

id2label = {
    0: "background",  # No object
    1: "car",
    2: "pedestrian",
}
label2id = {v: k for k, v in id2label.items()}

# Define paths
BASE_DIR = "/home/toukapy/Dokumentuak/Master CV/C5/mcv-c5-team2/data/KITTI_MOTS"
train_image_dir = os.path.join(BASE_DIR, "training/image_02/train")
val_image_dir = os.path.join(BASE_DIR, "training/image_02/val")
instances_txt_dir = os.path.join(BASE_DIR, "instances_txt")  # Folder with RLE annotations
annotation_file = "/home/toukapy/Dokumentuak/Master CV/C5/mcv-c5-team2/data/train_data_kitti_mots_coco.json"
annotation_file_val = "/home/toukapy/Dokumentuak/Master CV/C5/mcv-c5-team2/data/val_data_kitti_mots_coco.json"

# Load ground truth annotations
with open(annotation_file, "r") as f:
    annotations = json.load(f)

with open(annotation_file_val, "r") as f:
    annotations_val = json.load(f)

# Resize target to reduce memory usage
TARGET_SIZE = (256, 512)  # Reduce resolution to save memory
TARGET_MASK_SIZE = (128, 256)  # Reduce mask resolution
# Define Transform for Images
image_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),  # Resize image to smaller size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Mapping from KITTI-MOTS categories to COCO
COCO_CATEGORY_MAPPING = {
    1: 2,  # KITTI car → COCO car
    2: 0,  # KITTI pedestrian → COCO person
    10: None,  # Ignore class
}

# Maximum objects per image to avoid memory explosion
MAX_OBJECTS_PER_IMAGE = 10


# Function to parse RLE annotations
def parse_kitti_mots_annotation(annotation_line):
    elements = annotation_line.strip().split(" ")

    object_id = int(elements[0])   # Unique object ID
    instance_id = int(elements[1]) # Instance ID
    class_id = int(elements[2])    # KITTI-MOTS category (1=car, 2=pedestrian, 10=ignore)
    height = int(elements[3])      # Image height
    width = int(elements[4])       # Image width
    rle_str = " ".join(elements[5:])  # RLE string

    return {
        "object_id": object_id,
        "instance_id": instance_id,
        "class_id": class_id,
        "height": height,
        "width": width,
        "rle_str": rle_str
    }


# Function to decode RLE into sparse tensor masks
def decode_kitti_mots_rle(annotation_line):
    parsed = parse_kitti_mots_annotation(annotation_line)

    coco_rle = {
        "counts": parsed["rle_str"].encode("utf-8"),
        "size": [parsed["height"], parsed["width"]]
    }

    try:
        mask = maskUtils.decode(coco_rle)
        mask = (mask > 0).astype(np.uint8)  # Convert to binary

        # Convert to PyTorch sparse tensor for memory efficiency
        mask_tensor = torch.tensor(mask, dtype=torch.uint8)
        sparse_mask = mask_tensor.to_sparse()

        return sparse_mask, parsed
    except Exception as e:
        print("Error decoding RLE:", str(e))
        return None, parsed


# Function to resize masks for memory efficiency
def resize_mask(mask, target_size=TARGET_SIZE):
    mask = mask.to_dense()  # Convert sparse tensor back to dense
    mask = transforms.functional.resize(mask.unsqueeze(0), target_size, interpolation=Image.NEAREST)
    return mask.squeeze(0).to_sparse()


# KITTI Dataset using RLE masks
class KITTIDataset(Dataset):
    def __init__(self, annotations, image_dir, instances_txt_dir, transform=None):
        self.annotations = annotations
        self.image_dir = image_dir
        self.instances_txt_dir = instances_txt_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(self.image_dir, ann["image"])

        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            return None

        image = Image.open(image_path).convert("RGB")

        # Resize image
        image = image.resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.BILINEAR)

        if self.transform:
            image = self.transform(image)

        gt_labels, gt_masks = [], []

        # Check if annotation file exists
        instance_txt_file = os.path.join(self.instances_txt_dir, f"{ann['image'].split("/")[0]}.txt")
        if not os.path.exists(instance_txt_file):
            return None  # Skip missing annotation

        with open(instance_txt_file, "r") as f:
            lines = f.readlines()

        for line in lines[:MAX_OBJECTS_PER_IMAGE]:
            sparse_mask, metadata = decode_kitti_mots_rle(line)
            if sparse_mask is None:
                continue

            coco_class = COCO_CATEGORY_MAPPING.get(metadata["class_id"], None)
            if coco_class is None:
                continue  # Skip invalid classes

            dense_mask = sparse_mask.to_dense()
            resized_mask = resize(dense_mask.unsqueeze(0), TARGET_SIZE, interpolation=Image.NEAREST).squeeze(0)
            sparse_mask = resized_mask.to_sparse()

            gt_masks.append(sparse_mask)
            gt_labels.append(coco_class)

        if len(gt_labels) == 0:  # No valid objects in the image
            return {
                "pixel_values": image,  # Image Tensor
                "class_labels": torch.tensor([], dtype=torch.int64),
                "mask_labels": torch.tensor(np.array([m.to_dense().cpu().numpy() if m.is_sparse else m for m in []]), dtype=torch.float16) if len([]) > 0 else torch.empty((0, TARGET_MASK_SIZE[0], TARGET_MASK_SIZE[1]), dtype=torch.float16),
                }

            # Convert to tensors


        return {
            "pixel_values": image,  # Ensure images are on GPU
            "mask_labels": torch.tensor(np.array([m.to_dense().cpu().numpy() if m.is_sparse else m for m in gt_masks]), dtype=torch.float16) if len(gt_masks) > 0 else torch.empty((0, TARGET_MASK_SIZE[0], TARGET_MASK_SIZE[1]), dtype=torch.float16),  # Move masks to GPU
            "class_labels": torch.tensor(gt_labels, dtype=torch.int64),  # Move labels to GPU
        }

def custom_collate(batch):
    batch = [item for item in batch if item is not None]  # Remove None items

    if len(batch) == 0:
        raise RuntimeError("❌ All batch elements are None. Check dataset loading!")

    pixel_values = torch.stack([item["pixel_values"] for item in batch])  # Keep on CPU
    mask_labels = [item["mask_labels"] for item in batch]  # Keep on CPU
    class_labels = [item["class_labels"] for item in batch]  # Keep on CPU

    return {
        "pixel_values": pixel_values,  # Keep on CPU
        "mask_labels": mask_labels,  # Keep on CPU
        "class_labels": class_labels,  # Keep on CPU
    }

@dataclass
class ModelOutput:
    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor

from torchmetrics.detection.mean_ap import MeanAveragePrecision


import gc


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.5, id2label=None):
    """
    Computa mAP para segmentación de manera incremental, actualizando el evaluador
    batch a batch, similar al enfoque usado en DETR.
    """
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
    metric = MeanAveragePrecision(iou_type="segm", box_format="xyxy", class_metrics=True)

    # Iterar sobre cada batch (se asume que predictions y targets están alineados)
    for batch_pred, batch_target in zip(predictions, targets):
        # Extraer los logits (ajusta según tu estructura real)
        class_queries_logits = batch_pred[0]
        masks_queries_logits = batch_pred[1]

        output = ModelOutput(
            class_queries_logits=torch.as_tensor(class_queries_logits, device="cpu"),
            masks_queries_logits=torch.as_tensor(masks_queries_logits, device="cpu"),
        )

        # Postprocesar la salida para obtener las predicciones para todas las imágenes del batch
        processed = image_processor.post_process_instance_segmentation(output, threshold=threshold)

        batch_predictions = []
        # Iteramos sobre todas las imágenes del batch
        for proc in processed:
            for instance in proc:
                pred_masks = instance.get("masks", instance.get("segmentation"))
                batch_predictions.append({
                    "labels": instance["labels"].cpu(),
                    "masks": pred_masks,
                })

        # Formatear la ground truth para el batch
        if isinstance(batch_target, dict):
            batch_target_formatted = {
                "masks": batch_target["mask_labels"],
                "labels": batch_target["class_labels"],
            }
        else:
            batch_target_formatted = {
                "masks": batch_target[0],
                "labels": batch_target[1],
            }

        # Actualizar la métrica con las predicciones y la GT de este batch
        metric.update(batch_predictions, [batch_target_formatted])

        # Liberar variables temporales
        del processed, batch_predictions, batch_target_formatted, output
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    final_metrics = metric.compute()
    metric.reset()
    gc.collect()
    torch.cuda.empty_cache()

    return final_metrics


# Wrap it in a partial function for Trainer
eval_compute_metrics_fn = partial(
    compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.5
)



class Mask2FormerTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Cálculo optimizado de loss para Mask2Former.
        Se asegura de que los tensores se transfieran de manera asíncrona a GPU,
        evitando llamadas excesivas a empty_cache.
        """
        device = model.device

        # Mover pixel_values a GPU de forma asíncrona
        inputs["pixel_values"] = inputs["pixel_values"].to(device, non_blocking=True)

        # Mover las máscaras a GPU (se asume que ya están en formato denso)
        inputs["mask_labels"] = [mask.to(device, non_blocking=True) for mask in inputs["mask_labels"]]

        # Mover las etiquetas a GPU
        inputs["class_labels"] = [
            c.to(device, non_blocking=True) if isinstance(c, torch.Tensor)
            else torch.tensor(c, dtype=torch.int64, device=device)
            for c in inputs["class_labels"]
        ]

        outputs = model(
            pixel_values=inputs["pixel_values"],
            mask_labels=inputs["mask_labels"],
            class_labels=inputs["class_labels"],
        )

        loss = outputs.loss
        # No se llama empty_cache() aquí para evitar overhead en cada batch
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name_or_path="facebook/mask2former-swin-large-coco-instance",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    for name, param in model.named_parameters():
        if "pixel_level_module.encoder" in name:
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir="./mask2former_kitti_finetuned_v2",
        auto_find_batch_size=True,
        num_train_epochs=30,
        fp16=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        eval_accumulation_steps=1,
        metric_for_best_model="eval_loss",  # Se sigue el loss para guardar el mejor modelo
        greater_is_better=False,  # En este caso, lower eval_loss es mejor
        dataloader_num_workers=4,
        learning_rate=1e-6,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        push_to_hub=True,
        # Parámetros de logging:
        logging_dir="./logs",  # Directorio donde se guardarán los logs
        logging_strategy="steps",  # Se registrarán logs cada cierto número de pasos
        logging_steps=50,  # Se hará log cada 50 pasos
        log_level="info",  # Nivel de detalle de los logs
    )

    # Load the dataset
    train_dataset = KITTIDataset(annotations, train_image_dir, instances_txt_dir, transform=image_transform)
    eval_dataset = KITTIDataset(annotations_val, val_image_dir, instances_txt_dir, transform=image_transform)

    # Initialize Trainer
    trainer = Mask2FormerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        data_collator=custom_collate,
    )

    trainer.train()
