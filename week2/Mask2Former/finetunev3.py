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
    1: "pedestrian",  # No object
    2: "car",
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
train_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.RandomPerspective(),
    transforms.RandomHorizontalFlip(),  # Aumenta la variabilidad
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Ajustes de color
    transforms.ToTensor(),
])

# Transformación para validación (mínima)
val_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
])

# Mapping from KITTI-MOTS categories to COCO
COCO_CATEGORY_MAPPING = {
    1: 2,  # KITTI car → COCO car
    2: 0,  # KITTI pedestrian → COCO person
    10: None,  # Ignore class
}

# Maximum objects per image to avoid memory explosion
MAX_OBJECTS_PER_IMAGE = 15


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

import random
from torchvision.transforms import functional as F
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import ColorJitter

class JointTransform:
    def __init__(self, size, perspective_prob=0.5, horizontal_flip_prob=0.5, color_jitter_params=(0.2, 0.2, 0.2, 0.1)):
        """
        size: tamaño final, e.g., (256, 512)
        perspective_prob: probabilidad de aplicar perspectiva
        horizontal_flip_prob: probabilidad de flip horizontal
        color_jitter_params: parámetros para ColorJitter
        """
        self.size = size
        self.perspective_prob = perspective_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        #self.jitter = ColorJitter(*color_jitter_params)

    def __call__(self, image, masks):
        """
        image: PIL.Image en RGB
        masks: lista de PIL.Image (una por objeto) con modo 'L' o similar.
        """
        # 1. Resize (se aplica a la imagen y a cada máscara)
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        masks = [F.resize(m, self.size, interpolation=F.InterpolationMode.NEAREST) for m in masks]

        # 2. Random Perspective
        if random.random() < self.perspective_prob:
            startpoints, endpoints = self.get_perspective_params(image)
            image = F.perspective(image, startpoints, endpoints, interpolation=F.InterpolationMode.BILINEAR)
            masks = [F.perspective(m, startpoints, endpoints, interpolation=F.InterpolationMode.NEAREST) for m in masks]

        # 3. Random Horizontal Flip
        if random.random() < self.horizontal_flip_prob:
            image = F.hflip(image)
            masks = [F.hflip(m) for m in masks]

        # 4. Color Jitter solo en la imagen
        #image = self.jitter(image)

        # 5. Convertir imagen a tensor (float) y máscaras a tensor (long)
        image = F.to_tensor(image)  # Esto convierte la imagen a float y normaliza a [0,1]
        masks = [F.to_tensor(m).squeeze(0).float() for m in masks]  # Asumimos que cada máscara es de un solo canal
        masks = torch.stack(masks)  # [N, H, W]
        return image, masks

    def get_perspective_params(self, image):
        """
        Genera parámetros para una transformación de perspectiva.
        Se utiliza una aproximación simple, basándose en las esquinas de la imagen.
        """
        width, height = image.size
        startpoints = [(0, 0), (width, 0), (width, height), (0, height)]
        # Se define una escala de distorsión (ajustable)
        distortion_scale = 0.5
        max_dx = distortion_scale * width * 0.5
        max_dy = distortion_scale * height * 0.5
        endpoints = []
        for x, y in startpoints:
            new_x = x + random.uniform(-max_dx, max_dx)
            new_y = y + random.uniform(-max_dy, max_dy)
            endpoints.append((new_x, new_y))
        return startpoints, endpoints

class JointValTransform:
    def __init__(self, size):
        self.size = size
    def __call__(self, image, masks):
        # Redimensiona imagen y máscaras
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        masks = [F.resize(m, self.size, interpolation=F.InterpolationMode.NEAREST) for m in masks]
        # Convertir a tensor
        image = F.to_tensor(image)
        masks = [F.to_tensor(m).squeeze(0).float() for m in masks]
        masks = torch.stack(masks)
        return image, masks


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

        # Verificar que la imagen exista
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            return None

        # Abrir la imagen como PIL
        image = Image.open(image_path).convert("RGB")

        gt_labels, gt_masks = [], []

        # Verificar que exista el archivo de anotación
        instance_txt_file = os.path.join(self.instances_txt_dir, f"{ann['image'].split('/')[0]}.txt")
        if not os.path.exists(instance_txt_file):
            return None  # O bien, podrías devolver una muestra dummy

        with open(instance_txt_file, "r") as f:
            lines = f.readlines()

        for line in lines[:MAX_OBJECTS_PER_IMAGE]:
            sparse_mask, metadata = decode_kitti_mots_rle(line)
            if sparse_mask is None:
                continue

            coco_class = COCO_CATEGORY_MAPPING.get(metadata["class_id"], None)
            if coco_class is None:
                continue  # Saltar clases inválidas

            # Convertir la máscara esparsa a densa y luego a numpy
            dense_mask = sparse_mask.to_dense()
            # Redimensionar la máscara manualmente para tener un tamaño base consistente.
            # Si usas el JointTransform, éste se encargará del resize; sin embargo, si no se aplica,
            # lo hacemos aquí para garantizar un tamaño (TARGET_SIZE).
            resized_mask = F.resize(Image.fromarray(dense_mask.cpu().numpy().astype(np.uint8)),
                                     TARGET_SIZE, interpolation=F.InterpolationMode.NEAREST)
            gt_masks.append(resized_mask)
            gt_labels.append(coco_class)

        # En caso de que no se encuentren objetos válidos, se crea una máscara dummy
        if len(gt_masks) == 0:
            dummy_mask = Image.fromarray(np.zeros((TARGET_SIZE[0], TARGET_SIZE[1]), dtype=np.uint8))
            gt_masks = [dummy_mask]
            # Se podría dejar gt_labels vacío o asignar una etiqueta dummy según el caso

        # Aplicar la transformación conjunta (si se ha definido)
        if self.transform:
            image, mask_tensor = self.transform(image, gt_masks)
        else:
            # Convertir la imagen y las máscaras a tensor sin transformaciones adicionales
            image = F.to_tensor(image)
            mask_tensor = torch.stack([F.to_tensor(m).squeeze(0).long() for m in gt_masks])

        return {
            "pixel_values": image,       # Tensor [C, H, W]
            "mask_labels": mask_tensor,    # Tensor [N, H, W]
            "class_labels": torch.tensor(gt_labels, dtype=torch.int64),
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
    # Define la transformación conjunta
    joint_transform = JointTransform(size=(256, 512), perspective_prob=0.5, horizontal_flip_prob=0.5,
                                     color_jitter_params=(0.2, 0.2, 0.2, 0.1))
    joint_val_transform = JointValTransform((256, 512))


    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name_or_path="facebook/mask2former-swin-large-coco-instance",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir="mask2former_kitti_finetuned_horizon_v2_good_loss",
        auto_find_batch_size=True,
        num_train_epochs=30,
        fp16=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=500,
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
        logging_steps=50,
    )

    # Load the dataset
    train_dataset = KITTIDataset(annotations, train_image_dir, instances_txt_dir, transform=joint_transform)
    eval_dataset = KITTIDataset(annotations_val, val_image_dir, instances_txt_dir, transform=joint_val_transform)

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