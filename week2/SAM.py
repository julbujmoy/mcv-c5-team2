import torch
from segment_anything import SamPredictor, sam_model_registry
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# ---------------- CONFIGURACIÓN ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
results_path = "results"
os.makedirs(results_path, exist_ok=True)

# Cargar modelo SAM con los pesos entrenados
sam = sam_model_registry["vit_h"](checkpoint="sam_checkpoints/sam_vit_h_4b8939.pth")
sam.to(device)
sam.mask_decoder.load_state_dict(torch.load("fine_tuned_sam_decoder.pth"))
predictor = SamPredictor(sam)

# Cargar modelo DeTR para detección de objetos
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# ---------------- CARGAR KITTI-MOTS ----------------
def load_kitti_mots(dataset_path):
    images_path = os.path.join(dataset_path, "training/images")
    masks_path = os.path.join(dataset_path, "instances")

    image_files = {}
    mask_files = {}

    for folder in sorted(os.listdir(images_path)):
        folder_path = os.path.join(images_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, file)
            image_files[file] = img_path  

    for folder in sorted(os.listdir(masks_path)):
        folder_path = os.path.join(masks_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in sorted(os.listdir(folder_path)):
            mask_path = os.path.join(folder_path, file)
            mask_files[file] = mask_path  

    dataset = []
    for filename, img_path in image_files.items():
        mask_path = mask_files.get(filename)
        if mask_path is None:
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dataset.append((img, mask, filename))

    return dataset

# Cargar dataset
kitti_dataset = load_kitti_mots("KITTI-MOTS")
print(f" Total de imágenes cargadas: {len(kitti_dataset)}")

# ---------------- FUNCIÓN CÁLCULO mAP50 ----------------
def compute_mAP50(pred_masks, gt_mask):
    gt_mask = gt_mask.astype(bool)
    ious = []
    for pred_mask in pred_masks:
        pred_mask = pred_mask.astype(bool)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
    
    ious = np.array(ious)
    tp = (ious > 0.5).sum()
    fp = (ious <= 0.5).sum()
    fn = 1 - tp  

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    ap50 = precision * recall  

    return ap50

# ---------------- FUNCIÓN EVALUACIÓN DETR + SAM ----------------
def evaluate_with_detr(image, mask_gt, filename):
    image_pil = Image.fromarray(image)
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    outputs = detr_model(**inputs)

    logits = outputs.logits.softmax(-1)[0, :, :-1]
    boxes = outputs.pred_boxes[0]
    high_confidence = logits.max(dim=1).values > 0.7
    boxes = boxes[high_confidence].cpu().detach().numpy()

    if boxes.shape[0] == 0:
        print(f" No se encontraron objetos en {filename}")
        return 0.0

    points = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes])
    predictor.set_image(image)
    point_labels = np.ones(points.shape[0], dtype=int)

    masks, _, _ = predictor.predict(points, point_labels)

    # Guardar máscaras y visualización
    mask_path = os.path.join(results_path, f"{filename}_detr.npy")
    np.save(mask_path, masks)

    plt.imshow(image_pil)
    for mask in masks:
        plt.contour(mask, colors="blue", linewidths=2)
    plt.savefig(os.path.join(results_path, f"{filename}_detr.png"))
    plt.close()

    return compute_mAP50(masks, mask_gt)

# ---------------- EJECUTAR EVALUACIONES ----------------
mAP50_detr = []

for img, mask_gt, filename in kitti_dataset:
    ap50_detr = evaluate_with_detr(img, mask_gt, filename)
    mAP50_detr.append(ap50_detr)

print(f" mAP50 DeTR+SAM: {np.mean(mAP50_detr):.4f}")
