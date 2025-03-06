import os
import json
import cv2  # Para obtener dimensiones reales de las imágenes


DATASET_PATH = "/home/mcv/datasets/C5/KITTI-MOTS"
IMAGES_PATH = f"{DATASET_PATH}/training/image_02/0000"
ANNOTATIONS_PATH = f"{DATASET_PATH}/instances_txt"
OUTPUT_JSON = "/home/c5mcv02/instances_val_jul.json"

# Solo incluir estas categorías
CLASS_MAP = {1: "car", 2: "person"}

# Estructura COCO
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}]
}

annotation_id = 1

# Recorrer imágenes y anotaciones
for img_id, img_name in enumerate(sorted(os.listdir(IMAGES_PATH))):
    if not img_name.endswith(".png"):
        continue

    img_path = os.path.join(IMAGES_PATH, img_name)
    annotation_file = os.path.join(ANNOTATIONS_PATH, img_name.replace(".png", ".txt"))

    # Obtener tamaño de imagen real
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Añadir imagen a COCO
    coco_data["images"].append({
        "id": img_id,
        "file_name": img_name,
        "height": height,
        "width": width
    })

    # Leer anotaciones
    if os.path.exists(annotation_file):
        with open(annotation_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                track_id, class_id, x, y, w, h, *_ = map(int, parts[:6])

                # Filtrar solo "car" y "pedestrian"
                if class_id not in CLASS_MAP:
                    continue

                # Convertir anotación a formato COCO
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": 1 if class_id == 1 else 2,  # Para que 2 sea "person"
                    "bbox": [x, y, w, h],  # Formato COCO
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1

# Guardar JSON COCO
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f" Archivo COCO guardado en: {OUTPUT_JSON}")

