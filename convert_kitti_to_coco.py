import os
import json

# Rutas
DATASET_PATH = "/home/mcv/datasets/C5/KITTI-MOTS"
IMAGES_PATH = f"{DATASET_PATH}/training/image_02/0000"
ANNOTATIONS_PATH = f"{DATASET_PATH}/instances_txt"
OUTPUT_JSON = "/home/c5mcv02/instances_val.json"


# Categorías de KITTI-MOTS
categories = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "pedestrian"}
]

# Estructura COCO
coco_data = {
    "images": [],
    "annotations": [],
    "categories": categories
}

annotation_id = 1

# Recorrer imágenes y anotaciones
for img_id, img_name in enumerate(sorted(os.listdir(IMAGES_PATH))):
    if not img_name.endswith(".png"):
        continue

    img_path = os.path.join(IMAGES_PATH, img_name)
    annotation_file = os.path.join(ANNOTATIONS_PATH, img_name.replace(".png", ".txt"))

    # Obtener tamaño de imagen
    height, width = 375, 1242  # Reemplazar con valores reales si es necesario

    # Añadir imagen a COCO
    coco_data["images"].append({
        "id": img_id,
        "file_name": img_name,
        "height": height,
        "width": width
    })

    # Leer anotaciones del archivo .txt
    if os.path.exists(annotation_file):
        with open(annotation_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                track_id, class_id, x, y, w, h, *_ = map(int, parts[:6])
                
                # Añadir anotación a COCO
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0
                })
                annotation_id += 1

# Guardar como JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"Archivo COCO guardado en: {OUTPUT_JSON}")

