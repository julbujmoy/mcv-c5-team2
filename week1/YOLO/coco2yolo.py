import json
import os

# Ruta del archivo JSON con anotaciones
json_file = "YOLO/kitti_mots_coco.json"
# Carpeta donde se guardarán las etiquetas YOLO
labels_dir = "yolo_labels"

# Crear carpeta de etiquetas si no existe
os.makedirs(labels_dir, exist_ok=True)

# Mapeo de categorías a índices YOLO
category_map = {1: 0, 2: 1, 10: 2}  # "car" -> 0, "pedestrian" -> 1

# Cargar JSON
with open(json_file, "r") as f:
    annotations = json.load(f)

# Procesar cada imagen
for ann in annotations:
    # Obtener el nombre de la imagen y el directorio donde se guardará la etiqueta
    image_name = ann["image"].split("/")[-1].replace(".png", ".txt")
    
    # Extraer el prefijo de la imagen (por ejemplo, "0000")
    folder_name = ann["image"].split("/")[0]
    folder_path = os.path.join(labels_dir, folder_name)
    
    # Crear la subcarpeta si no existe
    os.makedirs(folder_path, exist_ok=True)

    # Crear la ruta completa del archivo de etiqueta
    label_path = os.path.join(folder_path, image_name)

    with open(label_path, "w") as label_file:
        for obj_id, bbox, category in zip(ann["objects"]["id"], ann["objects"]["bbox"], ann["objects"]["category"]):


            x, y, w, h = bbox
            x_center = (x + w / 2) / ann["width"]
            y_center = (y + h / 2) / ann["height"]
            w_norm = w / ann["width"]
            h_norm = h / ann["height"]

            # Escribir la etiqueta en el archivo
            label_file.write(f"{category_map[category]} {x_center} {y_center} {w_norm} {h_norm}\n")

print("✅ Conversión completada. Archivos guardados en:", labels_dir)
