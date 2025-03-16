import os
import cv2
import numpy as np

# Directorios del dataset
MASKS_DIR = "/home/c5mcv02/KITTI-MOTS/instances"  # Directorio con máscaras en PNG
OUTPUT_DIR = "/home/c5mcv02/KITTI-MOTS/training/labels_kitti"  # Directorio de salida para etiquetas YOLO
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapeo de clases de KITTI-MOTS a YOLO
category_map = {1: 0, 2: 1}  # 1 (car) → 0 (YOLO), 2 (pedestrian) → 1 (YOLO)

# Iterar sobre las secuencias (subcarpetas dentro de instances/)
for sequence in sorted(os.listdir(MASKS_DIR)):
    sequence_path = os.path.join(MASKS_DIR, sequence)

    # Crear carpeta de salida para la secuencia
    output_sequence_path = os.path.join(OUTPUT_DIR, sequence)
    os.makedirs(output_sequence_path, exist_ok=True)

    if not os.path.isdir(sequence_path):  # Saltar si no es una carpeta
        continue

    # Iterar sobre los archivos de máscara dentro de la carpeta
    for mask_file in sorted(os.listdir(sequence_path)):
        mask_path = os.path.join(sequence_path, mask_file)

        # Cargar la máscara
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"❌ Error: No se pudo cargar la máscara en {mask_path}")
            continue

        # Obtener valores únicos en la máscara (cada valor representa una instancia)
        unique_instances = np.unique(mask)

        # Crear archivo YOLO para esta imagen
        yolo_label_path = os.path.join(output_sequence_path, mask_file.replace(".png", ".txt"))
        with open(yolo_label_path, "w") as label_file:
            for instance_id in unique_instances:
                if instance_id == 0:  # Fondo, ignorar
                    continue

                class_id = instance_id // 1000  # Obtener la clase
                obj_id = instance_id % 1000  # Obtener el ID de la instancia

                if class_id not in category_map:  # Ignorar regiones no etiquetadas (ej: 10)
                    continue

                yolo_class = category_map[class_id]

                # Obtener máscara binaria de la instancia actual
                instance_mask = (mask == instance_id).astype(np.uint8)

                # Obtener bounding box de la instancia
                y_indices, x_indices = np.where(instance_mask > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)

                # Normalizar coordenadas a formato YOLO
                img_h, img_w = mask.shape
                x_center = (x_min + x_max) / 2 / img_w
                y_center = (y_min + y_max) / 2 / img_h
                bbox_w = (x_max - x_min) / img_w
                bbox_h = (y_max - y_min) / img_h

                # Guardar en formato YOLO
                label_file.write(f"{yolo_class} {x_center} {y_center} {bbox_w} {bbox_h}\n")

print("✅ Conversión completada. Archivos guardados en:", OUTPUT_DIR)
