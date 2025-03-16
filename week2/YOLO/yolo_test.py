from ultralytics import YOLO
import cv2
import os
import glob
from time import time

# Configuración
MODEL_PATH = "yolov8n-seg.pt"  # Cambia a tu modelo fine-tuneado si lo tienes
IMAGE_ROOT = "/export/home/c5mcv02/KITTI-MOTS/testing/images/0000"  # Carpeta 0000 de test
OUTPUT_ROOT = "runs/inference/yolo-seg"  # Carpeta donde se guardarán los resultados

# Cargar modelo YOLOv8
model = YOLO(MODEL_PATH)

# Carpeta de salida para los resultados
output_folder = os.path.join(OUTPUT_ROOT, "0000")  # Carpeta de salida específica para 0000
os.makedirs(output_folder, exist_ok=True)

# Obtener imágenes de la carpeta 0000
image_paths = sorted(glob.glob(os.path.join(IMAGE_ROOT, "*.png")))  # KITTI usa .png

t1 = time()
results = model("/export/home/c5mcv02/KITTI-MOTS/testing/images/0000/000000.png")
t2 = time()
print("Tiempo por imagen:", t2-t1)

# Ejecutar inferencia en cada imagen
for img_path in image_paths:
    results = model(img_path)  # Realizar inferencia
    
    # Guardar imagen con predicciones
    save_path = os.path.join(output_folder, os.path.basename(img_path))
    results[0].save(save_path)  # Guarda la imagen con las predicciones

print(f"Resultados guardados en {OUTPUT_ROOT}/0000")
