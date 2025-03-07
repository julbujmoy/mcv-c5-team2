import os
import shutil
import random

# Configurar rutas
DATASET_PATH = "/export/home/c5mcv02/KITTI-MOTS"
TRAINING_PATH = os.path.join(DATASET_PATH, "training")
TRAIN_OUTPUT = os.path.join(DATASET_PATH, "train")
VAL_OUTPUT = os.path.join(DATASET_PATH, "val")

# Crear carpetas de salida
for split in [TRAIN_OUTPUT, VAL_OUTPUT]:
    os.makedirs(os.path.join(split, "images"), exist_ok=True)
    os.makedirs(os.path.join(split, "labels"), exist_ok=True)

# Obtener todas las imágenes
image_files = sorted(os.listdir(os.path.join(TRAINING_PATH, "images")))
label_files = sorted(os.listdir(os.path.join(TRAINING_PATH, "labels")))

# Mezclar aleatoriamente y dividir (80% train, 20% val)
random.seed(42)
data = list(zip(image_files, label_files))
random.shuffle(data)

split_index = int(0.8 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

# Mover archivos a sus respectivas carpetas
def move_files(data, split_path):
    for img, lbl in data:
        shutil.move(os.path.join(TRAINING_PATH, "images", img), os.path.join(split_path, "images", img))
        shutil.move(os.path.join(TRAINING_PATH, "labels", lbl), os.path.join(split_path, "labels", lbl))

move_files(train_data, TRAIN_OUTPUT)
move_files(val_data, VAL_OUTPUT)

print(f"División completada: {len(train_data)} train, {len(val_data)} val")
