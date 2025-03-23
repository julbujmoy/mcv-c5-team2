from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

DATASET_PATH = "/home/mcv/datasets/C5/KITTI-MOTS"
ANNOTATION_FILE = "/home/c5mcv02/instances_val_jul.json"
IMAGE_DIR = f"{DATASET_PATH}/training/image_02/0000"

# Eliminar dataset anterior para evitar conflicto
if "kitti_mots_val" in DatasetCatalog.list():
    DatasetCatalog.remove("kitti_mots_val")
    MetadataCatalog.remove("kitti_mots_val")  # Asegúrate de eliminar también las metainformaciones

# Registrar nuevamente el dataset
register_coco_instances("kitti_mots_val", {}, ANNOTATION_FILE, IMAGE_DIR)

# Definir las clases de tu dataset
coco_classes = ["car", "person"]

# Registrar las categorías del dataset
MetadataCatalog.get("kitti_mots_val").set(thing_classes=coco_classes)

print(" Dataset KITTI-MOTS registrado en Detectron2.")
