from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch
import register  # Importa el dataset registrado


# Configurar Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Solo "car" y "pedestrian"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelo
predictor = DefaultPredictor(cfg)

# Evaluar usando COCOEvaluator
evaluator = COCOEvaluator("kitti_mots_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "kitti_mots_val")

print("Evaluando Faster R-CNN en KITTI-MOTS...")
metrics = inference_on_dataset(predictor.model, val_loader, evaluator)

print("Evaluación completada.")
print(metrics)
