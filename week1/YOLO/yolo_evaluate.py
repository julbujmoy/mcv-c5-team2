from ultralytics import YOLO

# Load model
model = YOLO("runs/train/all_aug/weights/best.pt")

# Evaluate KITTI-MOTS
metrics = model.val(data="YOLO/kitti_mots.yaml")
print(metrics)