from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Evaluate KITTI-MOTS
metrics = model.val(data="YOLO/kitti_mots.yaml")
print(metrics)