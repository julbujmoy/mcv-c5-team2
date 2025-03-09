from ultralytics import YOLO
import cv2
import os
import glob

# Configuration
MODEL_PATH = "runs/train/all_aug/weights/best.pt"  
IMAGE_ROOT = "/export/home/c5mcv02/KITTI-MOTS/testing/images/0000"  
OUTPUT_ROOT = "runs/inference/all_aug" 

# Load model
model = YOLO(MODEL_PATH)

# Output folder
output_folder = os.path.join(OUTPUT_ROOT, "0000") 
os.makedirs(output_folder, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(IMAGE_ROOT, "*.png"))) 

# Inference for each image
for img_path in image_paths:
    results = model(img_path)
    
    save_path = os.path.join(output_folder, os.path.basename(img_path))
    results[0].save(save_path) 

print(f"Results saved in {OUTPUT_ROOT}/0000")
