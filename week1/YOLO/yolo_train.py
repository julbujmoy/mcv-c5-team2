from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt") 

# Train with augmentations
model.train(
    data="kitti_mots.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    hsv_h=0.02,   
    hsv_s=0.8,   
    hsv_v=0.5,    
    project="runs/train",
    name="color_aug",
    save=True
)


# Default parameters
#    hsv_h=0.015,  # Increases hue variation in the image
#    hsv_s=0.7,    # Increases random saturation
#    hsv_v=0.4,    # Variation in brightness
#    degrees=5,    # Random rotation of ±5 degrees
#    translate=0.1,  # Image translation
#    scale=0.5,     # Random zoom
#    shear=5,       # Random shearing
#    flipud=0.5,    # Vertical flip
#    fliplr=0.5,    # Horizontal flip
#    mosaic=1.0,    # Mixes images to improve generalization
#    mixup=0.2,     # Image blending
#    copy_paste=0.1, # Copy-Paste augmentation

# Non agumented
#   All parameters to 0.0

#(color augmentations)
#    hsv_h=0.02,   
#    hsv_s=0.8,    
#    hsv_v=0.5,    
#  v2
#    hsv_h=0.01,  
#    hsv_s=0.5,   
#    hsv_v=0.3,   

#(geometric augmentations)
#    degrees=10,      
#    translate=0.2,  
#    scale=0.8,     
#    shear=10,      
#  v2
#    degrees=3,    
#    translate=0.05, 
#    scale=0.3,    
#    shear=3,      
#    fliplr=0.5,   
#    flipud=0.0,   


#(advance methods)
#    mosaic=1.0,     
#    mixup=0.2,   
#  v2
#    mosaic=0.8,  
#    mixup=0.1,    