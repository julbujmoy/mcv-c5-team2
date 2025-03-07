from ultralytics import YOLO

# Cargar YOLOv8 preentrenado
model = YOLO("yolov8n.pt")  # o "yolov8m.pt" para mejor precisión

# Entrenamiento con augmentaciones avanzadas
model.train(
    data="kitti_mots.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    hsv_h=0.02,   # Aumenta el cambio en matiz
    hsv_s=0.8,    # Aumenta la saturación
    hsv_v=0.5,    # Más variación en brillo
    project="runs/train",
    name="color_aug",
    save=True
)


#    hsv_h=0.015,  # Aumenta la variación de matiz en la imagen
#    hsv_s=0.7,    # Aumenta la saturación aleatoria
#    hsv_v=0.4,    # Variación en brillo
#    degrees=5,    # Rotación aleatoria de ±5 grados
#    translate=0.1,  # Desplazamiento de imagen
#    scale=0.5,     # Zoom aleatorio
#    shear=5,       # Cizallamiento aleatorio
#    flipud=0.5,    # Voltear verticalmente
#    fliplr=0.5,    # Voltear horizontalmente
#    mosaic=1.0,    # Mezcla imágenes para mejorar generalización
#    mixup=0.2,     # Fusión de imágenes
#    copy_paste=0.1, # Aumentación de Copy-Paste


# Fase 1
#    hsv_h=0.02,   # Aumenta el cambio en matiz
#    hsv_s=0.8,    # Aumenta la saturación
#    hsv_v=0.5,    # Más variación en brillo