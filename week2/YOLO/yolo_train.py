from ultralytics import YOLO

# Cargar YOLOv8-Seg preentrenado
model = YOLO("yolov8n-seg.pt")  # Usa "yolov8m-seg.pt" para mejor precisión

# Congelar todo el backbone
#for param in model.model.model[:10]:  # Normalmente las primeras 10 capas son el backbone
#    param.requires_grad = False

# Congelar capas iniciales para preservar características generales
#model.model.model[0].requires_grad = False  # Congela la primera capa
#model.model.model[1].requires_grad = False  # Congela la segunda capa

# Entrenar solo la cabeza de segmentación
model.train(
    data="kitti_mots.yaml",
    epochs=50,  # Menos épocas porque solo ajustamos la cabeza
    batch=8,
    
    # Augmentaciones de color
    hsv_h=0.01,  # Variación en matiz
    hsv_s=0.6,    # Saturación aleatoria
    hsv_v=0.3,    # Brillo aleatorio

    # Augmentaciones geométricas
    degrees=8,     # Rotación ±5 grados
    translate=0.1, # Desplazamiento
    scale=0.35,     # Zoom aleatorio
    shear=4,       # Cizallamiento
    fliplr=0.5,    # Volteo horizontal
    flipud=0,    # No voltear verticalmente

    # Métodos avanzados
    mosaic=1,    # Mezcla imágenes (Mosaic)
    mixup=0.3,     # Fusiona imágenes (MixUp)
    copy_paste=0.5, # Aumentación Copy-Paste
    
    lr0=0.001,
    lrf=0.00001,  # Termina en un LR más bajo
    warmup_epochs=3,  # Asegura un buen inicio
    warmup_momentum=0.8,
    optimizer="AdamW",  # Prueba AdamW en lugar de SGD
    cos_lr=True,  # Usa decaimiento coseno
    
    project="runs/seg",
    name="lr_nonft",
    save=True
)
 
print("✅ Entrenamiento finalizado")
