import torch
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import evaluate
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Configurar el dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo y el procesador de imágenes
model_id = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
feature_extractor = ViTImageProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configuración para generación de captions
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def generate_caption(image_path):
    """Genera un caption para una imagen usando ViT-GPT2."""
    image = Image.open(image_path).convert("RGB")
    
    # Procesar la imagen
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generar el caption
    with torch.no_grad():
        output_ids = model.generate(pixel_values, **gen_kwargs)

    # Decodificar el resultado
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

# Cargar métricas
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")

def evaluate_captions(predictions, references):
    """Calcula BLEU, METEOR y ROUGE para evaluar las captions generadas."""
    bleu1 = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references], max_order=1)
    bleu2 = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references], max_order=2)
    meteor_score = meteor_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    return {
        "BLEU-1": bleu1["bleu"],
        "BLEU-2": bleu2["bleu"],
        "METEOR": meteor_score["meteor"],
        "ROUGE": rouge_score["rougeL"]
    }

# Directorios de imágenes
test_img_folder = "FoodImages/test"

# Cargar datos de prueba
test_annotations = pd.read_csv("FoodImages/test_data.csv")

predictions = []
references = test_annotations["Title"].tolist()

print("Generando captions para las imágenes del conjunto de prueba...")

for img_name in tqdm(test_annotations["Image_Name"], desc="Procesando imágenes"):
    img_path = os.path.join(test_img_folder, img_name + ".jpg")
    caption = generate_caption(img_path)
    predictions.append(caption)

# Evaluar métricas
results = evaluate_captions(predictions, references)

print("\nResultados de Evaluación:")
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")

print("\nEjemplos de captions generadas:")
for i in range(min(5, len(predictions))):
    print(f"Predicción {i + 1}: '{predictions[i]}'")
    print(f"Anotación  {i + 1}: '{references[i]}'")
    print("-" * 50)
