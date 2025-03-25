
import torch
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import evaluate

# Configurar el dispositivo (GPU si está disponible, sino CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el modelo y el procesador de imágenes


from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
model.to(device)
model.eval()


def generate_caption(image_path):
    """Genera un caption para una imagen dada usando LLaMA 3.2-11B Multimodal."""
    image = Image.open(image_path).convert("RGB")
    
    # Procesar la imagen
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generar el caption
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)

    # Decodificar el resultado
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption



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

train_img_folder = "FoodImages/train"
valid_img_folder = "FoodImages/valid"
test_img_folder = "FoodImages/test"

# Load pre-split CSV annotation files
train_annotations = pd.read_csv("FoodImages/train_data.csv")
valid_annotations = pd.read_csv("FoodImages/valid_data.csv")
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
        print(f"Prediccion {i + 1}: '{predictions[i]}'")
        print(f"Anotacion  {i + 1}: '{references[i]}'")
        print("-" * 50)
