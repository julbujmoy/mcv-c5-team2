import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import evaluate
from tqdm import tqdm  # Progreso visual

# 📌 1. Configurar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📌 2. Cargar ViT preentrenado (puedes cambiarlo luego por tu modelo)
vit_model_id = "google/vit-base-patch16-224-in21k"
vit_model = ViTModel.from_pretrained(vit_model_id)
vit_model.eval()
for param in vit_model.parameters():
    param.requires_grad = False  # Congelar ViT

# 📌 3. Cargar LLaMA (1B o 3B) y aplicar LoRA
llama_model_id = "meta-llama/Llama-3.2-1B"
llama_model = LlamaForCausalLM.from_pretrained(llama_model_id)
tokenizer = LlamaTokenizer.from_pretrained(llama_model_id)

# Aplicar LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
llama_model = get_peft_model(llama_model, lora_config)

# 📌 4. Modelo combinado (ViT + LLaMA)
class VisionToTextModel(nn.Module):
    def __init__(self, vit, llama):
        super().__init__()
        self.vit = vit
        self.llama = llama
        self.projection = nn.Linear(768, 4096)

    def forward(self, images, input_ids):
        with torch.no_grad():
            vision_features = self.vit(images).last_hidden_state[:, 0, :]
        vision_features = self.projection(vision_features)
        outputs = self.llama(input_ids=input_ids, encoder_hidden_states=vision_features.unsqueeze(1))
        return outputs

# Inicializar modelo
model = VisionToTextModel(vit_model, llama_model).to(device)

# 📌 5. Dataset personalizado
class ImageCaptionDataset(Dataset):
    def __init__(self, img_folder, annotations_file, tokenizer):
        self.img_folder = img_folder
        self.data = pd.read_csv(annotations_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.data.iloc[idx, 0] + ".jpg")
        caption = self.data.iloc[idx, 1]

        # Procesar imagen
        image = Image.open(img_path).convert("RGB")
        image = vit_model.feature_extractor(image, return_tensors="pt").pixel_values

        # Tokenizar caption
        tokens = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=50)

        return image.squeeze(0), tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)

# 📌 6. Cargar datasets
batch_size = 8
train_dataset = ImageCaptionDataset("FoodImages/train", "FoodImages/train_data.csv", tokenizer)
valid_dataset = ImageCaptionDataset("FoodImages/valid", "FoodImages/valid_data.csv", tokenizer)
test_dataset = ImageCaptionDataset("FoodImages/test", "FoodImages/test_data.csv", tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 📌 7. Configurar optimizador y función de pérdida
optimizer = optim.AdamW(model.llama.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# 📌 8. Evaluación de métricas (BLEU, METEOR, ROUGE)
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")

def evaluate_model(dataloader, desc="Evaluando"):
    """Evalúa el modelo con BLEU, METEOR y ROUGE."""
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for images, input_ids, _ in tqdm(dataloader, desc=desc):
            images = images.to(device)
            pred_caption = generate_caption(images)
            predictions.append(pred_caption)
            references.append(tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True))

    bleu_score = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references], max_order=2)
    meteor_score = meteor_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    return {
        "BLEU-1": bleu_score["bleu"],
        "METEOR": meteor_score["meteor"],
        "ROUGE": rouge_score.get("rougeL", 0)
    }

def generate_caption(image):
    """Genera un caption para una imagen dada usando ViT + LLaMA."""
    input_ids = torch.tensor(tokenizer("a photo of", return_tensors="pt").input_ids).to(device)

    with torch.no_grad():
        outputs = model(image, input_ids)
    
    caption = tokenizer.decode(outputs.logits.argmax(-1).squeeze(), skip_special_tokens=True)
    return caption

# 📌 9. Entrenamiento con validación (con tqdm)
epochs = 3
best_valid_score = 0
best_model_path = "best_vit_llama_captioning.pth"

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for images, input_ids, attn_mask in tqdm(train_loader, desc=f"Entrenando Epoch {epoch + 1}/{epochs}", leave=False):
        images, input_ids, attn_mask = images.to(device), input_ids.to(device), attn_mask.to(device)

        optimizer.zero_grad()
        outputs = model(images, input_ids)
        loss = loss_fn(outputs.logits.view(-1, outputs.logits.shape[-1]), input_ids.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Evaluar en validación después de cada epoch
    print("\n🔹 Evaluando en validación...")
    valid_scores = evaluate_model(valid_loader, desc="Validando")

    print(f"\n🔹 Resultados Epoch {epoch + 1}:")
    print(f"   🔹 Train Loss: {avg_train_loss:.4f}")
    print(f"   🔹 Validación BLEU-1: {valid_scores['BLEU-1']:.4f}")
    print(f"   🔹 Validación METEOR: {valid_scores['METEOR']:.4f}")
    print(f"   🔹 Validación ROUGE: {valid_scores['ROUGE']:.4f}")

    # Guardar el mejor modelo basado en validación BLEU-1
    if valid_scores["BLEU-1"] > best_valid_score:
        best_valid_score = valid_scores["BLEU-1"]
        torch.save(model.state_dict(), best_model_path)
        print("✅ Modelo mejorado guardado.")

# 📌 10. Evaluación final en test
print("\n🔹 Evaluando en conjunto de test...")
model.load_state_dict(torch.load(best_model_path))
test_scores = evaluate_model(test_loader, desc="Test")

print("\n🔹 Resultados en Test:")
print(f"   🔹 Test BLEU-1: {test_scores['BLEU-1']:.4f}")
print(f"   🔹 Test METEOR: {test_scores['METEOR']:.4f}")
print(f"   🔹 Test ROUGE: {test_scores['ROUGE']:.4f}")
