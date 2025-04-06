import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import evaluate
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import wandb

# Initialize wandb
wandb.init(
    project="ViT-GPT2-FINETUNE",
    entity='laurasalort',
    name="vit_gpt2_finetuned",
    config={
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "model": "vit_gpt2_finetuned"
    }
)
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_id = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
feature_extractor = ViTImageProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Freeze decoder
#gpt2_params = model.decoder.parameters()
#for param in gpt2_params:
#    param.requires_grad = False  # No se entrena GPT-2

#vit_params = model.encoder.parameters()
#for param in vit_params:
#    param.requires_grad = False  # No se entrena GPT-2

#freeze_layers = 4
#for layer in model.decoder.transformer.h[-freeze_layers:]:
#    for param in layer.parameters():
#        param.requires_grad = False

# No congelamos nada: tanto ViT como GPT-2 serán entrenados
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-7)

# Load metrics
eval_bleu = evaluate.load("bleu")
eval_meteor = evaluate.load("meteor")
eval_rouge = evaluate.load("rouge")


# Custom dataset
class ImageCaptionDataset(Dataset):
    def __init__(self, img_folder, annotations):
        self.img_folder = img_folder
        self.annotations = annotations
        valid_rows = []
        for index, row in annotations.iterrows():
            img_name = row["Image_Name"]
            if not isinstance(img_name, str):
                img_name = str(img_name)
            file_path = os.path.join(self.img_folder, img_name) + ".jpg"
            if os.path.exists(file_path):
                valid_rows.append(row)
            else:
                print(f"Skipping missing image: {file_path}")
        self.annotations = pd.DataFrame(valid_rows)
        print(f"Initialized dataset with {len(self.annotations)} samples out of {len(annotations)} total rows.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx]["Image_Name"] + ".jpg"
        img_path = os.path.join(self.img_folder, img_name)
        caption = str(self.annotations.iloc[idx]["Title"])

        image = Image.open(img_path).convert("RGB")
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()
        tokenized_caption = tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True,
                                      max_length=20).input_ids.squeeze()

        return pixel_values, tokenized_caption


# Load data
train_img_folder = "FoodImages/train_filtered"
valid_img_folder = "FoodImages/valid_filtered"
train_annotations = pd.read_csv("FoodImages/train_filtered.csv")
valid_annotations = pd.read_csv("FoodImages/valid_filtered.csv")
train_annotations["Title"] = train_annotations["Title"].fillna("")
valid_annotations["Title"] = valid_annotations["Title"].fillna("")

dataset_train = ImageCaptionDataset(train_img_folder, train_annotations)
dataset_valid = ImageCaptionDataset(valid_img_folder, valid_annotations)
dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=config.batch_size, shuffle=False)

# Training loop
num_epochs = config.epochs
patience = 5  # Más paciencia porque ahora entrenamos más parámetros
best_val_loss = float("inf")
counter = 0

model.train()
for epoch in range(num_epochs):
    loop = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
    total_loss = 0
    for images, captions in loop:
        images, captions = images.to(device), captions.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images, labels=captions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(dataloader_train)

    # Validation
    # Validation
    model.eval()
    val_loss = 0
    predictions = []
    references = []
    with torch.no_grad():
        for images, captions in tqdm(dataloader_valid, desc=f"Epoch {epoch + 1} Validation"):
            images, captions = images.to(device), captions.to(device)
            outputs = model(pixel_values=images, labels=captions)
            val_loss += outputs.loss.item()

            # Generate captions using nuclear (top-p) sampling
            generated_ids = model.generate(
                pixel_values=images,
                max_length=20,
                do_sample=True,  # Enable sampling
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=2.0
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            references.extend(tokenizer.batch_decode(captions, skip_special_tokens=True))
            predictions.extend(preds)

    avg_val_loss = val_loss / len(dataloader_valid)

    print("Ejemplos de predicción y anotación:")
    for i in range(min(5, len(predictions))):
        print(f"Predicción {i + 1}: '{predictions[i]}'")
        print(f"Anotación  {i + 1}: '{references[i]}'")
        print("-" * 50)

    # Si todas las predicciones están vacías, asignar 0 a las métricas
    if all(len(pred.strip()) == 0 for pred in predictions):
        print("Todas las predicciones están vacías. Se asigna BLEU = 0 y METEOR = 0.")
        bleu1 = {"bleu": 0.0}
        bleu2 = {"bleu": 0.0}
        meteor_score = {"meteor": 0.0}
        rouge_score = {"rougeL": 0.0}
    else:
        bleu1 = eval_bleu.compute(predictions=predictions, references=[[ref] for ref in references], max_order=1)
        bleu2 = eval_bleu.compute(predictions=predictions, references=[[ref] for ref in references], max_order=2)
        meteor_score = eval_meteor.compute(predictions=predictions, references=references)
        rouge_score = eval_rouge.compute(predictions=predictions, references=references)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "bleu1": bleu1["bleu"],
        "bleu2": bleu2["bleu"],
        "meteor": meteor_score["meteor"],
        "rouge": rouge_score["rougeL"],
        "lr": optimizer.param_groups[0]['lr']
    })

    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print(
        f"BLEU-1: {bleu1['bleu'] * 100:.2f}, BLEU-2: {bleu2['bleu'] * 100:.2f}, METEOR: {meteor_score['meteor'] * 100:.2f}, ROUGE-L: {rouge_score['rougeL'] * 100:.2f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "vit_gpt2_finetuned.pth")
        print("Modelo guardado (mejor validación)")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activado!")
            break

model.save_pretrained("vit_gpt2_finetuned")
tokenizer.save_pretrained("vit_gpt2_finetuned")

print("Entrenamiento finalizado y modelo guardado.")
wandb.finish()
