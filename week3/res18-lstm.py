import numpy as np
import random
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from transformers import ResNetModel, AutoTokenizer
from sklearn.model_selection import train_test_split
import evaluate
import wandb
import sentencepiece as spm
from tqdm import tqdm
import re  # Para expresiones regulares

wandb.login(relogin=True,key="25e409ae656293628fa4bbb0cbd85a104c68d33d")


def clean_text(text):
    """
    Limpia el texto eliminando guiones repetidos, espacios extra y espacios alrededor de guiones.
    """
    text = re.sub(r'-{2,}', '-', text)
    text = re.sub(r'\s*-\s*', '-', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_annotation(text):
    """
    Preprocesa la anotacion: la convierte a minesculas y la limpia.
    """
    return clean_text(text.lower())


def decode_caption(token_ids):
    """
    Decodifica una secuencia de IDs de tokens en una cadena de texto y la limpia.
    """
    if not token_ids:
        return "--"
    if text_representation == "wordpiece":
        if token_ids[0] == special_tokens["<SOS>"]:
            token_ids = token_ids[1:]
        eos_token = special_tokens["<EOS>"]
        decoded_ids = []
        for tid in token_ids:
            if tid == eos_token:
                break
            decoded_ids.append(tid)
        decoded = tokenizer.decode(decoded_ids, skip_special_tokens=True).strip()
    else:
        if token_ids[0] == tokenizer["<SOS>"]:
            token_ids = token_ids[1:]
        eos_token = tokenizer["<EOS>"]
        decoded_tokens = []
        for tid in token_ids:
            if tid == eos_token:
                break
            decoded_tokens.append(detokenizer.get(tid, "<UNK>"))
        if text_representation == "char":
            decoded = "".join(decoded_tokens).strip()
        else:
            decoded = " ".join(decoded_tokens).strip()
    return clean_text(decoded)


# Configuracion
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Parameters and settings
text_representation = "char"  # Options: "char", "word", "wordpiece"
TEXT_MAX_LEN = 201

# Initialize wandb
wandb.init(
    project="baseline",
    name="ResNet18-lstm",
    config={
        "epochs": 100,
        "batch_size": 32,
        "layers":1,
        "learning_rate": 0.001,
        "model": "ResNet18-lstm",
        "text_representation": text_representation
    }
)
config=wandb.config
# Define image folders for the entire dataset (if needed) or per split.
# In this case, we use separate folders for each split:
train_img_folder = "C:/Users/User/Documents/MASTER/c5/FoodImages/FoodImages/train"
valid_img_folder = "C:/Users/User/Documents/MASTER/c5/FoodImages/FoodImages/valid"
test_img_folder = "C:/Users/User/Documents/MASTER/c5/FoodImages/FoodImages/test"

# Load pre-split CSV annotation files
train_annotations = pd.read_csv("C:/Users/User/Documents/MASTER/c5/FoodImages/FoodImages/train_data.csv")
valid_annotations = pd.read_csv("C:/Users/User/Documents/MASTER/c5/FoodImages/FoodImages/valid_data.csv")
test_annotations = pd.read_csv("C:/Users/User/Documents/MASTER/c5/FoodImages/FoodImages/test_data.csv")

# Set up the tokenizer and special tokens according to the text representation
if text_representation == "char":
    # Combine all titles from all splits
    all_text = "".join(
        pd.concat([train_annotations["Title"], valid_annotations["Title"], test_annotations["Title"]]).astype(
            str).values)
    unique_chars = sorted(set(all_text))
    vocab = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + unique_chars
    tokenizer = {ch: idx for idx, ch in enumerate(vocab)}
    detokenizer = {idx: ch for ch, idx in tokenizer.items()}
elif text_representation == "word":
    all_text = pd.concat([train_annotations["Title"], valid_annotations["Title"], test_annotations["Title"]]).astype(
        str).values
    all_words = set(word for title in all_text for word in title.split())
    vocab = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + sorted(all_words)
    tokenizer = {word: idx for idx, word in enumerate(vocab)}
    detokenizer = {idx: word for word, idx in tokenizer.items()}
elif text_representation == "wordpiece":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    special_tokens = {
        "<SOS>": tokenizer.unk_token_id,
        "<EOS>": tokenizer.sep_token_id,
        "<PAD>": tokenizer.pad_token_id,
        "<UNK>": tokenizer.unk_token_id
    }
    detokenizer = tokenizer.convert_ids_to_tokens

# Define the default image transformation
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# Define the dataset class
class FoodDataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None):
        self.data = dataframe
        self.img_folder = img_folder
        self.transform = transform if transform else default_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_name = self.data.iloc[idx]["Image_Name"]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path+".jpg").convert("RGB")
        image = self.transform(image)

        # Process caption text
        caption = str(self.data.iloc[idx]["Title"])
        if text_representation == "char":
            caption_tokens = [tokenizer.get(c, tokenizer['<UNK>']) for c in caption]
        elif text_representation == "word":
            caption_tokens = [tokenizer.get(word, tokenizer['<UNK>']) for word in caption.split()]
        elif text_representation == "wordpiece":
            caption_tokens = tokenizer.encode(caption, add_special_tokens=False)

        # Add start/end tokens and pad/truncate
        if text_representation == "wordpiece":
            caption_idx = [special_tokens["<SOS>"]] + caption_tokens + [special_tokens["<EOS>"]]
            caption_idx = caption_idx[:TEXT_MAX_LEN] + [special_tokens["<PAD>"]] * (TEXT_MAX_LEN - len(caption_idx))
        else:
            caption_idx = [tokenizer["<SOS>"]] + caption_tokens + [tokenizer["<EOS>"]]
            caption_idx = caption_idx[:TEXT_MAX_LEN] + [tokenizer["<PAD>"]] * (TEXT_MAX_LEN - len(caption_idx))

        input_caption = torch.tensor(caption_idx[:-1], dtype=torch.long)
        return image, input_caption


# Create datasets for each split
datasets = {
    "train": FoodDataset(train_annotations, train_img_folder),
    "valid": FoodDataset(valid_annotations, valid_img_folder),
    "test": FoodDataset(test_annotations, test_img_folder)
}

# Create DataLoaders; shuffle only the training data
dataloaders = {k: DataLoader(v, batch_size=config.batch_size, shuffle=(k == "train")) for k, v in datasets.items()}


# Definir el modelo
class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-18").to(device)
        # self.hidden_proj = nn.Linear(2048, 512)
        self.lstm = nn.LSTM(512, 512, num_layers=config.layers, batch_first=True)
        self.proj = nn.Linear(512, len(tokenizer))
        self.embed = nn.Embedding(len(tokenizer), 512)

    def forward(self, img):
        batch_size = img.shape[0]
        # Obtener la representacion de la imagen con ResNet
        feat = self.resnet(img).pooler_output  # (batch, 2048)
        # feat = self.hidden_proj(feat.view(batch_size, 1, -1))  # (batch, 1, 512)
        feat = feat.view(batch_size, 1, -1) #for resnet18 no projection
        # Convertir a forma (1, batch, 512) y replicar para num_layers = 5
        hidden = feat.permute(1, 0, 2).repeat(self.lstm.num_layers, 1, 1)  # (5, batch, 512)

        # Inicializar con el token de inicio (SOS)
        sos_token = tokenizer['<SOS>'] if text_representation != "wordpiece" else special_tokens['<SOS>']
        inp = self.embed(torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device))  # (batch, 1, 512)

        outputs = []
        for _ in range(TEXT_MAX_LEN - 1):
            # out, hidden = self.gru(inp, hidden)  # out: (batch, 1, 512), hidden: (5, batch, 512)
            out, (hidden_state, cell_state) = self.lstm(inp, (hidden, torch.zeros_like(hidden)))
            outputs.append(out)
            inp = out  # usar la salida como entrada en el siguiente paso

        res = torch.cat(outputs, dim=1)  # (batch, seq_len, 512)
        return self.proj(res).permute(0, 2, 1)  # (batch, vocab_size, seq_len)


model = CaptionModel().to(device)

# Usar AdamW y scheduler ReduceLROnPlateau
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

criterion = nn.CrossEntropyLoss()

# Cargar metrica BLEU y METEOR usando evaluate
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")

best_val_loss = float("inf")
patience = 10
counter = 0
# Bucle de entrenamiento
for epoch in range(config.epochs):
    model.train()
    total_loss = 0
    for imgs, captions in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1} Training"):
        imgs, captions = imgs.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)  # (batch, vocab_size, seq_len)
        loss = criterion(outputs, captions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloaders["train"])

    model.eval()
    val_loss = 0
    predictions = []
    references = []
    with torch.no_grad():
        for imgs, captions in dataloaders["valid"]:
            imgs, captions = imgs.to(device), captions.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, captions).item()
            preds = torch.argmax(outputs, dim=1)  # (batch, seq_len)
            preds = preds.cpu().numpy()
            captions_np = captions.cpu().numpy()
            for pred_ids, ref_ids in zip(preds, captions_np):
                pred_text = decode_caption(list(pred_ids)).lower()
                ref_text = preprocess_annotation(decode_caption(list(ref_ids)))
                predictions.append(pred_text)
                references.append(ref_text)
    val_loss /= len(dataloaders["valid"])

    # Imprimir algunas muestras antes de calcular BLEU y METEOR
    print("Ejemplos de prediccion y anotacion (despues del postprocesamiento):")
    for i in range(min(5, len(predictions))):
        print(f"Prediccion {i + 1}: {predictions[i]}")
        print(f"Anotacion  {i + 1}: {references[i]}")
        print("-" * 50)

    # Si todas las predicciones estan vacias, asignar 0 a las metricas
    if all(len(pred.strip()) == 0 for pred in predictions):
        print("Todas las predicciones estan vacias. Se asigna BLEU = 0 y METEOR = 0.")
        bleu1 = {"bleu1": 0.0}
        bleu2 = {"bleu2": 0.0}
        meteor_score = {"meteor": 0.0}
        rouge_score = {"rougeL": 0.0}
    else:
        bleu1 = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references], max_order=1)
        bleu2 = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references], max_order=2)
        meteor_score = meteor_metric.compute(predictions=predictions, references=references)
        rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_loss": val_loss,
        "bleu1": bleu1["bleu"],
        "bleu2": bleu2["bleu"],
        "meteor": meteor_score["meteor"],
        "rouge": rouge_score["rougeL"],
        "lr": optimizer.param_groups[0]['lr']
    })
    print(
        f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU1: {bleu1['bleu']*100:.1f}, BLEU2: {bleu2['bleu']*100:.1f}, ROUGE-L: {rouge_score['rougeL']*100:.1f}, METEOR: {meteor_score['meteor']*100:.1f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        model_artifact = wandb.Artifact('model_final', type='model')
        model_artifact.add_file('model_final.pth')
        wandb.log_artifact(model_artifact)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activated!")
            break

wandb.finish()
