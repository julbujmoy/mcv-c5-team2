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
    Preprocesa la anotación: la convierte a minúsculas y la limpia.
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


# Configuración
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Parameters and settings
text_representation = "wordpiece"  # Options: "char", "word", "wordpiece"
TEXT_MAX_LEN = 50

# Initialize wandb
wandb.init(
    project="captioning_experiment",
    name="ResNet50-GRU-wordpiece",
    config={
        "epochs": 40,
        "batch_size": 64,
        "learning_rate": 0.001,
        "model": "ResNet50-GRU-wordpiece",
        "text_representation": text_representation
    }
)

# Define image folders for the entire dataset (if needed) or per split.
# In this case, we use separate folders for each split:
train_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/train"
valid_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/valid"
test_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test"

# Load pre-split CSV annotation files
train_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/train_data.csv")
valid_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/valid_data.csv")
test_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_data.csv")

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

# Define separate transforms for training and validation.
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

# Dataset class
class FoodDataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None):
        self.data = dataframe
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image_Name"]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path + ".jpg").convert("RGB")
        image = self.transform(image)
        caption = str(self.data.iloc[idx]["Title"])
        if text_representation == "wordpiece":
            caption_tokens = tokenizer.encode(caption, add_special_tokens=False)
            caption_idx = [special_tokens["<SOS>"]] + caption_tokens + [special_tokens["<EOS>"]]
            caption_idx = caption_idx[:TEXT_MAX_LEN] + [special_tokens["<PAD>"]] * (TEXT_MAX_LEN - len(caption_idx))
        else:
            caption_idx = []
        input_caption = torch.tensor(caption_idx[:-1], dtype=torch.long)
        return image, input_caption

# Create datasets and dataloaders.
train_dataset = FoodDataset(train_annotations, train_img_folder, transform=train_transform)
valid_dataset = FoodDataset(valid_annotations, valid_img_folder, transform=val_transform)
test_dataset = FoodDataset(test_annotations, test_img_folder, transform=val_transform)

datasets = {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
dataloaders = {k: DataLoader(v, batch_size=32, shuffle=(k=="train")) for k, v in datasets.items()}


# Definir el modelo
class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, features, hidden):
        hidden = hidden.unsqueeze(1).repeat(1, features.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat([features, hidden], dim=2)))
        attention = self.v(energy).squeeze(2)
        weights = torch.softmax(attention, dim=1)
        context = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return context, weights

class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.attn = AdditiveAttention(2048, 512)
        self.embed = nn.Embedding(tokenizer.vocab_size, 512)
        self.gru = nn.GRU(512 + 2048, 512, batch_first=True)
        self.proj = nn.Linear(512, tokenizer.vocab_size)

    def forward(self, images):
        batch_size = images.size(0)
        features = self.resnet(images, output_hidden_states=True).last_hidden_state
        features = features.view(batch_size, -1, 2048)  # Flatten spatial dimensions
        hidden = torch.zeros(1, batch_size, 512).to(images.device)

        inp = torch.full((batch_size,), special_tokens["<SOS>"], dtype=torch.long, device=images.device)
        inp = self.embed(inp).unsqueeze(1)

        outputs = []
        for _ in range(TEXT_MAX_LEN - 1):
            context, _ = self.attn(features, hidden[-1])
            rnn_input = torch.cat([inp, context.unsqueeze(1)], dim=2)
            out, hidden = self.gru(rnn_input, hidden)
            logits = self.proj(out.squeeze(1))
            outputs.append(logits.unsqueeze(1))
            inp = self.embed(logits.argmax(dim=1)).unsqueeze(1)

        return torch.cat(outputs, dim=1).permute(0, 2, 1)

model = CaptionModel().to(device)

# Usar AdamW y scheduler ReduceLROnPlateau
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8, verbose=True)

criterion = nn.CrossEntropyLoss()

# Cargar métrica BLEU y METEOR usando evaluate
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")

best_val_loss = float("inf")
patience = 10
counter = 0

# Bucle de entrenamiento
for epoch in range(100):
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
    print("Ejemplos de predicción y anotación (después del postprocesamiento):")
    for i in range(min(5, len(predictions))):
        print(f"Predicción {i + 1}: {predictions[i]}")
        print(f"Anotación  {i + 1}: {references[i]}")
        print("-" * 50)

    # Si todas las predicciones están vacías, asignar 0 a las métricas
    if all(len(pred.strip()) == 0 for pred in predictions):
        print("Todas las predicciones están vacías. Se asigna BLEU = 0 y METEOR = 0.")
        bleu_score = {"bleu": 0.0}
        meteor_score = {"meteor": 0.0}
    else:
        bleu_score = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
        meteor_score = meteor_metric.compute(predictions=predictions, references=references)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_loss": val_loss,
        "bleu": bleu_score["bleu"],
        "meteor": meteor_score["meteor"],
        "lr": optimizer.param_groups[0]['lr']
    })
    print(
        f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu_score['bleu']:.4f}, METEOR: {meteor_score['meteor']:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activated!")
            break

wandb.finish()

