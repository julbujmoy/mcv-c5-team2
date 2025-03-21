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

# Configuración
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_folder = "FoodImages/images/"
csv_file = "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
text_representation = "wordpiece"  # Opciones: "char", "word", "wordpiece"

# Inicializar wandb
wandb.init(
    project="captioning_experiment",
    name="ResNet50-GRU-wordpiece",
    config={
        "epochs": 40,
        "batch_size": 32,
        "learning_rate": 0.001,
        "model": "ResNet50-GRU-wordpiece",
        "text_representation": text_representation
    }
)

# Cargar datos
data = pd.read_csv(csv_file)

# Filtrar nombres de imágenes inválidos
data = data[data["Image_Name"].apply(lambda x: isinstance(x, str) and not x.startswith("#"))]
data["Image_Name"] = data["Image_Name"].astype(str) + ".jpg"

# Filtrar imágenes que realmente existen
data = data[data["Image_Name"].apply(lambda x: os.path.exists(os.path.join(img_folder, x)))].reset_index(drop=True)

# Reemplazar valores NaN en la columna Title
data["Title"] = data["Title"].fillna("")

# Dividir en 80% entrenamiento, 10% validación, 10% prueba
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Manejo de diferentes niveles de representación del texto
if text_representation == "char":
    all_text = "".join(data["Title"].astype(str).values)
    unique_chars = sorted(set(all_text))
    vocab = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + unique_chars
    tokenizer = {word: idx for idx, word in enumerate(vocab)}
    detokenizer = {idx: word for word, idx in tokenizer.items()}

elif text_representation == "word":
    all_words = set(word for title in data["Title"].astype(str).values for word in title.split())
    vocab = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + sorted(all_words)
    tokenizer = {word: idx for idx, word in enumerate(vocab)}
    detokenizer = {idx: word for word, idx in tokenizer.items()}

elif text_representation == "wordpiece":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    special_tokens = {"<SOS>": tokenizer.unk_token_id, "<EOS>": tokenizer.sep_token_id, "<PAD>": tokenizer.pad_token_id, "<UNK>": tokenizer.unk_token_id}
    detokenizer = tokenizer.convert_ids_to_tokens

TEXT_MAX_LEN = 201

default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


class FoodDataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None):
        self.data = dataframe
        self.img_folder = img_folder
        self.transform = transform if transform else default_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image path and open the image
        img_name = self.data.iloc[idx]["Image_Name"]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Process caption text
        caption = str(self.data.iloc[idx]["Title"])

        if text_representation == "char":
            caption_tokens = [tokenizer.get(c, tokenizer['<UNK>']) for c in caption]
        elif text_representation == "word":
            caption_tokens = [tokenizer.get(word, tokenizer['<UNK>']) for word in caption.split()]
        elif text_representation == "wordpiece":
            caption_tokens = tokenizer.encode(caption, add_special_tokens=False)

        # Add special tokens and pad/truncate the caption sequence
        if text_representation == "wordpiece":
            caption_idx = [special_tokens['<SOS>']] + caption_tokens + [special_tokens['<EOS>']]
            caption_idx = caption_idx[:TEXT_MAX_LEN] + [special_tokens['<PAD>']] * (TEXT_MAX_LEN - len(caption_idx))
        else:
            caption_idx = [tokenizer['<SOS>']] + caption_tokens + [tokenizer['<EOS>']]
            caption_idx = caption_idx[:TEXT_MAX_LEN] + [tokenizer['<PAD>']] * (TEXT_MAX_LEN - len(caption_idx))

        input_caption = torch.tensor(caption_idx[:-1], dtype=torch.long)
        return image, input_caption


# Load CSV annotation files for each split
train_annotations = pd.read_csv("train_data.csv")
valid_annotations = pd.read_csv("valid_data.csv")
test_annotations = pd.read_csv("test_data.csv")

# Define image folders for each split – update these paths with your actual directories
train_img_folder = "path/to/train_images"
valid_img_folder = "path/to/valid_images"
test_img_folder = "path/to/test_images"

# Create datasets for each split
datasets = {
    "train": FoodDataset(train_annotations, train_img_folder),
    "valid": FoodDataset(valid_annotations, valid_img_folder),
    "test": FoodDataset(test_annotations, test_img_folder)
}

# Create DataLoaders with shuffling enabled only for the training set
dataloaders = {k: DataLoader(v, batch_size=32, shuffle=(k == "train")) for k, v in datasets.items()}

# Definir modelo
class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
        self.hidden_proj = nn.Linear(2048, 512)
        self.gru = nn.GRU(512, 512, num_layers=1, batch_first=True)
        self.proj = nn.Linear(512, len(tokenizer))
        self.embed = nn.Embedding(len(tokenizer), 512)

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img).pooler_output
        feat = self.hidden_proj(feat.view(batch_size, 1, -1)).permute(1, 0, 2)
        sos_token = tokenizer['<SOS>'] if text_representation != "wordpiece" else special_tokens['<SOS>']
        inp = self.embed(torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device))

        outputs = []
        for _ in range(TEXT_MAX_LEN - 1):
            out, hidden = self.gru(inp, feat)
            outputs.append(out)
            inp = out
        
        res = torch.cat(outputs, dim=1)
        return self.proj(res).permute(0, 2, 1)

# Inicializar modelo
model = CaptionModel().to(device)

# Configurar entrenamiento con Early Stopping
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_val_loss = float("inf")
patience = 5
counter = 0

# Entrenar modelo
for epoch in range(40):
    model.train()
    total_loss = 0
    for imgs, captions in dataloaders["train"]:
        imgs, captions = imgs.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, captions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloaders['train'])
    
    # Evaluación en validación
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, captions in dataloaders["valid"]:
            imgs, captions = imgs.to(device), captions.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, captions).item()
    val_loss /= len(dataloaders["valid"])

    wandb.log({"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": val_loss})

    print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

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
