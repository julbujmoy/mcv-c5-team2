import numpy as np
import random
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from transformers import ResNetModel
from sklearn.model_selection import train_test_split
import wandb  

# Configuración
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_folder = "FoodImages/images/"
csv_file = "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

# Inicializar wandb
wandb.init(
    project="captioning_experiment",
    name="ResNet50-GRU",
    config={
        "epochs": 40,
        "batch_size": 32,
        "learning_rate": 0.001,
        "model": "ResNet50-GRU"
    }
)

# Cargar datos
data = pd.read_csv(csv_file)
data = data[data["Image_Name"].apply(lambda x: isinstance(x, str) and not x.startswith("#"))]
data["Image_Name"] = data["Image_Name"].astype(str) + ".jpg"
data = data[data["Image_Name"].apply(lambda x: os.path.exists(os.path.join(img_folder, x)))].reset_index(drop=True)
data["Title"] = data["Title"].fillna("")

# Dividir datos
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Tokenización
all_text = "".join(data["Title"].astype(str).values)
unique_chars = sorted(set(all_text))
chars = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + unique_chars
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
TEXT_MAX_LEN = 201

# Dataset
class FoodDataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None):
        self.data = dataframe
        self.img_folder = img_folder
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image_Name"]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption = str(self.data.iloc[idx]["Title"])
        caption_idx = [char2idx['<SOS>']] + [char2idx.get(c, char2idx['<UNK>']) for c in caption] + [char2idx['<EOS>']]
        caption_idx = caption_idx[:TEXT_MAX_LEN] + [char2idx['<PAD>']] * (TEXT_MAX_LEN - len(caption_idx))
        input_caption = torch.tensor(caption_idx[:-1], dtype=torch.long)
        return image, input_caption

# DataLoaders
datasets = {"train": FoodDataset(train_data, img_folder), "valid": FoodDataset(valid_data, img_folder)}
dataloaders = {k: DataLoader(v, batch_size=32, shuffle=(k == "train")) for k, v in datasets.items()}

# Modelo
class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
        self.hidden_proj = nn.Linear(2048, 512)
        self.gru = nn.GRU(512, 512, num_layers=1, batch_first=True)
        self.proj = nn.Linear(512, len(chars))
        self.embed = nn.Embedding(len(chars), 512)
    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img).pooler_output
        feat = feat.view(batch_size, 1, -1)
        hidden = self.hidden_proj(feat).permute(1, 0, 2)
        inp = self.embed(torch.full((batch_size, 1), char2idx['<SOS>'], dtype=torch.long, device=device))
        outputs = []
        for _ in range(TEXT_MAX_LEN - 1):
            out, hidden = self.gru(inp, hidden)
            outputs.append(out)
            inp = out
        res = torch.cat(outputs, dim=1)
        return self.proj(res).permute(0, 2, 1)

# Inicializar modelo
model = CaptionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento con Early Stopping
def train(model, dataloaders, criterion, optimizer, epochs=40, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        for imgs, captions in dataloaders["train"]:
            imgs, captions = imgs.to(device), captions.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, captions)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(dataloaders['train'])
        
        # Validación
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, captions in dataloaders["valid"]:
                imgs, captions = imgs.to(device), captions.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, captions)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(dataloaders['valid'])
        model.train()
        
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_recipe_model_resnet50_gru.pth")
            print("Nuevo mejor modelo guardado.")
        else:
            epochs_no_improve += 1
            print(f"Early stopping count: {epochs_no_improve}/{patience}")
        
        if epochs_no_improve >= patience:
            print("No mejora en validación por 5 épocas, deteniendo entrenamiento.")
            break
    wandb.save("best_recipe_model_resnet50_gru.pth")

# Entrenar
train(model, dataloaders, criterion, optimizer, epochs=40, patience=5)
wandb.finish()

