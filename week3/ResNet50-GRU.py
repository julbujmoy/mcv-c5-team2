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
import evaluate
import wandb  # Importar wandb

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

# Extraer caracteres únicos del dataset
all_text = "".join(data["Title"].astype(str).values)
unique_chars = sorted(set(all_text))
chars = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + unique_chars
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
TEXT_MAX_LEN = 201

# Definir dataset
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

        # Convertir título a secuencia de índices
        caption = str(self.data.iloc[idx]["Title"])  # Asegurar que sea string
        caption_idx = [char2idx['<SOS>']] + [char2idx.get(c, char2idx['<UNK>']) for c in caption] + [char2idx['<EOS>']]
        caption_idx = caption_idx[:TEXT_MAX_LEN] + [char2idx['<PAD>']] * (TEXT_MAX_LEN - len(caption_idx))
        
        # Ajustar tamaño para CrossEntropyLoss (quitar último token)
        input_caption = torch.tensor(caption_idx[:-1], dtype=torch.long)  # 200 tokens
        
        return image, input_caption

# Crear DataLoaders
datasets = {
    "train": FoodDataset(train_data, img_folder),
    "valid": FoodDataset(valid_data, img_folder),
    "test": FoodDataset(test_data, img_folder)
}
dataloaders = {k: DataLoader(v, batch_size=32, shuffle=(k == "train")) for k, v in datasets.items()}

# Definir el modelo
class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)  # Cambiar a ResNet-50
        self.hidden_proj = nn.Linear(2048, 512)  # 🔹 Convertimos la salida de ResNet a 512
        self.gru = nn.GRU(512, 512, num_layers=1, batch_first=True)  # Ajustar tamaño del GRU
        self.proj = nn.Linear(512, len(chars))
        self.embed = nn.Embedding(len(chars), 512)

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img).pooler_output  # Extraer la salida correcta
        feat = feat.view(batch_size, 1, -1)  # Ajustar dimensiones a (batch, 1, 2048)

        hidden = self.hidden_proj(feat).permute(1, 0, 2)  # 🔹 Ahora tiene (1, batch, 512)
        inp = self.embed(torch.full((batch_size, 1), char2idx['<SOS>'], dtype=torch.long, device=device))

        outputs = []
        for _ in range(TEXT_MAX_LEN - 1):  # Generar solo 200 tokens
            out, hidden = self.gru(inp, hidden)
            outputs.append(out)
            inp = out
        
        res = torch.cat(outputs, dim=1)  # batch, seq, 512
        return self.proj(res).permute(0, 2, 1)  # batch, vocab_size, seq_len

# Inicializar modelo
model = CaptionModel().to(device)

# Configurar entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Función de entrenamiento
def train(model, dataloaders, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
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
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # Registrar en wandb
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    # Guardar el modelo
    torch.save(model.state_dict(), "recipe_model_resnet50.pth")
    print("Modelo guardado como recipe_model_resnet50.pth")

    # Subir el modelo a wandb
    wandb.save("recipe_model_resnet50.pth")

# Entrenar el modelo
train(model, dataloaders, criterion, optimizer, epochs=40)

# Finalizar wandb
wandb.finish()
