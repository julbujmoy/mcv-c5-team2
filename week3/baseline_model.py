import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import random
from transformers import ResNetModel
import evaluate
import os
from torchvision.models import resnet18

# Definición de parámetros y clases
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar los caracteres para las captions
chars = ['<SOS>', '<EOS>', '<PAD>', '<UNK>', ' ', '!', '‘', '~' ,'—', '"', "”","“", '–',  '’','#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'é', 'í', 'ó', 'ú', 'à', 'è', 'ì', 'ò', 'ù', 'â', 'ê', 'ë', 'î', 'ô', 'û', 'ç', 'ü', 'ñ', '¿', '¡', 'ö', 'ï']

NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

TEXT_MAX_LEN = 201

# Definición del Dataset
from torchvision import transforms

class FoodRecipeDataset(Dataset):
    def __init__(self, data, img_dir, num_captions=5, max_len=TEXT_MAX_LEN):
        self.data = data
        self.img_dir = img_dir
        self.num_captions = num_captions
        self.max_len = max_len
        self.img_proc = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        img_name = item['Image_Name']
        img_path = f'{self.img_dir}/{img_name}.jpg'
        
        # Cargar la imagen
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            return None
        
        img = self.img_proc(img)  # Procesar la imagen
        
        # Procesar la caption (título de la receta)
        caption = item['Title']
        if not isinstance(caption, str) or pd.isna(caption):  # Si es NaN, lo reemplazamos
            caption = ""
            
        cap_list = list(caption)
        final_list = [chars[0]]  # <SOS>
        final_list.extend(cap_list)
        final_list.extend([chars[1]])  # <EOS>
        gap = self.max_len - len(final_list)
        final_list.extend([chars[2]] * gap)  # <PAD>
        cap_idx = []
        for char in final_list:
            if char in char2idx:
               cap_idx.append(char2idx[char]) 
            else:
                cap_idx.append(char2idx[' '])  # Add <UNK> for unknown characters

        return img, torch.tensor(cap_idx)


# Definición del Modelo
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        
        # Ensure the feature tensor has the correct shape for the GRU (1, batch_size, 512)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)  # Shape: (1, batch_size, 512)
        
        # Start token for caption generation
        start = torch.tensor(char2idx['<SOS>']).to(DEVICE)
        start_embed = self.embed(start)  # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)  # (1, batch_size, 512)
        
        inp = start_embeds
        hidden = feat
        
        # Iterate over time steps
        for t in range(TEXT_MAX_LEN - 1):
            out, hidden = self.gru(inp, hidden)  # inp shape: (seq_len, batch, input_size), hidden shape: (num_layers, batch, hidden_size)
            
            # Ensure out[-1:] is the right shape (1, batch_size, hidden_size)
            inp = torch.cat((inp, out[-1:].squeeze(0).unsqueeze(0)), dim=0)  # Concatenate the output to form the next input
        
        # Project the output to the vocabulary size (NUM_CHAR)
        res = inp.permute(1, 0, 2)  # batch, seq, 512
        res = self.proj(res)  # batch, seq, NUM_CHAR
        res = res.permute(0, 2, 1)  # batch, NUM_CHAR, seq
        
        return res

def decode_captions(caption_indices):
    """Decodes a tensor of caption indices into text."""
    captions = []
    for caption in caption_indices:
        caption = caption.reshape(-1)  # Use reshape instead of view
        decoded_caption = []
        for idx in caption:
            idx_value = idx.item()
            # Check if the index is valid (in range of char2idx)
            if idx_value in char2idx.values() and idx_value != char2idx['<PAD>']:
                decoded_caption.append(idx2char[idx_value])
            else:
                decoded_caption.append(' ')  # Append an unknown token for invalid indices
        caption_str = ''.join(decoded_caption).replace('<SOS>', '').replace('<EOS>', '').strip()
        captions.append(caption_str)
    return captions



# Función de evaluación
def eval_epoch(model, crit, bleu, meteor, rouge, dataloader):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        # Wrap the dataloader with tqdm for progress tracking
        for imgs, captions in tqdm(dataloader, desc="Evaluating", ncols=100):
            imgs, captions = imgs.to(DEVICE), captions.to(DEVICE)
            outputs = model(imgs)
            outputs = outputs.reshape(-1, NUM_CHAR)  # Use reshape instead of view
            captions = captions.reshape(-1)
            
            loss = crit(outputs, captions)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)
# Función de entrenamiento por época
from tqdm import tqdm  # Import tqdm for progress bar

def train_one_epoch(model, optimizer, crit, bleu, meteor, rouge, train_loader):
    model.train()
    model = model.to(DEVICE)
    total_loss = 0
    all_predictions = []
    all_references = []
    scaler = torch.cuda.amp.GradScaler()
    
    for batch_idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", ncols=100):
        # Forward pass
        imgs, captions = imgs.to(DEVICE), captions.to(DEVICE)
        optimizer.zero_grad()
    
        with torch.cuda.amp.autocast(): 
            outputs = model(imgs)

             # Compute loss (your existing code for loss computation here)
            loss = crit(outputs, captions)
        total_loss += loss.item()
        
        scaler.scale(loss).backward()  
        scaler.step(optimizer)
        scaler.update()

    
    return total_loss / len(train_loader)

from torch.nn.utils.rnn import pad_sequence

# Function to collate a batch of samples
def collate_fn(batch):
    imgs, captions = zip(*batch)
    
    # Pad the captions (assuming captions are of variable length)
    padded_captions = pad_sequence(captions, batch_first=True, padding_value=char2idx['<PAD>'])

    # Stack images into a batch (already tensorized)
    imgs = torch.stack(imgs, dim=0)
    
    return imgs, padded_captions

# Entrenar el modelo
def train(EPOCHS):
    # Cargar los datos
    train_data = pd.read_csv('FoodImages/train_data.csv')
    valid_data = pd.read_csv('FoodImages/valid_data.csv')
    test_data = pd.read_csv('FoodImages/test_data.csv')
    
    train_dataset = FoodRecipeDataset(train_data, img_dir='FoodImages/train')
    valid_dataset = FoodRecipeDataset(valid_data, img_dir='FoodImages/valid')
    test_dataset = FoodRecipeDataset(test_data, img_dir='FoodImages/test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
   # Inicializar el modelo, optimizador y función de pérdida
    model = Model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss(ignore_index=char2idx['<PAD>'])
    
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    # Entrenamiento
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, optimizer, crit, bleu, meteor, rouge, train_loader)
        print(f'Train Loss: {train_loss:.2f}')
        
        # Validation (optional)
        val_loss = eval_epoch(model, crit, bleu, meteor, rouge, valid_loader)
        print(f'Validation Loss: {val_loss:.2f}')
        

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), "recipe_model.pth")
    print("Modelo guardado como recipe_model.pth")

# Entrenar el modelo durante 5 épocas
train(EPOCHS=20)
