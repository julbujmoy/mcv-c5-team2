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
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# Definición de parámetros y clases
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar los caracteres para las captions
chars = ['<SOS>', '<EOS>', '<PAD>', '<UNK>', ' ', '!', '"', "”","“", '–',  '’','#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'é', 'í', 'ó', 'ú', 'à', 'è', 'ì', 'ò', 'ù', 'â', 'ê', 'î', 'ô', 'û', 'ç', 'ü', 'ñ', '¿', '¡', 'ö']

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
        if not isinstance(caption, str):
            print(f"Warning: Caption at index {idx} is {caption} of type {type(caption)}")
    
            caption = str(caption)
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
                cap_idx.append(char2idx['<UNK>'])  # Add <UNK> for unknown characters
        
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
                decoded_caption.append('<UNK>')  # Append an unknown token for invalid indices
        caption_str = ''.join(decoded_caption).replace('<SOS>', '').replace('<EOS>', '').strip()
        captions.append(caption_str)
    return captions

# Function to collate a batch of samples
def collate_fn(batch):
    imgs, captions = zip(*batch)
    
    # Pad the captions (assuming captions are of variable length)
    padded_captions = pad_sequence(captions, batch_first=True, padding_value=char2idx['<PAD>'])

    # Stack images into a batch (already tensorized)
    imgs = torch.stack(imgs, dim=0)
    
    return imgs, padded_captions

# Función para evaluar el modelo con métricas
def eval_model():
    # Cargar datos de test
    test_data = pd.read_csv('FoodImages/test_data.csv')
    test_dataset = FoodRecipeDataset(test_data, img_dir='FoodImages/test')
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Cargar el modelo entrenado
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("recipe_model.pth", map_location=DEVICE))
    model.eval()

    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for imgs, captions in tqdm(test_loader, desc="Evaluating", ncols=100):
            imgs, captions = imgs.to(DEVICE), captions.to(DEVICE)
            outputs = model(imgs)

            predicted_captions = decode_captions(outputs.argmax(dim=1).cpu())
            reference_captions = decode_captions(captions.cpu())

            all_predictions.extend(predicted_captions)
            all_references.extend(reference_captions)

    # Calcular métricas
    bleu_1_score = bleu.compute(predictions=all_predictions, references=all_references, max_order=1)["bleu"]
    bleu_2_score = bleu.compute(predictions=all_predictions, references=all_references, max_order=2)["bleu"]
    meteor_score = meteor.compute(predictions=all_predictions, references=all_references)["meteor"]
    rouge_score = rouge.compute(predictions=all_predictions, references=all_references)["rougeL"]


    # Mostrar resultados
    print(f'BLEU-1: {bleu_1_score:.4f}, BLEU-2: {bleu_2_score:.4f}, ROUGE-L: {rouge_score:.4f}, METEOR: {meteor_score:.4f}')


if __name__ == "__main__":
    eval_model()
