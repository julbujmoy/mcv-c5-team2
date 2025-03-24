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
    Preprocesa la anotacion: la convierte a minesculas y la limpia.
    """
    return clean_text(text.lower())


def decode_caption(token_ids, tokenizer, detokenizer, text_representation, special_tokens):
    """
    Decodifica una secuencia de IDs de tokens a una cadena de texto y la limpia.
    """
    if not token_ids:
        return ""
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
        decoded = "".join(decoded_tokens).strip() if text_representation == "char" else " ".join(decoded_tokens).strip()
    return clean_text(decoded)


# Configuracion
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Parameters and settings
text_representation = "wordpiece"  # Options: "char", "word", "wordpiece"
TEXT_MAX_LEN = 100

# Initialize wandb
wandb.init(
    project="LSTM-attention",
    entity='laurasalort',
    name="ResNet50-lstm1-attention",
    config={
        "epochs": 100,
        "batch_size": 32,
        "layers":1,
        "learning_rate": 0.001,
        "model": "ResNet50-lstm1-attention",
        "text_representation": text_representation
    }
)
config=wandb.config
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


# Additive Attention Module
class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, features, hidden):
        if features.dim() == 4:
            features = features.flatten(2).permute(0, 2, 1)  # (B, N, D)
        hidden = hidden.unsqueeze(1).repeat(1, features.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat([features, hidden], dim=2)))
        attention = self.v(energy).squeeze(2)
        weights = torch.softmax(attention, dim=1)
        context = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return context, weights

# Definir el modelo con atención y teacher forcing
class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.embed = nn.Embedding(len(tokenizer), 256)
        self.lstm = nn.LSTM(256 + 2048, 512, num_layers=1, batch_first=True)
        self.proj = nn.Linear(512, len(tokenizer))
        self.attn = AdditiveAttention(2048, 512)

    def forward(self, img, captions=None, teacher_forcing_ratio=0.2):
        batch_size = img.size(0)
        features = self.resnet(img, output_hidden_states=True).last_hidden_state

        hidden = torch.zeros(config.layers, batch_size, 512).to(img.device)
        cell = torch.zeros(config.layers, batch_size, 512).to(img.device)

        sos_token = tokenizer['<SOS>'] if text_representation != "wordpiece" else special_tokens['<SOS>']
        inp_token = torch.full((batch_size,), sos_token, dtype=torch.long, device=img.device)
        inp = self.embed(inp_token).unsqueeze(1)

        outputs = []
        for t in range(TEXT_MAX_LEN - 1):
            context, _ = self.attn(features, hidden[-1])
            rnn_input = torch.cat([inp, context.unsqueeze(1)], dim=2)
            out, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
            logits = self.proj(out.squeeze(1))
            outputs.append(logits.unsqueeze(1))

            if captions is not None and random.random() < teacher_forcing_ratio:
                next_input_token = captions[:, t]
            else:
                next_input_token = logits.argmax(dim=1)
            inp = self.embed(next_input_token).unsqueeze(1)

        return torch.cat(outputs, dim=1).permute(0, 2, 1)

    def generate_beam(self, img, beam_width=7, max_len=TEXT_MAX_LEN):
        device = img.device
        features = self.resnet(img, output_hidden_states=True).last_hidden_state
        hidden = torch.zeros(config.layers, 1, 512).to(device)
        cell = torch.zeros(config.layers, 1, 512).to(device)
        sos_token = tokenizer['<SOS>'] if text_representation != "wordpiece" else special_tokens['<SOS>']
        eos_token = tokenizer['<EOS>'] if text_representation != "wordpiece" else special_tokens['<EOS>']

        beams = [([sos_token], 0.0, hidden, cell)]

        for _ in range(max_len - 1):
            new_beams = []
            for seq, score, h, c in beams:
                if seq[-1] == eos_token:
                    new_beams.append((seq, score, h, c))
                    continue
                inp = self.embed(torch.tensor([seq[-1]], device=device)).unsqueeze(1)
                context, _ = self.attn(features, h[-1])
                rnn_input = torch.cat([inp, context.unsqueeze(1)], dim=2)
                out, (h_new, c_new) = self.lstm(rnn_input, (h, c))
                logits = self.proj(out.squeeze(1))
                probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                topk_probs, topk_idxs = probs.topk(beam_width)
                for k in range(beam_width):
                    token = topk_idxs[k].item()
                    prob = topk_probs[k].item()
                    new_seq = seq + [token]
                    new_score = score + prob
                    new_beams.append((new_seq, new_score, h_new, c_new))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        best_seq = beams[0][0]
        # Truncate at EOS if present
        if eos_token in best_seq:
            best_seq = best_seq[:best_seq.index(eos_token)]
        return best_seq

model = CaptionModel().to(device)

# Usar AdamW y scheduler ReduceLROnPlateau
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6, verbose=True)

pad_token = tokenizer['<PAD>'] if text_representation != "wordpiece" else special_tokens['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

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
        for imgs, captions in tqdm(dataloaders["valid"], desc=f"Epoch {epoch + 1} Validation"):
            imgs, captions = imgs.to(device), captions.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, captions).item()
            preds = torch.argmax(outputs, dim=1)  # (batch, seq_len)
            preds = preds.cpu().numpy()
            captions_np = captions.cpu().numpy()
            for i in range(imgs.size(0)):
                seq = model.generate_beam(imgs[i].unsqueeze(0), beam_width=3)
                pred_text = decode_caption(seq, tokenizer, detokenizer, text_representation, special_tokens).lower()
                ref_text = preprocess_annotation(decode_caption(list(captions[i].cpu().numpy()),
                                                                tokenizer, detokenizer, text_representation,
                                                                special_tokens))
                predictions.append(pred_text)
                references.append(ref_text)
    val_loss /= len(dataloaders["valid"])

    # Imprimir algunas muestras antes de calcular BLEU y METEOR
    print("Ejemplos de prediccion y anotacion (despues del postprocesamiento):")
    for i in range(min(5, len(predictions))):
        print(f"Prediccion {i + 1}: '{predictions[i]}'")
        print(f"Anotacion  {i + 1}: '{references[i]}'")
        print("-" * 50)

    # Si todas las predicciones estan vacias, asignar 0 a las metricas
    if all(len(pred.strip()) == 0 for pred in predictions):
        print("Todas las predicciones estan vacias. Se asigna BLEU = 0 y METEOR = 0.")
        bleu1 = {"bleu": 0.0}
        bleu2 = {"bleu": 0.0}
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
        torch.save(model.state_dict(), "model_final.pth")
        model_artifact = wandb.Artifact('model_final', type='model')
        model_artifact.add_file('model_final.pth')
        wandb.log_artifact(model_artifact)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activated!")
            break

wandb.finish()
