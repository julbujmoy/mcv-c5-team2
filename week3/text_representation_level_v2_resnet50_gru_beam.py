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
    return clean_text(str(text).lower())

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
TEXT_MAX_LEN = 50  # Reduced maximum length

# Initialize wandb with updated experiment name to reflect LSTM decoder
wandb.init(
    project="captioning_experiment",
    name="ResNet50-LSTM-wordpiece",
    config={
        "epochs": 40,
        "batch_size": 64,
        "learning_rate": 0.001,
        "model": "ResNet50-LSTM-wordpiece",
        "text_representation": text_representation
    }
)

# Define image folders for the dataset
train_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/train"
valid_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/valid"
test_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test"

# Load pre-split CSV annotation files
train_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/train_data.csv")
valid_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/valid_data.csv")
test_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_data.csv")

# Set up the tokenizer and special tokens according to the text representation
if text_representation == "char":
    all_text = "".join(
        pd.concat([train_annotations["Title"], valid_annotations["Title"], test_annotations["Title"]]).astype(str).values)
    unique_chars = sorted(set(all_text))
    vocab = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + unique_chars
    tokenizer = {ch: idx for idx, ch in enumerate(vocab)}
    detokenizer = {idx: ch for ch, idx in tokenizer.items()}
elif text_representation == "word":
    all_text = pd.concat([train_annotations["Title"], valid_annotations["Title"], test_annotations["Title"]]).astype(str).values
    all_words = set(word for title in all_text for word in title.split())
    vocab = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + sorted(all_words)
    tokenizer = {word: idx for idx, word in enumerate(vocab)}
    detokenizer = {idx: word for word, idx in tokenizer.items()}
elif text_representation == "wordpiece":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    special_tokens = {
        "<SOS>": tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.unk_token_id,
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

# Define the dataset class with low-quality caption filtering
class FoodDataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None):
        self.data = dataframe.copy()
        # Preprocess and clean captions; filter out captions with fewer than 3 tokens
        self.data["Title"] = self.data["Title"].apply(preprocess_annotation)
        self.data = self.data[self.data["Title"].str.split().apply(len) >= 3]
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
dataloaders = {k: DataLoader(v, batch_size=64, shuffle=(k == "train")) for k, v in datasets.items()}

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (batch, 1, attention_dim)
        att = self.full_att(torch.tanh(att1 + att2)).squeeze(2)  # (batch, num_pixels)
        alpha = torch.softmax(att, dim=1)  # (batch, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch, encoder_dim)
        return attention_weighted_encoding, alpha

# Definir el modelo con beam search in inference
class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
        self.encoder_dim = 2048
        self.decoder_dim = 512
        self.embed_dim = 512
        self.attention_dim = 256
        self.vocab_size = len(tokenizer)

        # Extraer features y proyectarlas al espacio del decodificador
        self.linear_feat = nn.Linear(self.encoder_dim, self.decoder_dim)

        # Se usa el vector proyectado para la atención (dimensión decoder_dim)
        self.attention = BahdanauAttention(self.decoder_dim, self.decoder_dim, self.attention_dim)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        # GRU con 1 capa
        self.gru = nn.GRU(self.embed_dim + self.decoder_dim, self.decoder_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

    def forward(self, img, captions=None, mode='train', max_len=TEXT_MAX_LEN, beam_width=1):
        batch_size = img.size(0)
        # Extraer features espaciales con ResNet
        resnet_output = self.resnet(img, output_hidden_states=True)
        spatial_feats = resnet_output.last_hidden_state
        if spatial_feats.dim() == 4:
            b, c, h, w = spatial_feats.shape
            spatial_feats = spatial_feats.view(b, c, h * w).permute(0, 2, 1)  # (batch, num_patches, encoder_dim)

        # Proyectar las features para obtener encoder_out (dimensión decoder_dim)
        encoder_out = self.linear_feat(spatial_feats)  # (batch, num_patches, decoder_dim)
        # Inicializar el estado oculto de la GRU (1 capa)
        h = self.init_hidden_state(batch_size)

        if mode == 'train':
            embeddings = self.embedding(captions)  # (batch, seq_len, embed_dim)
            outputs = []
            for t in range(captions.size(1)):
                # Atención usando el vector proyectado y el último estado oculto
                context, _ = self.attention(encoder_out, h[-1])
                gru_input = torch.cat([embeddings[:, t, :], context], dim=1).unsqueeze(
                    1)  # (batch, 1, embed_dim+decoder_dim)
                out, h = self.gru(gru_input, h)
                outputs.append(self.fc(out.squeeze(1)))
            outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, vocab_size)
            return outputs
        else:
            if beam_width > 1:
                outputs = []
                for i in range(batch_size):
                    h_i = h[:, i:i + 1, :]
                    encoder_out_i = encoder_out[i].unsqueeze(0)  # (1, num_patches, decoder_dim)
                    best_seq = self.beam_search(encoder_out_i, h_i, beam_width, max_len)
                    outputs.append(torch.tensor(best_seq, device=device))
                outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True,
                                                          padding_value=special_tokens["<PAD>"])
                return outputs
            else:
                outputs = []
                input_token = torch.full((batch_size, 1), special_tokens["<SOS>"], dtype=torch.long, device=device)
                embedded = self.embedding(input_token)
                for _ in range(max_len):
                    context, _ = self.attention(encoder_out, h[-1])
                    gru_input = torch.cat([embedded.squeeze(1), context], dim=1).unsqueeze(1)
                    out, h = self.gru(gru_input, h)
                    logits = self.fc(out.squeeze(1))
                    outputs.append(logits)
                    predicted = torch.argmax(logits, dim=1)
                    embedded = self.embedding(predicted.unsqueeze(1))
                outputs = torch.stack(outputs, dim=1)
                return outputs

    def beam_search(self, encoder_out, h, beam_width, max_len):
        # Inicializar el beam: (secuencia, score, estado oculto)
        beam = [([special_tokens["<SOS>"]], 0.0, h)]
        completed_beams = []
        for _ in range(max_len):
            new_beam = []
            for seq, score, h_i in beam:
                if seq[-1] == special_tokens["<EOS>"]:
                    completed_beams.append((seq, score))
                    continue
                last_token = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                embedded = self.embedding(last_token)  # (1, 1, embed_dim)
                context, _ = self.attention(encoder_out, h_i[-1])  # (1, decoder_dim)
                gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (1, 1, embed_dim+decoder_dim)
                out, h_new = self.gru(gru_input, h_i.contiguous())
                logits = self.fc(out.squeeze(1))  # (1, vocab_size)
                log_probs = torch.log_softmax(logits, dim=1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=1)
                for i in range(beam_width):
                    token_id = topk_indices[0, i].item()
                    token_log_prob = topk_log_probs[0, i].item()
                    new_seq = seq + [token_id]
                    new_score = score + token_log_prob
                    new_beam.append((new_seq, new_score, h_new))
            if not new_beam:
                break
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
        completed_beams.extend([(seq, score) for seq, score, _ in beam])
        best_seq = max(completed_beams, key=lambda x: x[1])[0]
        return best_seq

    def init_hidden_state(self, batch_size):
        # Para GRU con 1 capa
        h = torch.zeros(1, batch_size, self.decoder_dim).to(device)
        return h


model = CaptionModel().to(device)

# Use AdamW and a cosine annealing scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6, verbose=True)

# Loss function with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Load BLEU and METEOR metrics using evaluate
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")

best_val_loss = 0
patience = 10
counter = 0
warmup_epochs = 5
base_lr = 0.0001
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Warmup: incrementar el learning rate linealmente durante las primeras 'warmup_epochs'
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    model.train()
    total_loss = 0
    for imgs, captions in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1} Training"):
        imgs, captions = imgs.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, captions=captions, mode='train')
        loss = criterion(outputs.view(-1, model.vocab_size), captions.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloaders["train"])

    # Validación y actualización del scheduler después del warmup
    model.eval()
    val_loss = 0
    predictions = []
    references = []
    with torch.no_grad():
        for imgs, captions in tqdm(dataloaders["valid"]):
            imgs, captions = imgs.to(device), captions.to(device)

            # Paso 1: Cálculo de pérdida con teacher forcing
            outputs_loss = model(imgs, captions=captions, mode='train')
            val_loss += criterion(outputs_loss.view(-1, model.vocab_size), captions.view(-1)).item()

            # Paso 2: Generación de predicciones con beam search
            outputs_pred = model(imgs, mode='eval', beam_width=7)
            preds = outputs_pred  # outputs_pred contiene índices de tokens
            captions_np = captions.cpu().numpy()
            for pred_ids, ref_ids in zip(preds, captions_np):
                pred_text = decode_caption(list(pred_ids)).lower()
                ref_text = preprocess_annotation(decode_caption(list(ref_ids)))
                predictions.append(pred_text)
                references.append(ref_text)
    val_loss /= len(dataloaders["valid"])

    # Actualizar scheduler solo después del warmup
    if epoch >= warmup_epochs:
        scheduler.step(val_loss)

    print("Ejemplos de predicción y anotación (después del postprocesamiento):")
    for i in range(min(5, len(predictions))):
        print(f"Predicción {i + 1}: {predictions[i]}")
        print(f"Anotación  {i + 1}: {references[i]}")
        print("-" * 50)

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

    if bleu_score["bleu"] >= best_val_loss:
        best_val_loss = bleu_score["bleu"]
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activated!")
            break

wandb.finish()

