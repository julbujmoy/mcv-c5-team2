import math
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
from torchvision import transforms
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smooth_fn = SmoothingFunction().method1

import unicodedata
import re


def normalize_text(text):
    # Elimina acentos y normaliza el texto a ASCII
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def clean_text(text):
    """
    Limpia el texto: normaliza, quita acentos, elimina guiones repetidos y espacios extra, y lo convierte a minúsculas.
    """
    text = normalize_text(text)
    text = re.sub(r'-{2,}', '-', text)
    text = re.sub(r'\s*-\s*', '-', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def preprocess_annotation(text):
    """
    Preprocesa la anotación: la convierte a minúsculas, limpia y normaliza.
    """
    return clean_text(text)


def postprocess_caption(caption):
    """
    Realiza un postprocesamiento a la caption generada para eliminar palabras duplicadas consecutivas.
    """
    words = caption.split()
    new_words = []
    for word in words:
        if not new_words or word != new_words[-1]:
            new_words.append(word)
    return " ".join(new_words)


def decode_caption(token_ids, tokenizer, detokenizer, text_representation, special_tokens):
    """
    Decodifica una secuencia de IDs de tokens a una cadena de texto, limpia y postprocesa la salida.
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

    # Limpieza final y postprocesamiento
    decoded = clean_text(decoded)
    decoded = postprocess_caption(decoded)
    return decoded


# Inicializar wandb
wandb.init(
    project="ViT-GPT2-FINETUNE",
    entity='laurasalort',
    name="vit_gpt2_finetuned",
    config={
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-4,  # LR aumentado
        "model": "vit_freeze_gpt2_finetuned"
    }
)
text_representation = "wordpiece"  # Options: "char", "word", "wordpiece"
TEXT_MAX_LEN = 20
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/train_filtered"
valid_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/valid_filtered"
test_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_filtered"

train_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/train_filtered.csv",
                                header=0)
valid_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/valid_filtered.csv",
                                header=0)
test_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_filtered.csv",
                               header=0)

# Cargar modelo
model_id = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
feature_extractor = ViTImageProcessor.from_pretrained(model_id)

# Congelar el encoder (ViT)
for param in model.encoder.parameters():
    param.requires_grad = False

if text_representation == "wordpiece":
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.bos_token is None:
        special_tokens_dict = {
            "bos_token": "<SOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>",
            "unk_token": "<UNK>"
        }
        tokenizer.add_special_tokens(special_tokens_dict)
    special_tokens = {
        "<SOS>": tokenizer.bos_token_id,
        "<EOS>": tokenizer.eos_token_id,
        "<PAD>": tokenizer.pad_token_id,
        "<UNK>": tokenizer.unk_token_id
    }
    detokenizer = tokenizer.convert_ids_to_tokens

# Optimización: LR = 1e-4 y weight_decay reducido
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.001)
pad_token = tokenizer['<PAD>'] if text_representation != "wordpiece" else special_tokens['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

# Scheduler: Usamos CosineAnnealingWarmRestarts con T_0=10 (más warm-up)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-7)

# Métricas
eval_bleu = evaluate.load("bleu")
eval_meteor = evaluate.load("meteor")
eval_rouge = evaluate.load("rouge")

default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# Dataset personalizado
class FoodDataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None):
        self.img_folder = img_folder
        self.transform = transform if transform is not None else default_transform
        valid_rows = []
        for index, row in dataframe.iterrows():
            img_name = row["Image_Name"]
            if not isinstance(img_name, str):
                img_name = str(img_name)
            file_path = os.path.join(self.img_folder, img_name) + ".jpg"
            if os.path.exists(file_path):
                valid_rows.append(row)
            else:
                print(f"Skipping missing image: {file_path}")
        self.data = pd.DataFrame(valid_rows)
        print(f"Initialized dataset with {len(self.data)} samples out of {len(dataframe)} total rows.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image_Name"]
        if not isinstance(img_name, str):
            img_name = str(img_name)
        img_path = os.path.join(self.img_folder, img_name) + ".jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption = str(self.data.iloc[idx]["Title"])
        if text_representation == "wordpiece":
            caption_tokens = tokenizer.encode(caption, add_special_tokens=False)
            caption_idx = [special_tokens["<SOS>"]] + caption_tokens + [special_tokens["<EOS>"]]
            caption_idx = caption_idx[:TEXT_MAX_LEN] + [special_tokens["<PAD>"]] * (TEXT_MAX_LEN - len(caption_idx))
        input_caption = torch.tensor(caption_idx[:-1], dtype=torch.long)
        return image, input_caption


datasets = {
    "train": FoodDataset(train_annotations, train_img_folder, transform=default_transform),
    "valid": FoodDataset(valid_annotations, valid_img_folder, transform=val_transform),
    "test": FoodDataset(test_annotations, test_img_folder, transform=val_transform),
}
dataloaders = {k: DataLoader(v, batch_size=config.batch_size, shuffle=(k == "train")) for k, v in datasets.items()}


# Función para actualizar alpha de forma cuadrática
def get_alpha(epoch, total_epochs, start=0.0, end=1.0):
    return start + (end - start) * ((epoch / total_epochs) ** 2)


patience = 15
best_val_loss = float("inf")
counter = 0
num_epochs = 100

for epoch in range(num_epochs):
    alpha = get_alpha(epoch, num_epochs, start=0.0, end=1.0)
    print(f"Epoch {epoch + 1}/{num_epochs} - Scheduled sampling alpha: {alpha:.4f}")

    model.train()
    total_loss = 0
    for images, captions in tqdm(dataloaders["train"], desc="Training"):
        images = images.to(device)
        captions = captions.to(device)

        # Pérdida con teacher forcing
        decoder_attention_mask = (captions != pad_token).long()
        outputs_ce = model(pixel_values=images, labels=captions, decoder_attention_mask=decoder_attention_mask)
        ce_loss = outputs_ce.loss

        # SCST: generar captions baseline (greedy) y muestreadas (sampling)
        baseline_ids = model.generate(
            pixel_values=images,
            max_length=TEXT_MAX_LEN,
            do_sample=False
        )
        baseline_texts = tokenizer.batch_decode(baseline_ids, skip_special_tokens=True)

        sample_ids = model.generate(
            pixel_values=images,
            max_length=TEXT_MAX_LEN,
            do_sample=True,
            top_p=0.7,
            temperature=0.9  # Temperatura reducida
        )
        sample_texts = tokenizer.batch_decode(sample_ids, skip_special_tokens=True)

        ref_texts = []
        for cap in captions:
            ref_texts.append(decode_caption(cap.cpu().tolist(), tokenizer, detokenizer,
                                            text_representation, special_tokens))

        # Cambiar la métrica de ranking a METEOR
        rewards_sample = []
        rewards_baseline = []
        for s_text, b_text, ref in zip(sample_texts, baseline_texts, ref_texts):
            r_sample = eval_meteor.compute(predictions=[s_text], references=[ref])["meteor"]
            r_baseline = eval_meteor.compute(predictions=[b_text], references=[ref])["meteor"]
            rewards_sample.append(r_sample)
            rewards_baseline.append(r_baseline)
        rewards_sample = torch.tensor(rewards_sample, device=device)
        rewards_baseline = torch.tensor(rewards_baseline, device=device)

        sample_ids_input = sample_ids[:, :-1]
        sample_ids_target = sample_ids[:, 1:]
        outputs_rl = model(pixel_values=images, decoder_input_ids=sample_ids_input, labels=sample_ids_target)
        logits = outputs_rl.logits
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        gathered_log_probs = log_probs.gather(2, sample_ids_target.unsqueeze(-1)).squeeze(-1)
        seq_log_probs = gathered_log_probs.sum(dim=1)

        scst_loss = -((rewards_sample - rewards_baseline) * seq_log_probs).mean()

        total_loss_batch = (1 - alpha) * ce_loss + alpha * scst_loss

        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total_loss_batch.item()

    avg_train_loss = total_loss / len(dataloaders["train"])

    # Validación: generación con nucleus sampling
    model.eval()
    val_loss = 0
    predictions = []
    references_list = []
    with torch.no_grad():
        for imgs, captions in tqdm(dataloaders["valid"], desc=f"Epoch {epoch + 1} Validation"):
            imgs, captions = imgs.to(device), captions.to(device)
            decoder_attention_mask = (captions != pad_token).long()
            outputs = model(pixel_values=imgs, labels=captions, decoder_attention_mask=decoder_attention_mask)
            val_loss += outputs.loss.item()

            for i in range(imgs.size(0)):
                generated_ids = model.generate(
                    pixel_values=imgs[i].unsqueeze(0),
                    decoder_input_ids=torch.tensor([[tokenizer.bos_token_id]], device=device),
                    decoder_attention_mask=torch.tensor([[1]], device=device),
                    max_length=TEXT_MAX_LEN,
                    do_sample=True,
                    top_p=0.7,
                    temperature=0.9,
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.0
                )
                pred_text = decode_caption(generated_ids.tolist()[0], tokenizer, detokenizer,
                                           text_representation, special_tokens).lower()
                ref_text = preprocess_annotation(
                    decode_caption(captions[i].cpu().tolist(), tokenizer, detokenizer, text_representation,
                                   special_tokens)
                )
                predictions.append(pred_text)
                references_list.append(ref_text)
    avg_val_loss = val_loss / len(dataloaders["valid"])

    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    for i in range(min(5, len(predictions))):
        print(f"Predicción {i + 1}: '{predictions[i]}'")
        print(f"Anotación  {i + 1}: '{references_list[i]}'")
        print("-" * 50)

    if all(len(pred.strip()) == 0 for pred in predictions):
        bleu1 = {"bleu": 0.0}
        bleu2 = {"bleu": 0.0}
        meteor_score = {"meteor": 0.0}
        rouge_score = {"rougeL": 0.0}
    else:
        bleu1 = eval_bleu.compute(predictions=predictions, references=[[ref] for ref in references_list], max_order=1)
        bleu2 = eval_bleu.compute(predictions=predictions, references=[[ref] for ref in references_list], max_order=2)
        meteor_score = eval_meteor.compute(predictions=predictions, references=references_list)
        rouge_score = eval_rouge.compute(predictions=predictions, references=references_list)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "bleu1": bleu1["bleu"],
        "bleu2": bleu2["bleu"],
        "meteor": meteor_score["meteor"],
        "rouge": rouge_score["rougeL"],
        "lr": optimizer.param_groups[0]['lr'],
        "alpha": alpha
    })

    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print(f"BLEU-1: {bleu1['bleu'] * 100:.2f}, BLEU-2: {bleu2['bleu'] * 100:.2f}, METEOR: {meteor_score['meteor'] * 100:.2f}, ROUGE-L: {rouge_score['rougeL'] * 100:.2f}")

    scheduler.step()  # CosineAnnealingWarmRestarts reinicia según T_0

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "vit_freeze_gpt2_finetuned.pth")
        print("Modelo guardado (mejor validación)")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activado!")
            break

model.save_pretrained("vit_freeze_gpt2_finetuned")
tokenizer.save_pretrained("vit_freeze_gpt2_finetuned")

print("Entrenamiento finalizado y modelo guardado.")
wandb.finish()

