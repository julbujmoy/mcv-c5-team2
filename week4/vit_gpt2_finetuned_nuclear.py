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


# Initialize wandb
wandb.init(
    project="ViT-GPT2-FINETUNE-NUCLEAR",
    entity='laurasalort',
    name="vit_gpt2_finetuned_nuclear",
    config={
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-5,
        "model": "vit_gpt2_finetuned_nuclear"
    }
)
text_representation = "wordpiece"  # Options: "char", "word", "wordpiece"
TEXT_MAX_LEN = 30
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/train_filtered"
valid_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/valid_filtered"
test_img_folder = "/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_filtered"

train_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/train_filtered.csv", header=0)
valid_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/valid_filtered.csv", header=0)
test_annotations = pd.read_csv("/home/toukapy/Dokumentuak/Master CV/C5/FoodImages/FoodImages/test_filtered.csv", header=0)

# Load model
model_id = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
feature_extractor = ViTImageProcessor.from_pretrained(model_id)

# Freeze decoder
#gpt2_params = model.decoder.parameters()
#for param in gpt2_params:
#    param.requires_grad = False  # No se entrena GPT-2

#vit_params = model.encoder.parameters()
#for param in vit_params:
#    param.requires_grad = False  # No se entrena GPT-2

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
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Verifica si el tokenizer ya tiene tokens especiales
    if tokenizer.bos_token is None:
        # Agrega tokens especiales si no están definidos
        special_tokens_dict = {
            "bos_token": "<SOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>",
            "unk_token": "<UNK>"
        }
        tokenizer.add_special_tokens(special_tokens_dict)

    # Define tu diccionario de tokens especiales usando los IDs del tokenizer cargado
    special_tokens = {
        "<SOS>": tokenizer.bos_token_id,
        "<EOS>": tokenizer.eos_token_id,
        "<PAD>": tokenizer.pad_token_id,
        "<UNK>": tokenizer.unk_token_id
    }
    detokenizer = tokenizer.convert_ids_to_tokens


# No congelamos nada: tanto ViT como GPT-2 serán entrenados
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
pad_token = tokenizer['<PAD>'] if text_representation != "wordpiece" else special_tokens['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8, verbose=True)

# Load metrics
eval_bleu = evaluate.load("bleu")
eval_meteor = evaluate.load("meteor")
eval_rouge = evaluate.load("rouge")

default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),                # Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Custom dataset
# Define the dataset class
class FoodDataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None):
        self.img_folder = img_folder
        self.transform = transform if transform is not None else default_transform

        # Filter the dataframe so that only rows with an existing image file are kept.
        valid_rows = []
        for index, row in dataframe.iterrows():
            img_name = row["Image_Name"]
            if not isinstance(img_name, str):
                img_name = str(img_name)
            # Build the full file path (assuming images are stored as .jpg)
            file_path = os.path.join(self.img_folder, img_name) + ".jpg"
            if os.path.exists(file_path):
                valid_rows.append(row)
            else:
                print(f"Skipping missing image: {file_path}")

        # Create a new DataFrame with only the valid rows.
        self.data = pd.DataFrame(valid_rows)
        print(f"Initialized dataset with {len(self.data)} samples out of {len(dataframe)} total rows.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_name = self.data.iloc[idx]["Image_Name"]
        if not isinstance(img_name, str):
            img_name = str(img_name)
        img_path = os.path.join(self.img_folder, img_name) + ".jpg"

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

        # Add start/end tokens and pad/truncate
        if text_representation == "wordpiece":
            caption_idx = [special_tokens["<SOS>"]] + caption_tokens + [special_tokens["<EOS>"]]
            caption_idx = caption_idx[:TEXT_MAX_LEN] + [special_tokens["<PAD>"]] * (TEXT_MAX_LEN - len(caption_idx))
        else:
            caption_idx = [tokenizer["<SOS>"]] + caption_tokens + [tokenizer["<EOS>"]]
            caption_idx = caption_idx[:TEXT_MAX_LEN] + [tokenizer["<PAD>"]] * (TEXT_MAX_LEN - len(caption_idx))

        input_caption = torch.tensor(caption_idx[:-1], dtype=torch.long)
        return image, input_caption

# Load data
# Create datasets for each split
datasets = {
    "train": FoodDataset(train_annotations, train_img_folder, transform=default_transform),
    "valid": FoodDataset(valid_annotations, valid_img_folder, transform=val_transform),
    "test": FoodDataset(test_annotations, test_img_folder, transform=val_transform),
}

# Create DataLoaders; shuffle only the training data
dataloaders = {k: DataLoader(v, batch_size=config.batch_size, shuffle=(k == "train")) for k, v in datasets.items()}


# Training loop
num_epochs = config.epochs
patience = 15
best_val_loss = float("inf")
counter = 0


for epoch in range(num_epochs):
    model.train()
    loop = tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{num_epochs} Training")
    total_loss = 0
    for images, captions in loop:
        images, captions = images.to(device), captions.to(device)

        # Compute attention mask: 1 for non-pad tokens, 0 for pad tokens.
        decoder_attention_mask = (captions != pad_token).long()

        optimizer.zero_grad()
        outputs = model(pixel_values=images, labels=captions, decoder_attention_mask=decoder_attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(dataloaders["train"])

    # Validation
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
                    top_p=0.9,
                    temperature=0.8,
                    no_repeat_ngram_size=4,
                    repetition_penalty=2.2
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

    print("Ejemplos de predicción y anotación:")
    for i in range(min(5, len(predictions))):
        print(f"Predicción {i + 1}: '{predictions[i]}'")
        print(f"Anotación  {i + 1}: '{references_list[i]}'")
        print("-" * 50)

    # Si todas las predicciones están vacías, asignar 0 a las métricas
    if all(len(pred.strip()) == 0 for pred in predictions):
        print("Todas las predicciones están vacías. Se asigna BLEU = 0 y METEOR = 0.")
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
        "lr": optimizer.param_groups[0]['lr']
    })
    
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print(f"BLEU-1: {bleu1['bleu']*100:.2f}, BLEU-2: {bleu2['bleu']*100:.2f}, METEOR: {meteor_score['meteor']*100:.2f}, ROUGE-L: {rouge_score['rougeL']*100:.2f}")
    
    scheduler.step()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "vit_gpt2_finetuned_nuclear.pth")
        print("Modelo guardado (mejor validación)")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activado!")
            break

print("Entrenamiento finalizado y modelo guardado.")
wandb.finish()
