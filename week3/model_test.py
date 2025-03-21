import torch
import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ResNetModel
import evaluate
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_folder = "FoodImages/images/"
csv_file = "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
model_path = "recipe_model_julia.pth"

# Cargar datos
data = pd.read_csv(csv_file)

# Filtrar imágenes existentes
data["Image_Name"] = data["Image_Name"].astype(str) + ".jpg"
data = data[data["Image_Name"].apply(lambda x: os.path.exists(os.path.join(img_folder, x)))].reset_index(drop=True)

# Dividir en 80% entrenamiento, 10% validación, 10% prueba
_, temp_data = train_test_split(data, test_size=0.2, random_state=42)
_, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Extraer caracteres únicos
all_text = "".join(data["Title"].astype(str).values)
unique_chars = sorted(set(all_text))
chars = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + unique_chars
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
TEXT_MAX_LEN = 201

# Dataset de prueba
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
        return image, torch.tensor(caption_idx[:-1], dtype=torch.long)

# DataLoader de prueba
test_dataset = FoodDataset(test_data, img_folder)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definir el modelo
class CaptionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-18").to(device)
        self.gru = torch.nn.GRU(512, 512, num_layers=1, batch_first=True)
        self.proj = torch.nn.Linear(512, len(chars))
        self.embed = torch.nn.Embedding(len(chars), 512)

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img).pooler_output
        feat = feat.view(batch_size, 1, -1)
        start = torch.full((batch_size, 1), char2idx['<SOS>'], dtype=torch.long, device=device)
        inp = self.embed(start)
        hidden = feat.permute(1, 0, 2)
        outputs = []
        for _ in range(TEXT_MAX_LEN - 1):
            out, hidden = self.gru(inp, hidden)
            outputs.append(out)
            inp = out
        res = torch.cat(outputs, dim=1)
        return self.proj(res).permute(0, 2, 1)

# Cargar el modelo correctamente según el dispositivo
model = CaptionModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Inicializar métricas
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

# Función para generar captions
def generate_caption(model, image):
    model.eval()
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted_indices = output.argmax(1).squeeze(0)
        predicted_caption = "".join([idx2char[idx.item()] for idx in predicted_indices if idx.item() not in [char2idx['<SOS>'], char2idx['<EOS>'], char2idx['<PAD>']]])
    return predicted_caption

# Evaluar el modelo
all_predictions = []
all_references = []

for imgs, captions in tqdm(test_dataloader, desc="Procesando batches", leave=True):
    imgs = imgs.to(device)
    for i in range(len(imgs)):
        pred_caption = generate_caption(model, imgs[i])
        true_caption = "".join([idx2char[idx.item()] for idx in captions[i] if idx.item() not in [char2idx['<SOS>'], char2idx['<EOS>'], char2idx['<PAD>']]])

        all_predictions.append(pred_caption)
        all_references.append([true_caption])

print("Ejemplo de predicciones y referencias:")
for i in range(5):  # Mostrar 5 ejemplos
    print(f"Predicción {i+1}: {all_predictions[i]}")
    print(f"Referencia  {i+1}: {all_references[i]}")
    print("-" * 50)

# Calcular métricas
bleu_1_score = bleu.compute(predictions=all_predictions, references=all_references, max_order=1)["bleu"]
bleu_2_score = bleu.compute(predictions=all_predictions, references=all_references, max_order=2)["bleu"]
meteor_score = meteor.compute(predictions=all_predictions, references=all_references)["meteor"]
rouge_score = rouge.compute(predictions=all_predictions, references=all_references)["rougeL"]

# Mostrar resultados
print(f'BLEU-1: {bleu_1_score:.4f}, BLEU-2: {bleu_2_score:.4f}, ROUGE-L: {rouge_score:.4f}, METEOR: {meteor_score:.4f}')