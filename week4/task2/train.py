import llm
import anymodal
import torch
import vision
from torch.utils.data import DataLoader
import schedulefree
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torch.amp import GradScaler
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import evaluate
from PIL import Image
import wandb
from huggingface_hub import login
login(token="MY ACCESS TOKEN")

wandb.init(
    project="LORA-LLAMA1B",
    entity='laurasalort',
    name="llama1b",

    config={
        "epochs": 100,
        "batch_size": 1,
        "learning_rate": 1e-5,  # LR aumentado
        "model": "llama1b"
    }
)

# Load language model and tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='MY ACCESS TOKEN'
)
llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)


# Load vision model components
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=False)

# PATHS
train_img_folder = "/export/home/c5mcv02/laura_w5/FoodImages_filtered/train_filtered"
valid_img_folder = "/export/home/c5mcv02/laura_w5/FoodImages_filtered/valid_filtered"
test_img_folder = "/export/home/c5mcv02/laura_w5/FoodImages_filtered/test_filtered"

train_annotations = pd.read_csv("/export/home/c5mcv02/laura_w5/FoodImages_filtered/train_filtered.csv",
                                header=0)
valid_annotations = pd.read_csv("/export/home/c5mcv02/laura_w5/FoodImages_filtered/valid_filtered.csv",
                                header=0)
test_annotations = pd.read_csv("/export/home/c5mcv02/laura_w5/FoodImages_filtered/test_filtered.csv",
                               header=0)

# Define the dataset class
class FoodDataset(Dataset):
    def __init__(self, dataframe, processor, image_root):
        self.data = dataframe[dataframe["Image_Name"].apply(lambda img: os.path.exists(os.path.join(image_root, str(img) + ".jpg")))].reset_index(drop=True)
        self.processor = processor
        self.image_root = image_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_name = self.data.iloc[idx]["Image_Name"]
        # Process caption text
        caption = str(self.data.iloc[idx]["Title"])
        return img_name, caption
    

    def __getitem__(self, idx):
        """
        Retrieves a single data item, including the processed image and corresponding text label.
        
        Parameters:
        - idx: int, index of the item to retrieve.
        
        Returns:
        - dict: Contains processed image, selected text, and the RGB numpy array.
        """
        img_name = self.data.iloc[idx]["Image_Name"]
        caption = str(self.data.iloc[idx]["Title"])

        
        image_path = os.path.join(self.image_root, img_name + ".jpg")
        
        image = Image.open(image_path).convert('RGB')

        rgb_val = np.array(image.resize((224, 224), Image.BICUBIC))
        image_inputs = self.processor(image, return_tensors="pt")
        image_inputs = {key: val.squeeze(0) for key, val in image_inputs.items()}

        return {
            'input': image_inputs,
            'text': caption,
            'image': rgb_val
        }
    
config = wandb.config
train_dataset = FoodDataset(train_annotations, image_processor,train_img_folder)
val_dataset = FoodDataset(valid_annotations, image_processor,valid_img_folder)
test_dataset = FoodDataset(test_annotations, image_processor,test_img_folder)

train_size = len(train_dataset)
val_size = len(val_dataset)


# DataLoader configuration
batch_size = 6
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

train_size = len(train_loader)
val_size = len(val_loader)
print(f"Train size: {train_size}, Validation size: {val_size}")


# Initialize vision tokenizer and encoder
vision_encoder = vision.VisionEncoder(vision_model)
vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)

# Initialize MultiModalModel
multimodal_model = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    lm_peft = llm.add_peft,
    prompt_text="The name of the dish in the image is: ")

# multimodal_model.language_model = llm.add_peft(multimodal_model.language_model)

# Training configuration
num_epochs = config.epochs
patience = 5
best_score =-float("inf")
counter = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multimodal_model = multimodal_model.to(device)
multimodal_model.train()

# Optimizer
optimizer = schedulefree.AdamWScheduleFree(multimodal_model.parameters(), lr=config.learning_rate)
optimizer.train()

scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8, verbose=True)
os.makedirs("image_captioning_model", exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    training_losses = []
    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} Training", leave=False):
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = multimodal_model(batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        training_losses.append(loss.item())
    
    avg_train_loss = sum(training_losses) / len(training_losses)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
    
    # Validation
    multimodal_model.eval()
    validation_losses = []
    predictions = []
    references = []
    
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    rouge_metric = evaluate.load("rouge")

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Epoch {epoch+1} Validation", leave=False):
            logits, loss = multimodal_model(batch)
            validation_losses.append(loss.item())
            
            for i in range(len(batch['text'])):
                answer=multimodal_model.generate({'pixel_values':batch['input']['pixel_values'][i,:,:,:]}, max_new_tokens=50)
                if batch_idx<6:
                  print("Actual Text: ", batch['text'][i].replace("<|end_of_text|>", "").strip())
                  print("Generated Text: ", answer)
                  print("-----------------------")
                
                predictions.append(answer)
                references.append([batch['text'][i].replace("<|end_of_text|>", "").strip()])
            
        avg_val_loss = sum(validation_losses) / len(validation_losses)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

  
    
    bleu1 = bleu_metric.compute(predictions=predictions, references=references, max_order=1)
    bleu2 = bleu_metric.compute(predictions=predictions, references=references, max_order=2)
    meteor_score = meteor_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)
    print(f"Epoch {epoch+1}: BLEU1: {bleu1['bleu']*100:.1f}, BLEU2: {bleu2['bleu']*100:.1f}, ROUGE-L: {rouge_score['rougeL']*100:.1f}, METEOR: {meteor_score['meteor']*100:.1f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "bleu1": bleu1["bleu"],
        "bleu2": bleu2["bleu"],
        "meteor": meteor_score["meteor"],
        "rouge": rouge_score["rougeL"],
        "lr": optimizer.param_groups[0]['lr'],
        
    })
    multimodal_model.train()

    scheduler.step()

    # Save the model
    score = (
            0.25 * bleu1["bleu"] +  
            0.25 * bleu2["bleu"] +
            0.25 * meteor_score["meteor"] +
            0.25 * rouge_score["rougeL"]
        )
    if score > best_score:
        count=0
        best_score = score
        multimodal_model._save_model("/export/home/c5mcv02/laura_w5/task2/final_model/")

        # torch.save(multimodal_model.state_dict(), "finetune-llama1b.pth")
        # model_artifact = wandb.Artifact('finetune-llama1b', type='model')
        # model_artifact.add_file('finetune-llama1b.pth',overwrite=True)
        # wandb.log_artifact(model_artifact,aliases=["latest"])
        print("Modelo guardado (mejor validacion)")

    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activado!")
            break



multimodal_model.eval()

        
        
predictions = []
references = []
    
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")

with torch.no_grad():
   
  for sample_idx in range(len(test_dataset)):
    sample = test_dataset[sample_idx]
    answer=multimodal_model.generate(sample['input'], max_new_tokens=120)
    
    print("Actual Text: ", sample['text'])
    print("Generated Text: ", answer)
    print("-----------------------")
    predictions.append(answer)
    references.append([sample['text']])
  
bleu1 = bleu_metric.compute(predictions=predictions, references=references, max_order=1)
bleu2 = bleu_metric.compute(predictions=predictions, references=references, max_order=2)
meteor_score = meteor_metric.compute(predictions=predictions, references=references)
rouge_score = rouge_metric.compute(predictions=predictions, references=references)
print(f"TEST RESUTS: BLEU1: {bleu1['bleu']*100:.1f}, BLEU2: {bleu2['bleu']*100:.1f}, ROUGE-L: {rouge_score['rougeL']*100:.1f}, METEOR: {meteor_score['meteor']*100:.1f}")
        
     
wandb.finish()
