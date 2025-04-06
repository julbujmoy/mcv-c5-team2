from transformers import AutoModelForCausalLM
import os
import evaluate
from tqdm import tqdm
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import ResNetModel, AutoTokenizer

from torchvision import transforms
from PIL import Image
import os

import evaluate

from tqdm import tqdm
import re  # Para expresiones regulares

default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
text_representation = "word"
TEXT_MAX_LEN = 201

# Define the dataset class
class FoodDataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None):
        self.data = dataframe[dataframe["Image_Name"].apply(lambda img: os.path.exists(os.path.join(img_folder, str(img) + ".jpg")))].reset_index(drop=True)
        self.img_folder = img_folder
        self.transform = transform if transform else default_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_name = self.data.iloc[idx]["Image_Name"]

        # Process caption text
        caption = str(self.data.iloc[idx]["Title"])
        
        return img_name, caption

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

test_path="/export/home/c5mcv02/laura_w5/FoodImages_filtered/test_filtered/"
csv_path = "/export/home/c5mcv02/laura_w5/FoodImages_filtered/test_filtered.csv"  # Path to your CSV file

# Load dataset
captions_df = pd.read_csv(csv_path)
dataset = FoodDataset(captions_df, test_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Load one image at a time


# Convert dataframe to dictionary for fast lookup
captions_dict = dict(zip(captions_df["Image_Name"], captions_df["Title"]))

predictions = []
references = []

bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")

for img_name, reference_caption in tqdm(dataloader):
    img_name = img_name[0] 
    if isinstance(reference_caption, tuple):
        reference_caption = reference_caption[0]  # Extract the string from the tuple
    reference_caption = str(reference_caption).strip("(),'")
    
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>Put a short title to the dish in this image",
            "images": [os.path.join(test_path, img_name + ".jpg")]
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=20,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    answer = answer.replace('"', '').strip()  # Remove double quotes

    reference_caption = str(reference_caption).replace('"', '').strip()  # Remove quotes from references

    print(f"ANNOTATION: {reference_caption}")
    print(f"PREDICTION: {answer}")
    print('-----------------------------------------------')
    predictions.append(answer)
    references.append([reference_caption])

bleu1 = bleu_metric.compute(predictions=predictions, references=references, max_order=1)
bleu2 = bleu_metric.compute(predictions=predictions, references=references, max_order=2)
meteor_score = meteor_metric.compute(predictions=predictions, references=references)
rouge_score = rouge_metric.compute(predictions=predictions, references=references)
print(f"BLEU1: {bleu1['bleu']*100:.1f}, BLEU2: {bleu2['bleu']*100:.1f}, ROUGE-L: {rouge_score['rougeL']*100:.1f}, METEOR: {meteor_score['meteor']*100:.1f}")
