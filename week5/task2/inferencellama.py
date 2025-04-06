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
from torchvision import transforms
import evaluate
from PIL import Image

from huggingface_hub import login
login(token="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")

# Load language model and tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B",
    access_token="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",
    use_peft=False,
)
llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

# Load vision model components
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder("google/vit-base-patch16-224", use_peft=False)

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
    input_start_token="<|imstart|>",
    input_end_token="<|imend|>",
    prompt_text="The name of dish in the image is: ",
)

# Download pre-trained model weights
if not os.path.exists("image_captioning_model"):
    os.makedirs("image_captioning_model")

#hf_hub_download("AnyModal/Image-Captioning-Llama-3.2-3B", filename="input_tokenizer.pt", local_dir="image_captioning_model")
multimodal_model._load_model("/export/home/c5mcv02/laura_w5/task2/final_model/")









# PATHS

test_img_folder = "/export/home/c5mcv02/laura_w5/FoodImages_filtered/test_filtered"

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
    
test_dataset = FoodDataset(test_annotations, image_processor,test_img_folder)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multimodal_model = multimodal_model.to(device)

multimodal_model.eval()
predictions = []
references = []
    
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")

with torch.no_grad():
   
  for sample_idx in range(len(test_dataset)):
    sample = test_dataset[sample_idx]
    answer=multimodal_model.generate(sample['input'], max_new_tokens=20)
    
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