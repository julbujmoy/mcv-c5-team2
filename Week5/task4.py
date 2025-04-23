import torch
from diffusers import StableDiffusion3Pipeline
import os
import time
from huggingface_hub import login

# Log in to Hugging Face
login(token="MYTOKEN")

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model to use
model_id = "stabilityai/stable-diffusion-3.5-medium"

# Read prompts from file
with open("/export/home/c5mcv02/diffusion/food_images_new_2-2.txt", "r", encoding="utf-8") as f:
    PROMPTS = [line.strip() for line in f if line.strip()]

def sanitize_filename(prompt):
    # Remove invalid characters and replace spaces
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in prompt.strip().replace(" ", "_"))

def generate_images():
    print("Loading model...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)

    print("Generating images...")
    model_start = time.time()

    out_dir = "outputs_task4"
    os.makedirs(out_dir, exist_ok=True)

    for i, prompt in enumerate(PROMPTS):
        print(f" Prompt {i+1}/{len(PROMPTS)}: {prompt}")
        image = pipe(prompt=prompt).images[0]

        filename = sanitize_filename(prompt)
        image.save(os.path.join(out_dir, f"{i+1:02d}_{filename}.png"))

    total_time = time.time() - model_start
    print(f"\n Total time: {total_time:.2f} seconds")

    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    generate_images()
