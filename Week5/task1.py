import torch
from huggingface_hub import login
login(token="MYTOKEN")

from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler,AutoPipelineForText2Image, StableDiffusionXLPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
import os
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model IDs and loading logic
MODELS = {
    "sd21": ("stabilityai/stable-diffusion-2-1", StableDiffusionPipeline),
    "sd21-turbo": ("stabilityai/sd-turbo", AutoPipelineForText2Image),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", StableDiffusionXLPipeline),
    "sdxl-turbo": ("stabilityai/sdxl-turbo", DiffusionPipeline),
    "sd35-medium": ("stabilityai/stable-diffusion-3.5-medium", DiffusionPipeline),
    "sd35-large-turbo": ("stabilityai/stable-diffusion-3.5-large-turbo", DiffusionPipeline),
}

PROMPTS = [
    "A plate of scrambled eggs, crispy bacon, and toast with butter, served on a white plate with orange juice in the background, morning sunlight.",
    "A bowl of steaming ramen with sliced pork, boiled egg, green onions, and seaweed, placed on a wooden table with chopsticks beside it.",
    "A slice of layered strawberry shortcake with whipped cream and fresh strawberries, served on a ceramic dessert plate with a fork.",
    "A colorful salad with arugula, cherry tomatoes, cucumbers, red onion, and olives, drizzled with vinaigrette, served in a clear glass bowl.",
    "A bowl of chicken tikka masala with basmati rice on the side, garnished with fresh coriander, set on a wooden table with naan bread nearby.",
    "A juicy cheeseburger with lettuce, tomato, and melted cheese in a sesame seed bun, served with fries on a metal tray.",
    "A flaky butter croissant on a white plate next to a cappuccino with latte art, placed on a small wooden table in a cozy cafe setting.",
    "Two soft corn tacos filled with grilled chicken, salsa, onions, and cilantro, served with lime wedges on a street-style paper tray.",
    "A traditional Japanese bento box with rice, tamagoyaki, karaage chicken, and vegetables, neatly arranged in compartments.",
    "A vibrant smoothie bowl topped with banana slices, blueberries, chia seeds, and coconut flakes, served in a ceramic bowl on a marble surface.",
]

# Defaults
SEED = 12345


def generate_images():
    for model_name, (model_id, pipeline_class) in MODELS.items():
        print(f"Loading model: {model_name}")
        
        import torch
        load_start = time.time()
        
        if model_name=='sd21':
          pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
          pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif model_name=='sd21-turbo':
          pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        elif model_name=='sdxl':
          pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        elif model_name=='sdxl-turbo':
          pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        elif model_name=='sd35-medium':
          from diffusers import StableDiffusion3Pipeline
          pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        elif model_name=='sd35-large-turbo':
          from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
          from diffusers import StableDiffusion3Pipeline
          from transformers import T5Model, T5EncoderModel

          import torch
          
          model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
          
          nf4_config = BitsAndBytesConfig(
              load_in_4bit=True,
              bnb_4bit_quant_type="nf4",
              bnb_4bit_compute_dtype=torch.bfloat16
          )
          model_nf4 = SD3Transformer2DModel.from_pretrained(
              model_id,
              subfolder="transformer",
              quantization_config=nf4_config,
              torch_dtype=torch.bfloat16
          )
          
          t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)
          
          pipe = StableDiffusion3Pipeline.from_pretrained(
              model_id, 
              transformer=model_nf4,
              text_encoder_3=t5_nf4,
              torch_dtype=torch.bfloat16
          )
          pipe.enable_model_cpu_offload()



        pipe = pipe.to(device)
        
        # Set seed for reproducibility
        torch.manual_seed(SEED)
        
        load_time = time.time() - load_start
        print(f"Loaded in {load_time:.2f} seconds")
        model_start = time.time()
        
        # Some models like SDXL-turbo and SD3.5-large-turbo require "prompt" as a string, not list
        for i, prompt in enumerate(PROMPTS):
            print(f"  [{model_name}] Prompt {i+1}/{len(PROMPTS)}")
            t0 = time.time()
            image = pipe(prompt=prompt).images[0]
            t1 = time.time()
            print(f" Done in {t1 - t0:.2f}s")

            out_dir = f"outputs/{model_name}"
            os.makedirs(out_dir, exist_ok=True)
            image.save(f"{out_dir}/{i+1:02d}.png")

        total_time = time.time() - model_start
        print(f"\n? Total time for {model_name}: {total_time:.2f} seconds")

        del pipe
        torch.cuda.empty_cache()

if __name__ == "__main__":
    generate_images()
