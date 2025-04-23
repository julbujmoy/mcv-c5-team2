import time
import torch
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline,StableDiffusionXLPipeline
import os

#model_id = "stabilityai/stable-diffusion-xl-base-1.0"
model_id="stabilityai/stable-diffusion-3.5-medium"

device = "cuda" if torch.cuda.is_available() else "cpu"

from huggingface_hub import login
login(token="MYTOKEN")

# List of positive and negative prompts
NEGATIVE_PROMPTS = [
    "No cluttered table, no messy background",
    "No cluttered table, no messy background",
    "No cluttered table, no messy background",
    "No cluttered table, no messy background",
    "No cluttered table, no messy background",
    "No cluttered table, no messy background",
    "No cluttered table, no messy background",
    "No cluttered table, no messy background",
    "No cluttered table, no messy background",
    "No cluttered table, no messy background",
]


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


# Function to generate images with various configurations
def generate_images(setting_name, value_list, change_param):
    for value in value_list:
        print(f"\nRunning experiment: {setting_name} = {value}")


        # Load pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        #pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

        pipe = pipe.to(device)
        
        # Adjust scheduler only if model id is "stabilityai/stable-diffusion-xl-base-1.0"
        if model_id== "stabilityai/stable-diffusion-xl-base-1.0":

            if change_param == "scheduler":
                if value == "DDPM":
                    pipe.scheduler = pipe.scheduler.from_config(pipe.scheduler.config)
                elif value == "DDIM":
                    from diffusers import DDIMScheduler
                    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                    
                elif value=="LMSDiscreteScheduler":
                    from diffusers import LMSDiscreteScheduler
                    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
                
                elif value=="PNDMScheduler":
                    from diffusers import PNDMScheduler
                    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

        # Set seed for reproducibility
        torch.manual_seed(SEED)

        start_time = time.time()

        for i, prompt in enumerate(PROMPTS):
            
            kwargs = {
                "prompt": prompt
            }


            if change_param == "cfg":
                kwargs["guidance_scale"] = value
            elif change_param == "steps":
                kwargs["num_inference_steps"] = value
            elif change_param == "negative_prompt" and value:
                kwargs["negative_prompt"]=NEGATIVE_PROMPTS[i]
            # Scheduler already handled above

            image = pipe(**kwargs).images[0]

            out_dir = f"outputs/{setting_name}_{value}"
            os.makedirs(out_dir, exist_ok=True)
            image.save(f"{out_dir}/{i+1:02d}.png")

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")


# Run experiments
if __name__ == "__main__":
    generate_images("cfg", [0,1,5, 10, 15], change_param="cfg")
    generate_images("steps", [5,10,25, 50, 75], change_param="steps")
    generate_images("negative_prompt", [True], change_param="negative_prompt")
    # generate_images("scheduler", ["DDIM","LMSDiscreteScheduler","PNDMScheduler"], change_param="scheduler")