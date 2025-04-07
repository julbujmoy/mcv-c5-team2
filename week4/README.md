# Transformer-based Image Captioning

This week, we continued with the project of image captioning. We had a baseline with a ViT-GPT2 architecture for task 1, and the Llama 3.2 models for task2. The aim of this project in task 1 was to finetune the encoder, the decoder, and the whole model trying to achieve better results in generating text that descrbed the picture. For task 2 the aim was to finetune Llama 3.2 1B and 3B using LoRA using a trained ViT, and also assess the perfomance of a pretrained large multimodal model, in our case DeepSeek-VL 7B

Link to the slides: [https://docs.google.com/presentation/d/1k3j3JOhO4CImBR6ymKt4p-C3FjSkIGPyGcAsiFlpjaM/edit?usp=sharing](https://docs.google.com/presentation/d/1_VTyKqShtiHdJu-5Lz6lIb9iYn4qzhcpEYAMDTXytzM/edit?usp=sharing)

The pdf of the paper is in the week4 folder.

## CLEANED DATA

In this project we cleaned the dataset in other to reduce noisy data. You can download it from this link:

https://drive.google.com/file/d/1z7B_C37B-HC8L95LO7s-p4Spv4QgFy1K/view?usp=sharing 

To clean these images, we used this scripts:

```
imageCleaning/preprocess.py
imageCleaning/remove_images.py
``` 

## TASK1

To run the first two approaches for this task use the scripts:

```
task1/vit_gpt2_attention.py
task1/vit_gpt2_teacherF.py
```

After data cleaning, we used this script:
```
task1/vit_gpt2_finetuned.py
```

Inside this script, you have commented options to freeze ViT entirely, GPT-2 entirely or just the deepest 4 layers of GPT-2.

Then, you have the Reinforcement Learning version of that script and the nuclear sampling version.

## TASK2

### DeepSeek-VL 7B
To run inference on this model use the script task2/deepseek.py

You will need some requirements for DeepSeek:

```
git clone https://github.com/deepseek-ai/DeepSeek-VL
cd DeepSeek-VL
pip install -e .
``` 

### Finetune Llama 3.2.
To finetune the 1B or 3B models use the main script task2/train.py. To change the Lora configuration, change the function add_peft in task2/llm.py. Make sure to input your token for huggingface to be able to access the models.

To run inference on the trained llama models, use the script task2/inferllama.py
