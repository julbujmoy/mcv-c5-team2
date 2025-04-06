# Transformer-based Image Captioning

This week, we continued with the project of image captioning. We had a baseline with a ViT-GPT2 architecture for task 1, and the Llama 3.2 models for task2. The aim of this project in task 1 was to finetune the encoder, the decoder, and the whole model trying to achieve better results in generating text that descrbed the picture. For task 2 the aim was to finetune Llama 3.2 1B and 3B using LoRA using a trained ViT, and also assess the perfomance of a pretrained large multimodal model, in our case DeepSeek-VL 7B

Link to the slides: https://docs.google.com/presentation/d/1k3j3JOhO4CImBR6ymKt4p-C3FjSkIGPyGcAsiFlpjaM/edit?usp=sharing

The pdf of the paper is in the week4 folder.

## CLEANED DATA

## TASK1

## TASK2

### DeepSeek-VL 7B
To run inference on this model use the script deepseek.py

You will need some requirements for DeepSeek:

```
git clone https://github.com/deepseek-ai/DeepSeek-VL
cd DeepSeek-VL
pip install -e .
``` 

### Finetune Llama 3.2.
To finetune the 1B or 3B models use the main script train.py. To change the Lora configuration, change the function add_peft in llm.py. Make sure to input your token for huggingface to be able to access the models.
