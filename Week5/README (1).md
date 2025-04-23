# Diffusion models and how we can use them


This week, we continued with the project of image captioning. We had a baseline with a ViT-GPT2 architecture for task 1, and the Llama 3.2 models for task2. The aim of this project in task 1 was to finetune the encoder, the decoder, and the whole model trying to achieve better results in generating text that descrbed the picture. For task 2 the aim was to finetune Llama 3.2 1B and 3B using LoRA using a trained ViT, and also assess the perfomance of a pretrained large multimodal model, in our case DeepSeek-VL 7B

Link to the slides: [https://docs.google.com/presentation/d/1J2cKVrkPPwtxlbx2JqLngvT9srhs1DttaTs8hgPI70w/edit?usp=sharing](https://docs.google.com/presentation/d/1J2cKVrkPPwtxlbx2JqLngvT9srhs1DttaTs8hgPI70w/edit?usp=sharing)



## TASK A

Play with opensourced models: Install Stable Diffusion (SD) and run the script to see the experiments on Stable-diffusion 2.1, Stable-diffusion 2.1 turbo, Stable-diffusion XL, Stable-diffusion XL turbo, Stable-diffusion 3.5 medium, Stable-diffusion 3.5 Large Turbo.

```
python task1.py
```
## TASK B

Explore effects of using different schedulers, strength of CFG number of denoising steps and positive & negative prompting. To change the model from Stable-diffusion 3.5 medium to Stable-diffusion XL change the model_id in line 7. Note that the different schedulers experiment is only available for the latter because the diffusion pipeline for SD 3.5 medium does not allow any other scheduler. Run the following line to perform the experiments:


```
python task2.py
```

## TASK C-D

The generated captions are stored in food_images_new_2-2.txt. To generate the new images with Stable-diffusion 3.5 medium, run the following script:

```
python task4.py
```
## TASK E

Train / fine-tune your captioning model from last week. To do so, use the file train.py from this week and use the pipeline from week4.

### Finetune Llama 3.2.
 To change the Lora configuration, change the function add_peft in week4/task2/llm.py. Make sure to input your token for huggingface to be able to access the models.

To run inference on the trained llama models, use the script week4/task2/inferllama.py

## CLEANED DATA

In this project we cleaned the dataset in other to reduce noisy data. You can download it from this link:

https://drive.google.com/file/d/1z7B_C37B-HC8L95LO7s-p4Spv4QgFy1K/view?usp=sharing 

To clean these images, we used this scripts:

```
week4/imageCleaning/preprocess.py
week4/imageCleaning/remove_images.py
``` 