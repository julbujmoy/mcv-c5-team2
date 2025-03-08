"""
# Week 1 Project: Object Detection on KITTI-MOTS Dataset

## Overview

This project focuses on applying and evaluating three state-of-the-art object detection models on the KITTI-MOTS dataset. The goal is to perform inference using pre-trained models, evaluate them using COCO metrics, fine-tune the models, and compare their performance based on various metrics such as inference time, model parameters, and robustness.

We implement and evaluate the following models:
- **Faster R-CNN** using Detectron2.
- **DeTR (DEtection TRansformers)** using Hugging Face.
- **YOLOv(>8)** using Ultralytics YOLO.

The project includes both quantitative and qualitative results for each model, along with a comparison between pre-trained and fine-tuned versions.

### Key Features:
- **Faster R-CNN, DeTR, and YOLO** for object detection.
- **Evaluation using COCO metrics** (e.g., mAP, precision, recall).
- **Fine-tuning** of models for improved performance.
- **Comparison of models** based on inference time, robustness, and model parameters.
  
---

## Task Breakdown

### Task (c): Run Inference with Pre-trained Models
- **Goal**: Apply Faster R-CNN, DeTR, and YOLO with pre-trained weights from COCO on the KITTI-MOTS dataset.
- **Frameworks**:
  - Faster R-CNN using Detectron2.
  - DeTR using Hugging Face's Transformers library.
  - YOLO using the Ultralytics YOLOv(>8).
  
- **Steps**:
  1. Load pre-trained models.
  2. Run inference on KITTI-MOTS dataset.
  3. Visualize and save detection results (images with bounding boxes).

---

### Task (d): Evaluate Pre-trained Models
- **Goal**: Evaluate Faster R-CNN, DeTR, and YOLOv(>8) using official COCO metrics.
  
- **Steps**:
  1. Map class labels from KITTI-MOTS to COCO.
  2. Run evaluation using COCO metrics.
  3. Report quantitative results (e.g., mAP, precision, recall).

---

### Task (e): Fine-Tune Models on KITTI-MOTS Dataset
- **Goal**: Fine-tune Faster R-CNN, DeTR, and YOLO on the KITTI-MOTS dataset using augmentations.
  
- **Frameworks**:
  - Faster R-CNN & DeTR: Use Albumentations for data augmentation.
  - YOLO: Use internal augmentations in the YOLO configuration.

- **Steps**:
  1. Split training and validation sets.
  2. Fine-tune models on the training set.
  3. Evaluate fine-tuned models on the validation set.
  4. Compare performance with pre-trained models.
  5. Report both quantitative and qualitative results.

---

### Task (f): Fine-Tune Models on a Different Dataset (Domain Shift)
- **Goal**: Fine-tune Faster R-CNN and DeTR models on a new dataset to test domain shift.
  
- **Steps**:
  1. Choose a new dataset.
  2. Apply augmentations (Albumentations for Faster R-CNN and DeTR, YOLO configuration for YOLO).
  3. Fine-tune models.
  4. Evaluate fine-tuned models using COCO metrics.
  5. Compare results with pre-trained models.

---

### Task (g): Model Comparison and Analysis
- **Goal**: Analyze the differences among Faster R-CNN, DeTR, and YOLOv(>8) in terms of:
  - Inference time.
  - Number of parameters.
  - Robustness.

- **Steps**:
  1. Measure inference time for each model.
  2. Compare the number of model parameters.
  3. Analyze robustness in handling different types of objects and occlusions.
  4. Present results quantitatively and qualitatively.

---

## Installation and Setup

### Prerequisites:
- Python 3.7 or higher.
- PyTorch 1.8+.
- Detectron2 (for Faster R-CNN).
- Hugging Face Transformers (for DeTR).
- Ultralytics YOLOv(>8).
- Albumentations (for data augmentation).

### Setup Instructions:
1. Install the necessary dependencies:
   ```bash
   pip install torch torchvision detectron2 transformers ultralytics albumentations
