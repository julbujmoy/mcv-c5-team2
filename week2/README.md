Link to the slides: https://docs.google.com/presentation/d/1w6qQSF4deRC42Zx6FWwJyF1HAxUWxjHguNAuZCukiP0/edit?usp=sharing
# Object Segmentation - Week 2

## Project: Object Segmentation

### Introduction
This project is part of the C5 course and focuses on object segmentation using Detectron2, Huggingface, and Ultralytics in PyTorch. The goal is to experiment with different object segmentation models and evaluate their performance.

### Tools Used
- **Detectron2** (Facebook AI Research)
- **Huggingface Transformers**
- **Ultralytics YOLO**

### Tasks
1. **Inference and Evaluation:** Perform inference with pre-trained models such as Mask R-CNN, Mask2Former, and YOLO-SEG on the KITTI-MOTS dataset.
2. **Fine-tuning on KITTI-MOTS:** Fine-tune Mask R-CNN, Mask2Former, and YOLO-SEG with data augmentation techniques.
3. **Fine-tuning on Another Dataset:** Evaluate model performance in a different domain.
4. **Model Comparison:** Analyze inference time, parameter count, and robustness of each model.
5. **(Optional)** Experimentation with the **Segment Anything Model (SAM).**

### Dataset
**KITTI-MOTS**
- Segmentation and tracking of cars and pedestrians.
- **Training:** 12 sequences (~8,073 pedestrian masks, ~18,831 car masks).
- **Validation:** 9 sequences (~3,347 pedestrian masks, ~8,068 car masks).
- **Testing:** 29 sequences.

### Expected Results
- **Quantitative Metrics:** Evaluation using COCO-standard metrics.
- **Qualitative Results:** Visualization of segmentations and model comparisons.

### Deliverables
- Code in a **GitHub** repository with instructions in this README.
- Presentation with analysis and results.
- A summary slide with key conclusions.

### Usage Instructions

### Resources
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Huggingface Transformers](https://huggingface.co/docs/transformers/tasks/object_detection)
- [Ultralytics YOLO](https://docs.ultralytics.com/es/tasks/detect/)
- [KITTI-MOTS Dataset](https://www.vision.rwth-aachen.de/page/mots)





