import sys, os, distutils.core
import torch, detectron2
from torchmetrics.detection import MeanAveragePrecision

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.structures import BoxMode
import albumentations as A
from detectron2.utils.visualizer import ColorMode
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer


#define data augmentation transforms
transform = A.Compose([
    # A.CenterCrop(width=500,height=250),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(translate_percent=0.05, scale=(0.8, 1.2), rotate=(-20, 20),shear=(-15,15), p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_id'],filter_invalid_bboxes=True))
#we define the format of the BboxParams as pascal_voc because we have them in the format XYXY_ABS in the custom dataset we have created


def custom_mapper(dataset_dict):
    """Apply Albumentations transforms on the fly"""
    dataset_dict = dataset_dict.copy()  # Avoid modifying the original dataset
    image = utils.read_image(dataset_dict["file_name"], format="BGR")  # Read image
    bboxes=[]
    category_id=[]
    for i in dataset_dict["annotations"]:
        bboxes.append(i['bbox'])
        category_id.append(i['category_id'])

    # Apply Albumentations augmentations
    transformed = transform(image=image,bboxes=bboxes,category_id=category_id)
    
    dataset_dict["image"] = torch.as_tensor(transformed["image"].copy().transpose(2, 0, 1))  # Convert to tensor format
    
    annos=[]
    for i in range(len(transformed["bboxes"])):
        annos.append({'bbox':torch.as_tensor(transformed["bboxes"][i]),
                      'category_id':torch.as_tensor(transformed["category_id"][i],dtype=torch.int),
                      'bbox_mode':BoxMode.XYXY_ABS})
    
    instances = utils.annotations_to_instances(annos, transformed["image"].shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict

def get_KITTI_dicts(img_dir,part):
    json_file = os.path.join(img_dir,part+"_data_kitti_mots_coco.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for _, v in enumerate(imgs_anns):
        record = {}
        
        filename = os.path.join(img_dir,"image_02", v["image"])
          
        record["file_name"] = filename
        record["image_id"] = v["image_id"]
        record["height"] = v["height"]
        record["width"] = v["width"]

        annos = v["objects"]

        objs = []
        for i in range(len(annos["id"])):
            if annos["category"][i]==10: #ignore
                continue
            elif  annos["category"][i]==1: #cars
                cat=0
            else: #person
                cat=1

            obj = {
                "bbox": [annos["bbox"][i][0],annos["bbox"][i][1],annos["bbox"][i][0]+annos["bbox"][i][2],annos["bbox"][i][1]+annos["bbox"][i][3]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": cat,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

if __name__ == '__main__':
    
    for d in ["train","val"]:
        DatasetCatalog.register("KITTI_" + d, lambda d=d: get_KITTI_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/",d))
        MetadataCatalog.get("KITTI_" + d).set(thing_classes=["car","person"])
    KITTI_metadata = MetadataCatalog.get("KITTI_train")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    ## predict a few images on the pretrained model to compare with finetuned results##

    dataset_dicts = get_KITTI_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/","val")
    paths=[]
    for d in random.sample(dataset_dicts, 3):
        paths.append(d)
        
        print(d["file_name"])
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join("./output/",d["file_name"].split('/')[-2])+'_'+d["file_name"].split('/')[-1],np.array(out.get_image()[:, :, ::-1]))



    ## TRAINING ##

    class AugmentedTrainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, cfg):
            return build_detection_train_loader(cfg, mapper=custom_mapper)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("KITTI_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    trainer = AugmentedTrainer(cfg)  # Use our custom trainer
    trainer.resume_or_load(resume=False)
    trainer.train()


    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    predictor = DefaultPredictor(cfg)


    #run inference on same images as for pretrained model at the beggining #

    dataset_dicts = get_KITTI_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/","val")
    for d in paths:
        
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  
        v = Visualizer(im[:, :, ::-1],
                    metadata=KITTI_metadata,
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(os.path.join("./output/",'FINETUNED_flip_'+d["file_name"].split('/')[-2]+'_'+d["file_name"].split('/')[-1]))
        cv2.imwrite(os.path.join("./output/",'FINETUNED_flip_'+d["file_name"].split('/')[-2]+'_'+d["file_name"].split('/')[-1]),np.array(out.get_image()[:, :, ::-1]))


    evaluator = COCOEvaluator("KITTI_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "KITTI_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

