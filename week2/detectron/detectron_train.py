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
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
import PIL.Image as Image
from torchvision.ops import masks_to_boxes
from torch import tensor
import pycocotools

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

        img=np.array(Image.open(os.path.join(img_dir[:-9],'instances',v["image"])))
        # print(np.max(img))
        obj_ids = np.unique(img)

        
        objs = []   
        for i in range(len(obj_ids)):
            if obj_ids[i]!=0:
                mask=img==obj_ids[i]
                mask=np.array(mask.astype(int),dtype=np.uint8)

            # # to correctly interpret the id of a single object
                obj_id = obj_ids[i]
                class_id = obj_id // 1000
                # obj_instance_id = obj_id % 1000
                # print(class_id)
                if class_id==10: #ignore
                    continue
                elif  class_id==1: #cars
                    cat=0 
                else: #person
                    cat=1
                
                box = masks_to_boxes(torch.Tensor(np.expand_dims(mask, axis=0)))
                obj = {
                    "bbox": np.array(box).squeeze(0),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation":pycocotools.mask.encode(np.asarray(mask, order="F")),
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
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg.INPUT.MASK_FORMAT="bitmask"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    ## predict a few images on the pretrained model to compare with finetuned results##

    dataset_dicts = get_KITTI_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/","val")
    paths=[]
    for d in random.sample(dataset_dicts, 3):
        paths.append(d)
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        out_path=os.path.join("./output/",d["file_name"].split('/')[-2])+'_'+d["file_name"].split('/')[-1]
        print(out_path)
        os.makedirs(os.path.join("./output/",d["file_name"].split('/')[-2][:-5]),exist_ok=True)
        cv2.imwrite(out_path,np.array(out.get_image()[:, :, ::-1]))

    ## TRAINING ##


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("KITTI_train")
    cfg.INPUT.MASK_FORMAT="bitmask"
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 4  
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = 10000  
    cfg.SOLVER.STEPS = [6000,8000] 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Two classes, car and person
    
    # cfg.MODEL.BACKBONE.FREEZE_AT = 2
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256]]
    # cfg.MODEL.RPN.NMS_THRESH = 0.7
    # cfg.SOLVER.IMS_PER_BATCH = 4  
    # cfg.SOLVER.BASE_LR = 3e-4 
    # cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    # cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    # cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    # cfg.SOLVER.AMP.ENABLED = True
    # cfg.SOLVER.IMS_PER_BATCH = 4  


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_KITTI_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/","val")
    
    #run inference on same images as for pretrained model at the beggining #
    for d in paths:
        
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  
        v = Visualizer(im[:, :, ::-1],
                    metadata=KITTI_metadata,
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW 
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        new_out_path=os.path.join("./output/",'FINETUNED_'+d["file_name"].split('/')[-2]+'_'+d["file_name"].split('/')[-1])
        print(new_out_path)
        os.makedirs(os.path.join("./output/",'FINETUNED_'+d["file_name"].split('/')[-2][:-5]),exist_ok=True)
        cv2.imwrite(os.path.join("./output/",'FINETUNED_'+d["file_name"].split('/')[-2]+'_'+d["file_name"].split('/')[-1]),np.array(out.get_image()[:, :, ::-1]))


    evaluator = COCOEvaluator("KITTI_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "KITTI_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

