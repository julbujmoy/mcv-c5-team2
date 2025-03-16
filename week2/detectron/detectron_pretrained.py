import sys, os, distutils.core
import torch, detectron2
from torchmetrics.detection import MeanAveragePrecision
import PIL.Image as Image
from torchvision.ops import masks_to_boxes
from torch import tensor

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
from detectron2.structures import BoxMode


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
                    cat=2 #it is 2 to correspond with coco labels
                else: #person
                    cat=0
                
                box = masks_to_boxes(torch.Tensor(np.expand_dims(mask, axis=0)))
                obj = {
                    "bbox": np.array(box).squeeze(0),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "masks":[mask],
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
    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    ## INFERENCE OFF-THE-SHELF ##


    dataset_dicts = get_KITTI_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/","val")
    for d in random.sample(dataset_dicts, 3):        
        print(d["file_name"])
        im = cv2.imread(d["file_name"])
        outputs = predictor(im) 
        # print(outputs) 
        # Use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_path=os.path.join("./output/",d["file_name"].split('/')[-2])+'_'+d["file_name"].split('/')[-1]
        print(out_path)
        os.makedirs(os.path.join("./output/",d["file_name"].split('/')[-2][:-5]),exist_ok=True)
        cv2.imwrite(out_path,np.array(out.get_image()[:, :, ::-1]))

        # mask=outputs['instances'].pred_masks.long().to("cpu")
        # mask_path=os.path.join("./output/masks",d["file_name"].split('/')[-2][9:])+'_'+d["file_name"].split('/')[-1]
        
        # final_mask=np.zeros((np.shape(mask)[1],np.shape(mask)[2],))
        # for i in range(np.shape(mask)[0]):
        #     # print(np.shape(mask[i,:,:]))
        #     print(outputs["instances"].pred_classes.tolist()[i])
        #     final_mask=final_mask+np.array(mask[i,:,:])*outputs["instances"].pred_classes.tolist()[i]
            
        # final_mask=cv2.normalize(final_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # cv2.imwrite(mask_path,np.array(final_mask))

 


        # gt_path="C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/instances/"+d["file_name"].split('/')[-2][8:]+'/'+d["file_name"].split('/')[-1]
        # img = np.array(Image.open(gt_path))


        # print(obj_ids)
        # for i in range(len(obj_ids)):
        #     obj_id = obj_ids[i]
        #     class_id = obj_id // 1000
        #     obj_instance_id = obj_id % 1000
        #     print('Object nº '+str(i)+' of class '+str(class_id)+' and instance id '+str(obj_instance_id))



    all_pred_boxes, all_gt_boxes = [], []
    print(len(dataset_dicts))
    for d in dataset_dicts:
        box=[]
        scores=[]
        labels=[]
        masks=[]
        for i in range(len(d['annotations'])):
            box.append(d['annotations'][i]["bbox"])
            labels.append(d['annotations'][i]["category_id"])
            masks.append(d['annotations'][i]["masks"])
        
        # print(np.shape(np.array(masks).squeeze(1)))

        all_gt_boxes.append(
                {"boxes": torch.Tensor(np.array(box)), "labels": tensor(np.array(labels),dtype=torch.int),"masks":tensor(np.array(masks).squeeze(1),dtype=torch.bool) })
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  
        pred_mask=np.array(outputs['instances'].pred_masks.to("cpu"),dtype=np.uint8)

        # print(np.shape(pred_mask))
        # print(outputs['instances'].pred_masks.to("cpu"))
        all_pred_boxes.append({"boxes": torch.Tensor(outputs['instances'].pred_boxes.tensor.tolist()), 
                               "labels": tensor(outputs['instances'].pred_classes.tolist(),dtype=torch.int), 
                               "scores": torch.Tensor(outputs['instances'].scores.tolist()),
                               "masks": tensor(pred_mask,dtype=torch.bool)})
            
    
    # print(all_pred_boxes[-1])
    # print(all_gt_boxes[-1])
    # Compute mean Average Precision (mAP)
    metric = MeanAveragePrecision(iou_type='segm',box_format='xyxy',class_metrics=True)
    metric = MeanAveragePrecision(iou_type='bbox',box_format='xyxy',class_metrics=True)
    metric.update(all_pred_boxes, all_gt_boxes)
    video_metrics = metric.compute()
    print(video_metrics)