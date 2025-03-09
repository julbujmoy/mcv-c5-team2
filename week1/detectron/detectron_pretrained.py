import sys, os, distutils.core
import torch, detectron2
from torchmetrics.detection import MeanAveragePrecision

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode

#/gpfs/home/lsalort/mcv-c5-team2/week1/detectron/data_tracking_image_2/training/


def get_KITTI_dicts(img_dir,part):
    json_file = os.path.join(img_dir,part+"_data_kitti_mots_coco.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    # for idx,v in enumerate(imgs_anns):
    #   print(v)

    dataset_dicts = []
    for _, v in enumerate(imgs_anns):
        record = {}
        

        filename = os.path.join(img_dir,"image_02", v["image"])
          
        # print(filename)
        # height, width = cv2.imread(filename).shape[:2]

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
                cat=2
            else: #person
                cat=0

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
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    # outputs = predictor(im)


    ## INFERENCE OFF-THE-SHELF ##

    # evaluator = COCOEvaluator("KITTI_val", output_dir="./output")
    # val_loader = build_detection_test_loader(cfg, "KITTI_val",)
    # print(inference_on_dataset(predictor.model, val_loader, evaluator))


    dataset_dicts = get_KITTI_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/","val")
    for d in random.sample(dataset_dicts, 3):        
        print(d["file_name"])
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # print(os.path.join("./output/",d["file_name"].split('/')[-2]+d["file_name"].split('/')[-1]))
        cv2.imwrite(os.path.join("./output/",d["file_name"].split('/')[-2])+'_'+d["file_name"].split('/')[-1],np.array(out.get_image()[:, :, ::-1]))

    if os.path.exists("predo.txt"):
        print('reading predictions')
        # Read the JSON file
        with open("pred.txt", "r") as f:
            loaded_data = json.load(f)

        # Convert lists back to PyTorch tensors
        all_pred_boxes = []
        for item in loaded_data:
            restored_item = {
                'boxes': torch.tensor(item['boxes']),  # Convert list back to tensor
                'labels': torch.tensor(item['labels'],dtype=torch.int),
                'scores': torch.tensor(item['scores'])  # Convert list back to tensor
            }
            all_pred_boxes.append(restored_item)
        
        # Read the JSON file
        with open("gt.txt", "r") as f:
            loaded_data = json.load(f)

        # Convert lists back to PyTorch tensors
        all_gt_boxes = []
        for item in loaded_data:
            restored_item = {
                'boxes': torch.tensor(item['boxes']),  # Convert list back to tensor
                'labels': torch.tensor(item['labels'],dtype=torch.int)  # Convert list back to tensor
            }
            all_gt_boxes.append(restored_item)

    else: 
        all_pred_boxes, all_gt_boxes = [], []
        for d in dataset_dicts:
            box=[]
            scores=[]
            labels=[]
            for i in range(len(d['annotations'])):
                box.append(d['annotations'][i]["bbox"])
                labels.append(d['annotations'][i]["category_id"])
            all_gt_boxes.append(
                    {"boxes": torch.Tensor(box), "labels": torch.Tensor(labels,dtpye=torch.int)})
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            all_pred_boxes.append({"boxes": torch.Tensor(outputs['instances'].pred_boxes.tensor.tolist()), "labels": torch.Tensor(outputs['instances'].pred_classes.tolist(),dtype=torch.int), "scores": torch.Tensor(outputs['instances'].scores.tolist())})
            
            
        converted_data = []
        for item in all_pred_boxes:
            converted_item = {
                'boxes': item['boxes'].tolist(),
                'labels': item['labels'].tolist(),
                'scores': item['scores'].tolist()
            }
            converted_data.append(converted_item)
        # Save to a file
        with open("pred.txt", "w") as f:
            json.dump(converted_data, f)


        converted_data = []
        for item in all_gt_boxes:
            converted_item = {
                'boxes': item['boxes'].tolist(),
                'labels': item['labels'].tolist()
            }
            converted_data.append(converted_item)
        # Save to a file
        with open("gt.txt", "w") as f:
            json.dump(converted_data, f)

     # Compute mean Average Precision (mAP)
    metric = MeanAveragePrecision(iou_type="bbox",box_format='xyxy',class_metrics=True)
    metric.update(all_pred_boxes, all_gt_boxes)
    video_metrics = metric.compute()
    print(video_metrics)