import sys, os, distutils.core
import torch, detectron2

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


def get_balloon_dicts(img_dir,part):
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
                cat=0
            else: #person
                cat=1

            obj = {
                "bbox": annos["bbox"][i],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cat,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

if __name__ == '__main__':
    
    for d in ["train","val"]:
        DatasetCatalog.register("KITTI_" + d, lambda d=d: get_balloon_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/",d))
        MetadataCatalog.get("KITTI_" + d).set(thing_classes=["car","person"])
    balloon_metadata = MetadataCatalog.get("KITTI_train")

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    # outputs = predictor(im)


    ## INFERENCE OFF-THE-SHELF ##

    evaluator = COCOEvaluator("KITTI_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "KITTI_val")
    # print(inference_on_dataset(predictor.model, val_loader, evaluator))


    dataset_dicts = get_balloon_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/","val")
    paths=[]
    for d in random.sample(dataset_dicts, 3):
        paths.append(d)
        
        print(d["file_name"])
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow('v',out.get_image()[:, :, ::-1])        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # print(os.path.join("./output/",d["file_name"].split('/')[-2]+d["file_name"].split('/')[-1]))
        cv2.imwrite(os.path.join("./output/",d["file_name"].split('/')[-2])+'_'+d["file_name"].split('/')[-1],np.array(out.get_image()[:, :, ::-1]))


    ## TRAINING ##

    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("KITTI_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = get_balloon_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/","val")
    for d in paths:
        
        
        # print(d["file_name"])
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1],
                    metadata=balloon_metadata,
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(os.path.join("./output/",'FINETUNED_'+d["file_name"].split('/')[-2]+'_'+d["file_name"].split('/')[-1]))
        cv2.imwrite(os.path.join("./output/",'FINETUNED_'+d["file_name"].split('/')[-2]+'_'+d["file_name"].split('/')[-1]),np.array(out.get_image()[:, :, ::-1]))

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    evaluator = COCOEvaluator("KITTI_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "KITTI_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`

