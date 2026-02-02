# Number plate detection project 3 opencv with deep leanring
# uses detectron2 RetinaNet model for object detection as baseline model
# 5 parts:
# 1. Plot ground truth bounding boxes
# 2. Retrain RetinaNet with custom dataset
# 3. Inference
# 4. Evaluate model performance COCO detection metrics
# 5. Run inference on video


import matplotlib
matplotlib.use('TkAgg') # or 'Qt5Agg', 'GTK3Agg', 'WXAgg'
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt

# model_zoo has lots of pre-trained models
from detectron2 import model_zoo

# DefaultTrainer is a class for training object detector
from detectron2.engine import DefaultTrainer
# DefaultPredictor is class for inference
from detectron2.engine import DefaultPredictor

# detectron2 has its configuration format
from detectron2.config import get_cfg
# detectron2 has implemented Visualizer of object detection
from detectron2.utils.visualizer import Visualizer

# from DatasetCatalog, detectron2 gets dataset and from MetadatCatalog it gets metadata of the dataset
from detectron2.data import DatasetCatalog, MetadataCatalog

# BoxMode supports bounding boxes in different format
from detectron2.structures import BoxMode

# COCOEvaluator based on COCO evaluation metric, inference_on_dataset is used for evaluation for a given metric
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# build_detection_test_loader, used to create test loader for evaluation
from detectron2.data import build_detection_test_loader

# function to prepare dataset in detectron2 format
def prepare_pjt3_dataset(dataroot):
    if not os.path.exists(dataroot):
        print(f"Data root path {dataroot} does not exist.")
        return []
    dataroot = os.path.join(dataroot, 'Vehicle registration plate')
    # Data root contains images with name as ID, a folder "Lables" cotains annotation of number plates in txt files.
    # file name format: <image_id>.txt
    # Each annotation file contain multiple lines, each line contains:
    # <Vehicle> <registration> <plate> <x_min> <y_min> <x_max> <y_max>
    
    data_set_dicts = []
    bberror = False  # Flag to indicate bounding box error
    # Loop through all image files in the dataroot
    for img_filename in os.listdir(dataroot):
        if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
  
        record = {}
        image_id = os.path.splitext(img_filename)[0]
        img_path = os.path.join(dataroot, img_filename)
        height, width = cv2.imread(img_path).shape[:2]
        
        record["file_name"] = img_path
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width
        
        annos = []
        label_file_path = os.path.join(dataroot, "Label", f"{image_id}.txt")
        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 7:
                        x_min, y_min, x_max, y_max = map(float, parts[3:7])
                        obj = {
                            "bbox": [x_min, y_min, x_max, y_max],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": 0,  # single class 'number_plate'
                            "iscrowd": 0
                        }
                        annos.append(obj)
                        # do basic validation of bbox
                        if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
                            print(f"Warning: Bounding box out of image bounds in file {label_file_path}")
                            bberror = True
        
        
        record["annotations"] = annos
        if bberror:
            print(f"Skipping image {img_path} due to bounding box errors.")
            bberror = False  # Reset the flag for next image
            continue
        data_set_dicts.append(record)
    return data_set_dicts

def main():
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train_data_root = "/mnt/rama_ml/data/num_plt_reco_opencv_dl_pytch_wk10_pjt/train"
    test_data_root = "/mnt/rama_ml/data/num_plt_reco_opencv_dl_pytch_wk10_pjt/validation"
    
    if not os.path.exists('train_data_set_dict.pkl') or not os.path.exists('test_data_set_dict.pkl'):
        print("Preparing dataset dicts...")
        train_data_set_dict = prepare_pjt3_dataset(train_data_root) #TBD
        test_data_set_dict = prepare_pjt3_dataset(test_data_root)
    
        # save the dataset dicts for future use
        import pickle
        with open('train_data_set_dict.pkl', 'wb') as f:
            pickle.dump(train_data_set_dict, f)
        with open('test_data_set_dict.pkl', 'wb') as f:
            pickle.dump(test_data_set_dict, f)
    else:
        print("Loading dataset dicts from pickle files...")
        import pickle
        with open('train_data_set_dict.pkl', 'rb') as f:
            train_data_set_dict = pickle.load(f)
        with open('test_data_set_dict.pkl', 'rb') as f:
            test_data_set_dict = pickle.load(f)
    
    #clear cuda cache
    import torch
    torch.cuda.empty_cache()
    
    #register dataset to detectron2
    DatasetCatalog.register("pjt3_train", lambda : train_data_set_dict)
    MetadataCatalog.get("pjt3_train").set(thing_classes=["number_plate"])
    
    DatasetCatalog.register("pjt3_test", lambda : test_data_set_dict)
    MetadataCatalog.get("pjt3_test").set(thing_classes=["number_plate"])
    
    # Part 1: Plot ground truth bounding boxes for validation dataset
    pjt3_metadata = MetadataCatalog.get("pjt3_test")
    dataset_dicts = DatasetCatalog.get("pjt3_test")
    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=pjt3_metadata, scale=0.5)
    #     out = visualizer.draw_dataset_dict(d)
    #     plt.figure(figsize=(15,10))
    #     plt.imshow(out.get_image()[:, :, ::-1])
    #     plt.show()
        
    # pause = 1  # Debug pause
    # input("Press Enter to continue...")
    # cv2.waitKey(1)
    
    #Part 2 : Retrain RetinaNet with custom dataset
    
    #update the default configuration for RetinaNet model
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ("pjt3_train")
    # cfg.DATASETS.TEST = ("pjt3_test")
    # cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.RETINANET.NUM_CLASSES = 1  # only one class 'number_plate'
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # use pre-trained weights
    
    # cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 1000   # 1000 iterations TBD update it later
    # cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"
    # cfg.MODEL.RETINANET.SMOOTH_L1_BETA = 0.11
    # cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 1.5
    # cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
    # cfg.OUTPUT_DIR = "./output_pjt3_retinanet"
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # from detectron2.config import get_cfg
    # from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    ))

    # -----------------------------
    # DATASETS
    # -----------------------------
    cfg.DATASETS.TRAIN = ("pjt3_train",)
    cfg.DATASETS.TEST = ("pjt3_test",)
    cfg.DATALOADER.NUM_WORKERS = 2

    # -----------------------------
    # MODEL
    # -----------------------------
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.RETINANET.NUM_CLASSES = 1   # number_plate

    # --- CRITICAL FOR NUMBER PLATES ---
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]

    # -----------------------------
    # INPUT RESOLUTION (IMPORTANT)
    # -----------------------------
    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024

    # -----------------------------
    # SOLVER
    # -----------------------------
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    # cfg.SOLVER.AMP.ENABLED = True


    # A reasonable schedule for 5k–10k images
    cfg.SOLVER.MAX_ITER = 25000
    cfg.SOLVER.STEPS = (10000, 13000, 18000, 22000)
    cfg.SOLVER.GAMMA = 0.1

    # -----------------------------
    # RETINANET LOSS SETTINGS
    # -----------------------------
    cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"
    cfg.MODEL.RETINANET.SMOOTH_L1_BETA = 0.1
    cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 1.5
    cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25

    # -----------------------------
    # OUTPUT
    # -----------------------------
    cfg.OUTPUT_DIR = "./output_pjt3_retinanet"


    
    #setup tensorboard logger
    # tensorboard --logdir output_pjt3_retinanet
    
    #create trainer and start training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True) #TBD
    trainer.train() #TBD
    
    #Part 3 : Inference
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   #
    cfg.DATASETS.TEST = ("pjt3_test",)
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get("pjt3_test")
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=pjt3_metadata,
                       scale=0.5
        )
        #draw the prediction with higher confidence score
        out = v.draw_instance_predictions(outputs["instances"].to("cpu")[0])
        plt.figure(figsize=(15,10))
        plt.imshow(out.get_image())
        plt.show()
    
    pause = 1  # Debug pause
    # input("Press Enter to continue...")   
    # cv2.waitKey(1)
    print("Inference completed")
    
    
    # Part 4: Model evaluation using COCO metrics
    eval_dir = os.path.join(cfg.OUTPUT_DIR, 'coco_eval')
    evaluator = COCOEvaluator("pjt3_test", cfg, False, output_dir=eval_dir)
    val_loader = build_detection_test_loader(cfg, 'pjt3_test')
    print("Starting model evaluation...")
    inference_on_dataset(predictor.model, val_loader, evaluator)
    print("Model evaluation completed.")
    
    
    
    

if __name__ == "__main__":
    print("Number plate detection with detectron2")
    print("Detectron2 version:", detectron2.__version__)
    main()
    