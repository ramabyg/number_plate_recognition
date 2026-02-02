# Number plate detection project 3 opencv with deep learning
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
import yaml
import pickle
import torch

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


def load_config():
    """Load configuration from YAML files."""
    with open('config/paths.yaml', 'r') as f:
        paths_config = yaml.safe_load(f)
    return paths_config


def prepare_pjt3_dataset(dataroot):
    """
    Prepare dataset in detectron2 format.
    
    Args:
        dataroot: Root path to the dataset
        
    Returns:
        List of dataset dictionaries in detectron2 format
    """
    if not os.path.exists(dataroot):
        print(f"Data root path {dataroot} does not exist.")
        return []
    
    dataroot = os.path.join(dataroot, 'Vehicle registration plate')
    # Data root contains images with name as ID, a folder "Labels" contains annotation of number plates in txt files.
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


def load_or_prepare_dataset(config):
    """Load cached dataset or prepare new one."""
    cache_dir = config['CACHE']
    train_pkl = cache_dir['TRAIN_DICT_FILE']
    test_pkl = cache_dir['TEST_DICT_FILE']
    train_data_root = config['DATA']['TRAIN_ROOT']
    test_data_root = config['DATA']['TEST_ROOT']
    
    if not os.path.exists(train_pkl) or not os.path.exists(test_pkl):
        print("Preparing dataset dicts...")
        train_data_set_dict = prepare_pjt3_dataset(train_data_root)
        test_data_set_dict = prepare_pjt3_dataset(test_data_root)
        
        # Save the dataset dicts for future use
        with open(train_pkl, 'wb') as f:
            pickle.dump(train_data_set_dict, f)
        with open(test_pkl, 'wb') as f:
            pickle.dump(test_data_set_dict, f)
        print(f"Dataset dicts saved to {train_pkl} and {test_pkl}")
    else:
        print(f"Loading dataset dicts from cache...")
        with open(train_pkl, 'rb') as f:
            train_data_set_dict = pickle.load(f)
        with open(test_pkl, 'rb') as f:
            test_data_set_dict = pickle.load(f)
    
    return train_data_set_dict, test_data_set_dict


def setup_config(config):
    """
    Setup detectron2 configuration.
    
    Args:
        config: Configuration dictionary from YAML
        
    Returns:
        detectron2 cfg object
    """
    # Load base RetinaNet config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    
    # Merge custom config from YAML
    cfg.merge_from_file("config/retinanet_config.yaml")
    
    # Set model weights from pretrained checkpoint
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    
    # Update output directory
    cfg.OUTPUT_DIR = config['OUTPUT']['MODEL_DIR']
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print(f"Configuration loaded from: config/retinanet_config.yaml")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    
    return cfg


def register_datasets(train_data_set_dict, test_data_set_dict):
    """Register datasets with detectron2."""
    DatasetCatalog.register("pjt3_train", lambda: train_data_set_dict)
    MetadataCatalog.get("pjt3_train").set(thing_classes=["number_plate"])
    
    DatasetCatalog.register("pjt3_test", lambda: test_data_set_dict)
    MetadataCatalog.get("pjt3_test").set(thing_classes=["number_plate"])
    
    return MetadataCatalog.get("pjt3_test")


def train_model(cfg, tf_events_dir):
    """
    Train the RetinaNet model.
    
    Args:
        cfg: detectron2 configuration object
        tf_events_dir: Directory to store TensorFlow events
    """
    print("\n" + "="*60)
    print("PART 2: Training RetinaNet on custom dataset")
    print("="*60)
    print(f"TensorFlow events being saved to: {tf_events_dir}")
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
    
    print("\nTraining completed!")
    print(f"Model checkpoints saved to: {cfg.OUTPUT_DIR}")


def run_inference(cfg, config, pjt3_metadata):
    """
    Run inference on test dataset.
    
    Args:
        cfg: detectron2 configuration object
        config: Configuration dictionary from YAML
        pjt3_metadata: Metadata for visualization
    """
    print("\n" + "="*60)
    print("PART 3: Running inference on test dataset")
    print("="*60)
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, config['MODEL']['FINAL_WEIGHTS'])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['MODEL']['SCORE_THRESH_TEST']
    cfg.DATASETS.TEST = ("pjt3_test",)
    
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get("pjt3_test")
    
    for d in random.sample(dataset_dicts, min(3, len(dataset_dicts))):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=pjt3_metadata,
                       scale=0.5
        )
        # Draw the prediction with higher confidence score
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(15, 10))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()
    
    print("Inference completed!")
    return predictor


def evaluate_model(cfg, config, predictor):
    """
    Evaluate model using COCO metrics.
    
    Args:
        cfg: detectron2 configuration object
        config: Configuration dictionary from YAML
        predictor: DefaultPredictor object
    """
    print("\n" + "="*60)
    print("PART 4: Evaluating model using COCO metrics")
    print("="*60)
    
    eval_dir = config['OUTPUT']['COCO_EVAL_DIR']
    os.makedirs(eval_dir, exist_ok=True)
    
    evaluator = COCOEvaluator("pjt3_test", cfg, False, output_dir=eval_dir)
    val_loader = build_detection_test_loader(cfg, 'pjt3_test')
    
    print("Starting model evaluation...")
    inference_on_dataset(predictor.model, val_loader, evaluator)
    print("Model evaluation completed.")
    print(f"Evaluation results saved to: {eval_dir}")


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Load configuration
    config = load_config()
    tf_events_dir = config['OUTPUT']['TF_EVENTS_DIR']
    
    # Create necessary directories
    os.makedirs(tf_events_dir, exist_ok=True)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # ===== Load or Prepare Dataset =====
    train_data_set_dict, test_data_set_dict = load_or_prepare_dataset(config)
    
    # ===== Register Datasets =====
    pjt3_metadata = register_datasets(train_data_set_dict, test_data_set_dict)
    
    # ===== Setup Configuration =====
    cfg = setup_config(config)
    
    # ===== PART 2: Train Model =====
    train_model(cfg, tf_events_dir)
    
    # ===== PART 3: Run Inference =====
    predictor = run_inference(cfg, config, pjt3_metadata)
    
    # ===== PART 4: Evaluate Model =====
    evaluate_model(cfg, config, predictor)
    
    print("\n" + "="*60)
    print("All processes completed successfully!")
    print("="*60)


if __name__ == "__main__":
    print("Number plate detection with detectron2")
    print("Detectron2 version:", detectron2.__version__)
    main()
