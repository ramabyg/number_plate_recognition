# Number plate detection project 3 opencv with deep learning
# uses detectron2 RetinaNet model for object detection as baseline model
# 5 parts:
# 1. Plot ground truth bounding boxes
# 2. Retrain RetinaNet with custom dataset
# 3. Inference
# 4. Evaluate model performance COCO detection metrics
# 5. Run inference on video

from logging import config
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
    with open('configs/paths.yaml', 'r') as f:
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


def setup_config(training_config, project_config):
    """
    Setup detectron2 configuration.
    
    Args:
        training_config: Path to training config file
        project_config: Project configuration dictionary from YAML
        
    Returns:
        detectron2 cfg object
    """
    
    if not os.path.exists(training_config):
        raise FileNotFoundError(f"Configuration file {training_config} not found.")
    
    # Load base config from detectron2 model zoo
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Merge custom config from YAML
    cfg.merge_from_file(training_config)
    
    
    # Set model weights from pretrained checkpoint
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    
    # Update output directory
    with open(training_config, 'r') as f:
        training_configconfig = yaml.safe_load(f)
    cfg.OUTPUT_DIR = os.path.join(project_config['OUTPUT']['MODEL_DIR'], training_configconfig['OUTPUT_DIR'])
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    
    print(f"Configuration loaded from: {training_config}")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    
    return cfg, training_configconfig['OUTPUT_DIR']


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

# ===== Extra: Video read and write function =====
def video_read_write(cfg, config, pjt3_metadata,video_path):
    """
    Read video frames one-by-one, flip it, and write in the other video.
    video_path (str): path/to/video
    """
    video = cv2.VideoCapture(video_path)
    
    # Check if camera opened successfully
    if not video.isOpened(): 
        print("Error opening video file")
        return
    
    # create video writer
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_fname = '{}_out.mp4'.format(os.path.splitext(video_path)[0])
    
    output_file = cv2.VideoWriter(
        filename=output_fname,
        # some installation of opencv may not support x264 (due to its license),
        # you can try other format (e.g. MPEG)
        fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )
    
    #Inference logic
    
    print("\n" + "="*60)
    print("PART 4: Running inference on video file")
    print("="*60)
    #unfreeze cfg to modify for inference
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, config['MODEL']['FINAL_WEIGHTS_FILE'])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['MODEL']['SCORE_THRESH_TEST']
    # cfg.DATASETS.TEST = ("pjt3_test",)
    cfg.freeze()
    
    predictor = DefaultPredictor(cfg)
    # dataset_dicts = DatasetCatalog.get("pjt3_test")
    
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            outputs = predictor(frame)
            v = Visualizer(frame[:, :, ::-1],
                           metadata=pjt3_metadata,
                           scale=1.0
            )
            # Draw the prediction with higher confidence score
            #check if there are any instances detected
            if len(outputs["instances"]) == 0:
                print(f"No instances detected in frame {i}")
                output_file.write(frame)
#                 cv2.imwrite('anpd_out/frame_{}.png'.format(str(i).zfill(3)), frame[:, ::-1, :])
                i += 1
                continue
            print(f"Instances detected in frame {i}: {len(outputs['instances'])}")
            out = v.draw_instance_predictions(outputs["instances"].to("cpu")[0])
            frame = out.get_image()[:, :, ::-1]           
            output_file.write(frame)
#             cv2.imwrite('anpd_out/frame_{}.png'.format(str(i).zfill(3)), frame[:, ::-1, :])
            i += 1
        else:
            break
        
    video.release()
    output_file.release()
    
    return

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
    #unfreeze cfg to modify for inference
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, config['MODEL']['FINAL_WEIGHTS_FILE'])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['MODEL']['SCORE_THRESH_TEST']
    cfg.DATASETS.TEST = ("pjt3_test",)
    cfg.freeze()
    
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
        #check if there are any instances detected
        if len(outputs["instances"]) == 0:
            print(f"No instances detected in image {d['file_name']}")
            continue
        out = v.draw_instance_predictions(outputs["instances"].to("cpu")[0])
        plt.figure(figsize=(15, 10))
        plt.imshow(out.get_image())
        plt.show()
        cv2.waitKey(0)
        # pause = input("Press Enter to continue to next image...")
        
        
        plt.axis('off')
        plt.tight_layout()
        output_path = f"prediction_{d['file_name'].split('/')[-1]}"
        output_dir = os.path.join(config['OUTPUT']['INFERENCE_DIR'])
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_path)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved prediction image to {output_path}")
    
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
    
    #get the training configuration from command line argument
    # if not, use the default config file
    import argparse
    parser = argparse.ArgumentParser(description="Number Plate Detection with Detectron2")
    parser.add_argument("-c", "--config", type=str, default="config/retinanet_config.yaml",
                        help="Path to the configuration YAML file")
    args = parser.parse_args()
    training_config = args.config
    
    # Load configuration for fixed paths
    project_config = load_config()
    # tf_events_dir = config['OUTPUT']['TF_EVENTS_DIR']
    
    # Create necessary directories
    # os.makedirs(tf_events_dir, exist_ok=True)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # ===== Load or Prepare Dataset =====
    train_data_set_dict, test_data_set_dict = load_or_prepare_dataset(project_config)
    
    # ===== Register Datasets =====
    pjt3_metadata = register_datasets(train_data_set_dict, test_data_set_dict)
    
    # ===== Setup Configuration =====
    cfg, output_dir = setup_config(training_config, project_config)
    
    # ===== PART 2: Train Model =====
    tf_events_dir = os.path.join(output_dir, project_config['OUTPUT']['TF_EVENTS_DIR'])
    os.makedirs(tf_events_dir, exist_ok=True)
    # train_model(cfg, tf_events_dir)
    
    # ===== PART 3: Run Inference =====
    # predictor = run_inference(cfg, project_config, pjt3_metadata)
    
    # ===== PART 4: Evaluate Model =====
    # evaluate_model(cfg, project_config, predictor)
    
    # ===== Extra: Run Inference on Video =====
    video_path = project_config['DATA']['TEST_VIDEO_PATH']
    if os.path.exists(video_path):
        video_read_write(cfg, project_config, pjt3_metadata, video_path)
    else:
        print(f"Test video path {video_path} does not exist. Skipping video inference.")
    
    print("\n" + "="*60)
    print("All processes completed successfully!")
    print("="*60)


if __name__ == "__main__":
    print("Number plate detection with detectron2")
    print("Detectron2 version:", detectron2.__version__)
    main()
