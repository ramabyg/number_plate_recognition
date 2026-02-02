# Number Plate Recognition - Reorganized Code Structure

This document explains the new project organization with YAML configuration files and TensorFlow events management.

## Project Structure

```
number_plate_recognition/
├── config/
│   ├── retinanet_config.yaml       # RetinaNet model and training parameters
│   └── paths.yaml                  # Data paths and output directories
├── tf_events/                      # TensorFlow event files (tracked by git)
├── output_pjt3_retinanet/          # Model outputs (large files ignored)
│   ├── coco_eval/                  # Evaluation metrics
│   └── *.pth                        # Model checkpoints (ignored in git)
├── number_plate_detection_refactored.py  # New refactored script
├── Project3_object_detection.ipynb # Jupyter notebook
└── .gitignore                      # Git ignore rules
```

## Configuration Files

### 1. `config/retinanet_config.yaml`

This file stores all RetinaNet training and model parameters that detectron2 can directly read:

- **MODEL**: RetinaNet-specific settings (classes, loss functions, anchor generators)
- **INPUT**: Image resolution settings
- **SOLVER**: Training hyperparameters (learning rate, batch size, iterations)
- **DATASETS**: Train/test dataset names
- **OUTPUT_DIR**: Output directory for model checkpoints

**To modify training parameters**, edit this file instead of changing Python code.

### 2. `config/paths.yaml`

This file stores all path configurations:

- **DATA**: Paths to training and test datasets
- **OUTPUT**: Output directories for models, events, and evaluations
- **CACHE**: Cache file locations for dataset dictionaries
- **MODEL**: Model checkpoint filenames and thresholds

**To change data paths or output locations**, edit this file.

## Usage

### Running the Refactored Script

```bash
python number_plate_detection_refactored.py
```

The script will:
1. Load configuration from YAML files
2. Load or prepare dataset dictionaries
3. Register datasets with detectron2
4. Setup detectron2 configuration
5. Train the RetinaNet model
6. Run inference on test data
7. Evaluate using COCO metrics

### Modifying Configuration

**For training parameters:**
Edit `config/retinanet_config.yaml` and change values under:
- `SOLVER.IMS_PER_BATCH` - Images per batch
- `SOLVER.BASE_LR` - Learning rate
- `SOLVER.MAX_ITER` - Maximum iterations

**For dataset paths:**
Edit `config/paths.yaml` and update:
- `DATA.TRAIN_ROOT` - Training data path
- `DATA.TEST_ROOT` - Test data path

## TensorFlow Events

TensorFlow event files are automatically saved during training by detectron2. These files are now:

- **Tracked in git** (in the `tf_events/` directory) for future analysis
- **Viewable with TensorBoard**: 
  ```bash
  tensorboard --logdir output_pjt3_retinanet
  ```

## Code Improvements

### Refactored Design

The new script (`number_plate_detection_refactored.py`) includes:

1. **Modular Functions**:
   - `load_config()` - Load YAML configuration
   - `prepare_pjt3_dataset()` - Prepare dataset
   - `load_or_prepare_dataset()` - Load from cache or prepare
   - `setup_config()` - Configure detectron2
   - `register_datasets()` - Register datasets
   - `train_model()` - Train the model
   - `run_inference()` - Run inference
   - `evaluate_model()` - Evaluate using COCO metrics

2. **Better Organization**:
   - Configuration in YAML files (not hardcoded)
   - Clear separation of concerns
   - Reusable functions
   - Improved logging and status messages

3. **Detectron2-Native Configuration**:
   - Uses `cfg.merge_from_file()` to load YAML directly
   - No manual PyYAML parsing needed for model config
   - Leverages detectron2's built-in configuration system

## Git Integration

### What's Tracked
- Configuration files (`.yaml`)
- Source code (`.py`)
- TensorFlow events (`tf_events/`)
- Jupyter notebooks (`.ipynb`)
- This README

### What's Ignored
- Large model checkpoints (`*.pth`)
- Dataset cache files (`*.pkl`)
- Model evaluation results (kept locally)
- Python cache files

