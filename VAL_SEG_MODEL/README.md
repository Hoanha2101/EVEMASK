# EVEMASK v1.0.0

<div align="center">

![EVEMASK Logo](https://img.shields.io/badge/EVEMASK-%201.0.0-blue?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/Python-3.9%2B-green?style=for-the-badge&logo=python)
![TensorRT](https://img.shields.io/badge/TensorRT-8.6.1-orange?style=for-the-badge&logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green?style=for-the-badge&logo=opencv)

**Real-Time NBA Tournaments Betting Logo Segmentation and Masking Module**
</div>

# VAL_SEG_MODEL - Validation Segmentation Model

## Description
This project performs evaluation and performance comparison of different segmentation models on a NBA dataset.

## Directory Structure

```
VAL_SEG_MODEL/
├── DATASET/                 # Training and validation dataset
├── WEIGHTS/                 # Model weight files
│   ├── Mask_RCNN/          # Weights for Mask R-CNN
│   ├── yolo11_seg/         # Weights for YOLO11-seg
│   └── yolov8_seg/         # Weights for YOLOv8-seg
├── CSV_REPORT/             # CSV result reports
├── PLOT_REPORT/            # Charts and report images
├── JSON_DATA/              # JSON data
├── MASK_RCNN_RUN/          # Mask R-CNN run results
├── YOLO_RUN/               # YOLO run results
└── TRAIN_LOG_MASK_RCNN/    # Mask R-CNN training logs
```

## Models Evaluated

### 1. Mask R-CNN
- **Backbone**: ResNet50, ResNet101
- **Augmentation**: With and without data augmentation

### 2. YOLO11-seg
- **Sizes**: Small (s), Medium (m), Large (l)
- **Augmentation**: With and without data augmentation

### 3. YOLOv8-seg
- **Sizes**: Small (s), Medium (m), Large (l)
- **Augmentation**: With and without data augmentation

## Usage

1. **Prepare dataset**: Place images and labels in `DATASET/` directory
2. **Run validation**: Use `VAL_base.ipynb` notebook
3. **View results**: Check report directories

## Results

The project generates:
- Performance reports (AP50, AP75)
- Runtime comparisons
- Radar charts and heatmaps
- Per-class analysis

## Requirements

- Python 3.8+
- PyTorch
- Detectron2 (for Mask R-CNN)
- Ultralytics (for YOLO)
- OpenCV
- NumPy, Pandas, Matplotlib