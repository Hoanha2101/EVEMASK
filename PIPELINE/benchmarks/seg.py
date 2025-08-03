"""
Segmentation Model Benchmark Evaluation System.
This system provides comprehensive benchmarking capabilities for comparing PyTorch and TensorRT 
segmentation models in terms of accuracy (mAP) and inference speed.

The benchmark evaluates:
- Mean Average Precision (mAP) at IoU thresholds of 0.5 and 0.75
- Per-class Average Precision (AP) scores
- Inference time comparison between PyTorch and TensorRT models
- Detailed performance analysis and visualization

Author: EVEMASK Team
Version: 1.0.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools.utils import *
from src.models.initNet import net1
import torch
import cv2
import numpy as np
from torchvision.ops import nms
import glob
import time
import matplotlib.pyplot as plt
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
print("Loading model...")
model_pytorch = torch.load("weights/pytorch/seg_v1.0.0.pth", weights_only=False)
model_pytorch.eval()
model_pytorch = model_pytorch.to(device)
print("Model loaded successfully!")

# Model Configuration
iou_thres = 0.5      # IoU threshold for Non-Maximum Suppression (NMS)
conf_thres = 0.5     # Confidence threshold for detection filtering
number_of_class = 14 # Number of classes in the dataset

# Class names - modify according to your dataset
CLASS_NAMES = [
    "unbet",
    "betrivers",
    "fanduel",
    "betway",
    "caesars",
    "bally",
    "draftkings",
    "pointsbet",
    "bet365",
    "fanatics",
    "betparx",
    "betmgm",
    "gilariver",
    "casino"
]

def preprocessing(original_frame, half=True):
    """
    Preprocess input image for segmentation model inference.
    
    Args:
        original_frame (numpy.ndarray): Input image in BGR format
        half (bool): Whether to use half precision (float16) for processing
        
    Returns:
        tuple: (original_frame, img_tensor, ratio, dwdh)
            - original_frame: Original input image
            - img_tensor: Preprocessed tensor ready for model input
            - ratio: Scaling ratio for coordinate transformation
            - dwdh: Padding offsets for coordinate transformation
    """
    input_image, ratio, dwdh = letterbox(original_frame, (640, 640))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image, img_no_255, seg_img = blob(input_image, return_seg=True, half=half)
    img_tensor = np.ascontiguousarray(input_image)
    return original_frame, img_tensor, ratio, dwdh

def xywh2xyxy(x):
    """
    Convert bounding boxes from center format (x_center, y_center, width, height) 
    to corner format (x1, y1, x2, y2).
    
    Args:
        x (numpy.ndarray): Bounding boxes in center format [N, 4]
        
    Returns:
        numpy.ndarray: Bounding boxes in corner format [N, 4]
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1 = x_center - width/2
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1 = y_center - height/2
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2 = x_center + width/2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2 = y_center + height/2
    return y

def scale_boxes(img1_shape, boxes, img0_shape, ratio, dwdh):
    """
    Scale bounding boxes from model input size back to original image size.
    
    Args:
        img1_shape (tuple): Shape of the model input image (height, width)
        boxes (numpy.ndarray): Bounding boxes in model input coordinates [N, 4]
        img0_shape (tuple): Shape of the original image (height, width)
        ratio (float): Scaling ratio used during preprocessing
        dwdh (tuple): Padding offsets (width_pad, height_pad) used during preprocessing
        
    Returns:
        numpy.ndarray: Bounding boxes scaled to original image coordinates [N, 4]
    """
    boxes[:, [0, 2]] -= dwdh[0]  # Remove width padding
    boxes[:, [1, 3]] -= dwdh[1]  # Remove height padding
    boxes[:, :4] /= ratio        # Scale back to original size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img0_shape[1])  # Clip to image width
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img0_shape[0])  # Clip to image height
    return boxes

def scale_masks(masks, ratio, dwdh, original_shape):
    """
    Scale segmentation masks from model output size back to original image size.
    
    Args:
        masks (numpy.ndarray): Segmentation masks from model output [N, H, W]
        ratio (float): Scaling ratio used during preprocessing
        dwdh (tuple): Padding offsets (width_pad, height_pad) used during preprocessing
        original_shape (tuple): Shape of the original image (height, width)
        
    Returns:
        numpy.ndarray: Segmentation masks scaled to original image size [N, H_orig, W_orig]
    """
    top, left = int(dwdh[1]), int(dwdh[0])  # Top and left padding offsets
    bottom, right = int(640 - dwdh[1]), int(640 - dwdh[0])  # Bottom and right crop boundaries
    masks = masks[:, top:bottom, left:right]  # Remove padding from masks
    
    masks_resized = []
    for mask in masks:
        # Resize each mask to original image dimensions
        mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        masks_resized.append(mask_resized)
    
    return np.array(masks_resized)

def postprocessing(tensor_pred0, tensor_pred1_2, original_shape, ratio, dwdh):
    """
    Post-process model predictions to extract bounding boxes, scores, classes, and masks.
    
    Args:
        tensor_pred0 (torch.Tensor): Model output containing detection predictions
        tensor_pred1_2 (torch.Tensor): Model output containing prototype masks
        original_shape (tuple): Shape of the original input image (height, width)
        ratio (float): Scaling ratio used during preprocessing
        dwdh (tuple): Padding offsets used during preprocessing
        
    Returns:
        tuple: (boxes, scores, classes, masks) or (None, None, None, None) if no detections
            - boxes: Bounding boxes in original image coordinates [N, 4]
            - scores: Confidence scores for each detection [N]
            - classes: Class indices for each detection [N]
            - masks: Segmentation masks for each detection [N, H, W]
    """
    pred = tensor_pred0[0].transpose(0, 1).cpu().numpy()
    protos = tensor_pred1_2.cpu().numpy()
    
    # Extract predictions
    boxes = pred[:, 0:4]  # Bounding box coordinates
    scores = pred[:, 4:number_of_class+4]  # Class confidence scores
    masks_coef = pred[:, number_of_class+4:]  # Mask coefficients
    
    # Get maximum scores and corresponding classes
    max_scores = np.max(scores, axis=1)
    classes = np.argmax(scores, axis=1)
    
    # Filter by confidence threshold
    mask_keep = max_scores > conf_thres
    if not np.any(mask_keep):
        return None, None, None, None
    
    boxes = boxes[mask_keep]
    max_scores = max_scores[mask_keep]
    classes = classes[mask_keep]
    masks_coef = masks_coef[mask_keep]
    
    # Convert to corner format and apply NMS
    boxes = xywh2xyxy(boxes)
    keep_indices = nms(torch.from_numpy(boxes).float(), torch.from_numpy(max_scores).float(), iou_thres)
    keep_indices = keep_indices.cpu().numpy()
    
    if len(keep_indices) == 0:
        return None, None, None, None
    
    boxes = boxes[keep_indices]
    max_scores = max_scores[keep_indices]  
    classes = classes[keep_indices]
    masks_coef = masks_coef[keep_indices]
    
    # Generate masks from prototypes and coefficients
    protos_reshaped = protos[0].reshape(32, -1)
    masks = masks_coef @ protos_reshaped
    masks = masks.reshape(-1, 160, 160)
    masks = 1 / (1 + np.exp(-masks))  # Apply sigmoid activation
    
    # Scale masks to model input size and then to original size
    masks_640 = np.array([cv2.resize(mask, (640, 640), interpolation=cv2.INTER_LINEAR) for mask in masks])
    masks_original = scale_masks(masks_640, ratio, dwdh, original_shape)
    boxes = scale_boxes((640, 640), boxes, original_shape, ratio, dwdh)
    
    # Final filtering and refinement
    final_masks = []
    final_boxes = []
    final_scores = []
    final_classes = []
    
    for mask, box, score, cls in zip(masks_original, boxes, max_scores, classes):
        binary_mask = mask > 0.5
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_shape[1], x2), min(original_shape[0], y2)
        
        # Refine mask to bounding box region
        refined_mask = np.zeros_like(binary_mask)
        if x2 > x1 and y2 > y1:
            refined_mask[y1:y2, x1:x2] = binary_mask[y1:y2, x1:x2]
        
        # Filter out masks that are too large (likely false positives)
        if np.sum(refined_mask) / refined_mask.size <= 0.8:
            final_masks.append(refined_mask)
            final_boxes.append(box)
            final_scores.append(score)
            final_classes.append(cls)
    
    if len(final_masks) == 0:
        return None, None, None, None
    
    return np.array(final_boxes), np.array(final_scores), np.array(final_classes), np.array(final_masks)

# Pre-allocated test tensors for timing measurements
input_tensor_mask_gpu = torch.randn(1, 3, 640, 640).to(device)  # PyTorch tensor for GPU timing
input_tensor_mask_half = np.ascontiguousarray(np.random.randn(1, 3, 640, 640).astype(np.float16))  # TensorRT tensor for timing

def seg_pytorch(image, time_infer=False):
    """
    Perform segmentation inference using PyTorch model.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        time_infer (bool): If True, only measure inference time without processing
        
    Returns:
        tuple: (boxes, scores, classes, masks) or (None, None, None, None) if no detections
    """
    if time_infer:
        # Time inference only using pre-allocated tensor
        with torch.no_grad():
            preds = model_pytorch(input_tensor_mask_gpu)
        return None, None, None, None
    else:
        # Full inference pipeline
        original_frame, img_tensor, ratio, dwdh = preprocessing(image, half=False)
        input_tensor = torch.from_numpy(img_tensor).float().to(device)
        
        with torch.no_grad():
            preds = model_pytorch(input_tensor)
        
        return postprocessing(preds[0], preds[1][2], original_frame.shape, ratio, dwdh)

def seg_trt_half(image, time_infer=False):
    """
    Perform segmentation inference using TensorRT model with half precision.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        time_infer (bool): If True, only measure inference time without processing
        
    Returns:
        tuple: (boxes, scores, classes, masks) or (None, None, None, None) if no detections
    """
    if time_infer:
        # Time inference only using pre-allocated tensor
        net1.cuda_ctx.push()
        results = net1.infer(input_tensor_mask_half)
        net1.cuda_ctx.pop()
        return None, None, None, None
    else:   
        # Full inference pipeline
        original_frame, img_tensor, ratio, dwdh = preprocessing(image, half=True)
        net1.cuda_ctx.push()
        results = net1.infer(img_tensor)
        net1.cuda_ctx.pop()
    
        # Convert TensorRT outputs to PyTorch tensors for postprocessing
        tensor_pred0 = copy_trt_output_to_torch_tensor(results[0])
        tensor_pred1_2 = copy_trt_output_to_torch_tensor(results[1])
        
        return postprocessing(tensor_pred0, tensor_pred1_2, original_frame.shape, ratio, dwdh)

def load_gt_annotations_efficient(val_images_dir, val_labels_dir):
    """
    Load ground truth annotations efficiently by scanning image directory.
    
    Args:
        val_images_dir (str): Path to validation images directory
        val_labels_dir (str): Path to validation labels directory
        
    Returns:
        list: List of dictionaries containing image and label file information
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(val_images_dir, ext)))
    
    image_files = sorted(image_files)
    gt_info = []
    
    for idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(val_labels_dir, txt_name)
        
        gt_info.append({
            'image_id': idx + 1,
            'image_path': img_path,
            'label_path': txt_path
        })
    
    return gt_info

def load_single_gt_annotations(label_path, img_shape):
    annotations = []
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                    
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
                
                if len(coords) % 2 != 0:
                    coords = coords[:-1]
                
                if len(coords) < 6:
                    continue
                
                points = np.array(coords).reshape(-1, 2)
                points[:, 0] *= img_shape[1]
                points[:, 1] *= img_shape[0]
                
                try:
                    polygon = points.astype(np.int32)
                    area = cv2.contourArea(polygon)
                    if area > 0:
                        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
                        cv2.fillPoly(mask, [polygon], 1)
                        annotations.append({
                            'category_id': cls,
                            'mask': mask,
                            'area': area
                        })
                except:
                    continue
    
    return annotations

def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1 (numpy.ndarray): First binary mask
        mask2 (numpy.ndarray): Second binary mask
        
    Returns:
        float: IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def evaluate_ap_per_class_streaming(gt_info_list, inference_func, iou_threshold=0.5):
    class_detections = defaultdict(list)
    class_gt_counts = defaultdict(int)
    
    print(f"Calculating per-class AP@{int(iou_threshold*100)}...")
    
    for idx, gt_info in enumerate(gt_info_list):
        img = cv2.imread(gt_info['image_path'])
        if img is None:
            continue
        
        gt_annotations = load_single_gt_annotations(gt_info['label_path'], img.shape)
        
        for gt in gt_annotations:
            class_gt_counts[gt['category_id']] += 1
        
        boxes, scores, classes, masks = inference_func(img)
        
        if masks is not None and len(masks) > 0:
            for mask, score, cls in zip(masks, scores, classes):
                class_detections[cls].append({
                    'image_id': gt_info['image_id'],
                    'mask': mask,
                    'score': score,
                    'gt_path': gt_info['label_path'],
                    'img_shape': img.shape
                })
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(gt_info_list)} images")
    
    class_aps = {}
    
    for class_id in range(number_of_class):
        if class_id not in class_detections or class_gt_counts[class_id] == 0:
            class_aps[class_id] = 0.0
            continue
        
        detections = sorted(class_detections[class_id], key=lambda x: x['score'], reverse=True)
        
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        used_gts = set()
        
        for det_idx, detection in enumerate(detections):
            gt_annotations = load_single_gt_annotations(detection['gt_path'], detection['img_shape'])
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_annotations):
                gt_key = f"{detection['image_id']}_{gt_idx}_{class_id}"
                
                if (gt_key not in used_gts and gt['category_id'] == class_id):
                    iou = calculate_iou(detection['mask'] > 0.5, gt['mask'])
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                tp[det_idx] = 1
                gt_key = f"{detection['image_id']}_{best_gt_idx}_{class_id}"
                used_gts.add(gt_key)
            else:
                fp[det_idx] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / class_gt_counts[class_id]
        
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            p_interp = precision[recall >= t]
            if len(p_interp) > 0:
                ap += np.max(p_interp) / 11
        
        class_aps[class_id] = ap
    
    mean_ap = np.mean(list(class_aps.values())) if class_aps else 0.0
    return class_aps, mean_ap

def create_per_class_bar_chart(results):
    """Tạo biểu đồ cột so sánh AP per class"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    class_ids = list(range(number_of_class))
    class_names_short = [name[:8] + '...' if len(name) > 8 else name for name in CLASS_NAMES]
    
    pytorch_ap50 = [results['PyTorch']['class_ap50'].get(i, 0) for i in class_ids]
    tensorrt_ap50 = [results['TensorRT']['class_ap50'].get(i, 0) for i in class_ids]
    pytorch_ap75 = [results['PyTorch']['class_ap75'].get(i, 0) for i in class_ids]
    tensorrt_ap75 = [results['TensorRT']['class_ap75'].get(i, 0) for i in class_ids]
    
    x = np.arange(len(class_ids))
    width = 0.35
    
    # AP@50 Bar Chart
    bars1 = ax1.bar(x - width/2, pytorch_ap50, width, label='PyTorch', alpha=0.8, color='#2E86C1', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, tensorrt_ap50, width, label='TensorRT', alpha=0.8, color='#E74C3C', edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AP@50 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class AP@50 Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names_short, rotation=45, ha='right')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(max(pytorch_ap50), max(tensorrt_ap50)) * 1.1)
    
    # Add value labels for top 3 performers
    top3_indices = sorted(range(len(pytorch_ap50)), key=lambda i: max(pytorch_ap50[i], tensorrt_ap50[i]), reverse=True)[:3]
    for i in top3_indices:
        if pytorch_ap50[i] > 0.01:
            ax1.text(bars1[i].get_x() + bars1[i].get_width()/2., bars1[i].get_height() + 0.01,
                    f'{pytorch_ap50[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        if tensorrt_ap50[i] > 0.01:
            ax1.text(bars2[i].get_x() + bars2[i].get_width()/2., bars2[i].get_height() + 0.01,
                    f'{tensorrt_ap50[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # AP@75 Bar Chart
    bars3 = ax2.bar(x - width/2, pytorch_ap75, width, label='PyTorch', alpha=0.8, color='#2E86C1', edgecolor='black', linewidth=0.5)
    bars4 = ax2.bar(x + width/2, tensorrt_ap75, width, label='TensorRT', alpha=0.8, color='#E74C3C', edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AP@75 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class AP@75 Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names_short, rotation=45, ha='right')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(max(pytorch_ap75), max(tensorrt_ap75)) * 1.1)
    
    # Add value labels for top 3 performers
    top3_indices_75 = sorted(range(len(pytorch_ap75)), key=lambda i: max(pytorch_ap75[i], tensorrt_ap75[i]), reverse=True)[:3]
    for i in top3_indices_75:
        if pytorch_ap75[i] > 0.01:
            ax2.text(bars3[i].get_x() + bars3[i].get_width()/2., bars3[i].get_height() + 0.01,
                    f'{pytorch_ap75[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        if tensorrt_ap75[i] > 0.01:
            ax2.text(bars4[i].get_x() + bars4[i].get_width()/2., bars4[i].get_height() + 0.01,
                    f'{tensorrt_ap75[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('benchmarks/results/seg_per_class_bar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_per_class_line_chart(results):
    """Tạo biểu đồ line so sánh AP per class"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    class_ids = list(range(number_of_class))
    class_names_short = [name[:8] + '...' if len(name) > 8 else name for name in CLASS_NAMES]
    
    pytorch_ap50 = [results['PyTorch']['class_ap50'].get(i, 0) for i in class_ids]
    tensorrt_ap50 = [results['TensorRT']['class_ap50'].get(i, 0) for i in class_ids]
    pytorch_ap75 = [results['PyTorch']['class_ap75'].get(i, 0) for i in class_ids]
    tensorrt_ap75 = [results['TensorRT']['class_ap75'].get(i, 0) for i in class_ids]
    
    x = np.arange(len(class_ids))
    
    # AP@50 Line Chart
    line1 = ax1.plot(x, pytorch_ap50, 'o-', linewidth=3, markersize=8, label='PyTorch', color='#2E86C1', markerfacecolor='white', markeredgewidth=2)
    line2 = ax1.plot(x, tensorrt_ap50, 's-', linewidth=3, markersize=8, label='TensorRT', color='#E74C3C', markerfacecolor='white', markeredgewidth=2)
    
    # Fill area between lines
    ax1.fill_between(x, pytorch_ap50, tensorrt_ap50, alpha=0.2, color='gray')
    
    ax1.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AP@50 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class AP@50 Trend Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names_short, rotation=45, ha='right')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max(pytorch_ap50), max(tensorrt_ap50)) * 1.1)
    
    # Add annotations for significant differences
    for i in range(len(class_ids)):
        diff = abs(tensorrt_ap50[i] - pytorch_ap50[i])
        if diff > 0.05:  # Only annotate if difference > 5%
            mid_y = (pytorch_ap50[i] + tensorrt_ap50[i]) / 2
            ax1.annotate(f'delta{diff:.4f}', xy=(i, mid_y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # AP@75 Line Chart
    line3 = ax2.plot(x, pytorch_ap75, 'o-', linewidth=3, markersize=8, label='PyTorch', color='#2E86C1', markerfacecolor='white', markeredgewidth=2)
    line4 = ax2.plot(x, tensorrt_ap75, 's-', linewidth=3, markersize=8, label='TensorRT', color='#E74C3C', markerfacecolor='white', markeredgewidth=2)
    
    # Fill area between lines
    ax2.fill_between(x, pytorch_ap75, tensorrt_ap75, alpha=0.2, color='gray')
    
    ax2.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AP@75 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class AP@75 Trend Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names_short, rotation=45, ha='right')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(max(pytorch_ap75), max(tensorrt_ap75)) * 1.1)
    
    # Add annotations for significant differences
    for i in range(len(class_ids)):
        diff = abs(tensorrt_ap75[i] - pytorch_ap75[i])
        if diff > 0.05:  # Only annotate if difference > 5%
            mid_y = (pytorch_ap75[i] + tensorrt_ap75[i]) / 2
            ax2.annotate(f'delta{diff:.4f}', xy=(i, mid_y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('benchmarks/results/seg_per_class_line_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_map_comparison_chart(results):
    """Tạo biểu đồ riêng cho mAP comparison"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = ['PyTorch', 'TensorRT']
    mean_ap50 = [results[model]['mean_ap50'] for model in models]
    mean_ap75 = [results[model]['mean_ap75'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    # mAP bars
    bars1 = ax.bar(x - width/2, mean_ap50, width, label='mAP@50', alpha=0.8, color='#28B463', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, mean_ap75, width, label='mAP@75', alpha=0.8, color='#F39C12', edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars1, mean_ap50):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}\n({val*100:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    for bar, val in zip(bars2, mean_ap75):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}\n({val*100:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Mean Average Precision (mAP)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison - mAP Scores', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(mean_ap50), max(mean_ap75)) * 1.15)
    
    # Add difference annotation
    ap50_diff = abs(mean_ap50[1] - mean_ap50[0])
    ap75_diff = abs(mean_ap75[1] - mean_ap75[0])
    
    ax.text(0.5, max(max(mean_ap50), max(mean_ap75)) * 1.05, 
            f'AP@50 delta: {ap50_diff:.4f} • AP@75 delta: {ap75_diff:.4f}', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('benchmarks/results/seg_map_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_inference_time_comparison_chart(results):
    """Tạo biểu đồ riêng cho inference time comparison"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = ['PyTorch', 'TensorRT']
    inference_times = [results[model]['avg_inference_time'] * 1000 for model in models]  # Convert to ms
    
    x = np.arange(len(models))
    width = 0.6
    
    # Inference time bars với màu gradient
    colors = ['#E74C3C', '#2E86C1']  # Red for PyTorch, Blue for TensorRT
    bars = ax.bar(x, inference_times, width, alpha=0.8, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, inference_times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(inference_times) * 0.02,
                f'{val:.1f} ms', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Inference Time (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_title('Model Speed Comparison - Inference Time', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(inference_times) * 1.2)
    
    # Add speedup annotation
    speedup = inference_times[0] / inference_times[1]
    ax.text(0.5, max(inference_times) * 0.85, 
            f'TensorRT is {speedup:.1f}x faster\n({inference_times[0] - inference_times[1]:.1f}ms saved per inference)', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.8))
    
    # Add FPS information
    fps_pytorch = 1000 / inference_times[0]
    fps_tensorrt = 1000 / inference_times[1]
    
    ax.text(0, inference_times[0] * 0.5, f'{fps_pytorch:.1f} FPS', 
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.text(1, inference_times[1] * 0.5, f'{fps_tensorrt:.1f} FPS', 
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('benchmarks/results/seg_inference_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_combined_summary_table(results):
    """Tạo bảng tóm tắt kết quả"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    models = ['PyTorch', 'TensorRT']
    table_data = []
    
    for model in models:
        inference_time_ms = results[model]['avg_inference_time'] * 1000
        fps = 1000 / inference_time_ms
        
        table_data.append([
            model,
            f"{results[model]['mean_ap50']:.4f} ({results[model]['mean_ap50']*100:.1f}%)",
            f"{results[model]['mean_ap75']:.4f} ({results[model]['mean_ap75']*100:.1f}%)",
            f"{inference_time_ms:.1f} ms",
            f"{fps:.1f} FPS"
        ])
    
    # Add comparison row
    speedup = (results['PyTorch']['avg_inference_time'] / results['TensorRT']['avg_inference_time'])
    ap50_diff = abs(results['TensorRT']['mean_ap50'] - results['PyTorch']['mean_ap50'])
    ap75_diff = abs(results['TensorRT']['mean_ap75'] - results['PyTorch']['mean_ap75'])
    time_saved = (results['PyTorch']['avg_inference_time'] - results['TensorRT']['avg_inference_time']) * 1000
    
    table_data.append([
        "Difference",
        f"±{ap50_diff:.4f}",
        f"±{ap75_diff:.4f}",
        f"{speedup:.1f}x faster",
        f"+{time_saved:.1f}ms saved"
    ])
    
    headers = ['Model', 'mAP@50', 'mAP@75', 'Inference Time', 'Performance']
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i == len(table_data):  # Last row (difference)
                table[(i, j)].set_facecolor('#FFE082')
                table[(i, j)].set_text_props(weight='bold')
            elif i == 2:  # TensorRT row
                table[(i, j)].set_facecolor('#E4F2FD')
            else:  # PyTorch row
                table[(i, j)].set_facecolor('#FFEBEE')
    
    plt.title('Model Performance Summary', fontsize=18, fontweight='bold', pad=30)
    plt.savefig('benchmarks/results/seg_summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_per_class_detailed_table(results):
    """Tạo bảng chi tiết AP per class"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = []
    headers = ['Class', 'Class Name', 'PyTorch AP@50', 'TensorRT AP@50', 'PyTorch AP@75', 'TensorRT AP@75', 'delta AP@50', 'delta AP@75']
    
    for i in range(number_of_class):
        pytorch_ap50 = results['PyTorch']['class_ap50'].get(i, 0)
        tensorrt_ap50 = results['TensorRT']['class_ap50'].get(i, 0)
        pytorch_ap75 = results['PyTorch']['class_ap75'].get(i, 0)
        tensorrt_ap75 = results['TensorRT']['class_ap75'].get(i, 0)
        
        diff_ap50 = tensorrt_ap50 - pytorch_ap50
        diff_ap75 = tensorrt_ap75 - pytorch_ap75
        
        table_data.append([
            str(i),
            CLASS_NAMES[i],
            f"{pytorch_ap50:.4f}",
            f"{tensorrt_ap50:.4f}",
            f"{pytorch_ap75:.4f}",
            f"{tensorrt_ap75:.4f}",
            f"{diff_ap50:+.4f}",
            f"{diff_ap75:+.4f}"
        ])
    
    # Add average row
    table_data.append([
        "Avg",
        "All Classes",
        f"{results['PyTorch']['mean_ap50']:.4f}",
        f"{results['TensorRT']['mean_ap50']:.4f}",
        f"{results['PyTorch']['mean_ap75']:.4f}",
        f"{results['TensorRT']['mean_ap75']:.4f}",
        f"{results['TensorRT']['mean_ap50'] - results['PyTorch']['mean_ap50']:+.4f}",
        f"{results['TensorRT']['mean_ap75'] - results['PyTorch']['mean_ap75']:+.4f}"
    ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i == len(table_data):  # Average row
                table[(i, j)].set_facecolor('#FFD54F')
                table[(i, j)].set_text_props(weight='bold')
            else:
                # Alternate row colors
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
                
                # Color code difference columns
                if j >= 6:  # Difference columns
                    try:
                        val = float(table_data[i-1][j].replace('+', ''))
                        if val > 0:
                            table[(i, j)].set_facecolor('#C8E6C9')  # Green for positive
                        elif val < 0:
                            table[(i, j)].set_facecolor('#FFCDD2')  # Red for negative
                    except:
                        pass
    
    plt.title('Per-Class AP Detailed Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('benchmarks/results/seg_per_class_detailed_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def ensure_results_directory():
    """
    Create results directory if it doesn't exist.
    
    This function ensures that the benchmarks/results directory exists
    for saving visualization outputs and benchmark results.
    """
    results_dir = 'benchmarks/results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

def benchmark_models():
    """
    Main benchmark function that evaluates PyTorch and TensorRT models.
    
    This function:
    1. Loads validation dataset
    2. Measures inference times for both models
    3. Calculates mAP@50 and mAP@75 for both models
    4. Generates comprehensive performance reports
    
    Returns:
        dict: Dictionary containing benchmark results for both models
    """
    val_images_dir = "DATASET/images/val"
    val_labels_dir = "DATASET/labels/val"
    
    # Ensure results directory exists
    ensure_results_directory()
    
    print("Loading validation set...")
    gt_info = load_gt_annotations_efficient(val_images_dir, val_labels_dir)
    total_images = len(gt_info)
    print(f"Found {total_images} images")
    
    if total_images == 0:
        print("ERROR: No images found! Please check your dataset paths.")
        return None
    
    results = {}
    models = {'PyTorch': seg_pytorch, 'TensorRT': seg_trt_half}
    
    for model_name, inference_func in models.items():
        print(f"\n{'='*50}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*50}")
        
        # Measure inference times (model inference only)
        print("Measuring inference times...")
        inference_times = []
        
        # Use smaller sample for timing to speed up
        timing_samples = min(100, total_images)
        
        for idx in range(timing_samples):
            img = cv2.imread(gt_info[idx]['image_path'])
            if img is None:
                continue
            
            # Warmup runs
            if idx < 5:
                _ = inference_func(img, time_infer=True)
                continue
            
            # Measure inference time
            inference_start = time.time()
            _ = inference_func(img, time_infer=True)
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            if (idx + 1) % 20 == 0:
                avg_time = np.mean(inference_times) if inference_times else 0
                print(f"  Sampled {idx + 1}/{timing_samples} images, avg time: {avg_time:.4f}s ({avg_time*1000:.1f}ms)")
        
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        print(f"Average model inference time: {avg_inference_time:.4f}s ({avg_inference_time*1000:.1f}ms)")
        
        # Calculate AP - can limit number of images for faster testing
        print("Calculating AP50...")
        class_ap50, mean_ap50 = evaluate_ap_per_class_streaming(gt_info, inference_func, 0.5)
        
        print("Calculating AP75...")
        class_ap75, mean_ap75 = evaluate_ap_per_class_streaming(gt_info, inference_func, 0.75)
        
        results[model_name] = {
            'class_ap50': class_ap50,
            'class_ap75': class_ap75,
            'mean_ap50': mean_ap50,
            'mean_ap75': mean_ap75,
            'avg_inference_time': avg_inference_time
        }
        
        print(f"\n{model_name} Results:")
        print(f"  mAP@50: {mean_ap50:.4f} ({mean_ap50*100:.1f}%)")
        print(f"  mAP@75: {mean_ap75:.4f} ({mean_ap75*100:.1f}%)")
        print(f"  Avg inference time: {avg_inference_time:.4f}s ({avg_inference_time*1000:.1f}ms)")
        print(f"  FPS: {1/avg_inference_time:.1f}")
    
    return results

def print_detailed_results(results):
    """
    Print detailed benchmark results to console.
    
    Args:
        results (dict): Dictionary containing benchmark results for PyTorch and TensorRT models
    """
    print(f"\n{'='*80}")
    print("DETAILED BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    # Overall comparison
    print("\nOVERALL PERFORMANCE:")
    print("-" * 50)
    for model_name in ['PyTorch', 'TensorRT']:
        result = results[model_name]
        inference_ms = result['avg_inference_time'] * 1000
        fps = 1 / result['avg_inference_time']
        
        print(f"{model_name:>10}: mAP@50={result['mean_ap50']:.4f} | mAP@75={result['mean_ap75']:.4f} | "
              f"Time={inference_ms:.1f}ms | FPS={fps:.1f}")
    
    # Speed comparison
    speedup = results['PyTorch']['avg_inference_time'] / results['TensorRT']['avg_inference_time']
    time_saved = (results['PyTorch']['avg_inference_time'] - results['TensorRT']['avg_inference_time']) * 1000
    print(f"\nSPEED IMPROVEMENT:")
    print(f"   TensorRT is {speedup:.1f}x faster ({time_saved:.1f}ms saved per inference)")
    
    # Accuracy comparison
    ap50_diff = results['TensorRT']['mean_ap50'] - results['PyTorch']['mean_ap50']
    ap75_diff = results['TensorRT']['mean_ap75'] - results['PyTorch']['mean_ap75']
    print(f"\nACCURACY DIFFERENCE:")
    print(f"   mAP@50: {ap50_diff:+.4f} ({ap50_diff*100:+.1f}%)")
    print(f"   mAP@75: {ap75_diff:+.4f} ({ap75_diff*100:+.1f}%)")
    
    # Per-class top/bottom performers
    print(f"\nPER-CLASS ANALYSIS (AP@50):")
    print("-" * 50)
    
    # Calculate differences per class
    class_diffs = {}
    for i in range(number_of_class):
        pytorch_ap = results['PyTorch']['class_ap50'].get(i, 0)
        tensorrt_ap = results['TensorRT']['class_ap50'].get(i, 0)
        class_diffs[i] = tensorrt_ap - pytorch_ap
    
    # Top 3 improvements
    top_improvements = sorted(class_diffs.items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top 3 Improvements (TensorRT vs PyTorch):")
    for class_id, diff in top_improvements:
        pytorch_ap = results['PyTorch']['class_ap50'].get(class_id, 0)
        tensorrt_ap = results['TensorRT']['class_ap50'].get(class_id, 0)
        print(f"   {CLASS_NAMES[class_id]:>12}: {pytorch_ap:.4f} → {tensorrt_ap:.4f} ({diff:+.4f})")
    
    # Top 3 degradations
    top_degradations = sorted(class_diffs.items(), key=lambda x: x[1])[:3]
    print("\nTop 3 Degradations (TensorRT vs PyTorch):")
    for class_id, diff in top_degradations:
        pytorch_ap = results['PyTorch']['class_ap50'].get(class_id, 0)
        tensorrt_ap = results['TensorRT']['class_ap50'].get(class_id, 0)
        print(f"   {CLASS_NAMES[class_id]:>12}: {pytorch_ap:.4f} → {tensorrt_ap:.4f} ({diff:+.4f})")

if __name__ == "__main__":
    print("Starting Comprehensive Benchmark Evaluation...")
    print("=" * 60)
    
    # Run benchmark
    results = benchmark_models()
    
    if results is None:
        print("Benchmark failed! Please check your dataset paths and try again.")
        sys.exit(1)
    
    # Print detailed results to console
    print_detailed_results(results)
    
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    try:
        # Create per-class bar chart
        print("Creating per-class bar chart...")
        create_per_class_bar_chart(results)
        
        # Create per-class line chart
        print("Creating per-class line chart...")
        create_per_class_line_chart(results)
        
        # Create mAP comparison chart
        print("Creating mAP comparison chart...")
        create_map_comparison_chart(results)
        
        # Create inference time comparison chart
        print("Creating inference time comparison chart...")
        create_inference_time_comparison_chart(results)
        
        # Create summary table
        print("Creating summary table...")
        create_combined_summary_table(results)
        
        # Create detailed per-class table
        print("Creating detailed per-class table...")
        create_per_class_detailed_table(results)
        
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("\nGenerated files:")
        print("   benchmarks/results/seg_per_class_bar_comparison.png")
        print("   benchmarks/results/seg_per_class_line_comparison.png")
        print("   benchmarks/results/seg_map_comparison.png")
        print("   benchmarks/results/seg_inference_time_comparison.png")
        print("   benchmarks/results/seg_summary_table.png")
        print("   benchmarks/results/seg_per_class_detailed_table.png")
        
        # Final summary
        speedup = results['PyTorch']['avg_inference_time'] / results['TensorRT']['avg_inference_time']
        ap50_diff = abs(results['TensorRT']['mean_ap50'] - results['PyTorch']['mean_ap50'])
        
        print(f"\nFINAL SUMMARY:")
        print(f"   Speed: TensorRT is {speedup:.1f}x faster")
        print(f"   Accuracy: mAP@50 difference of {ap50_diff:.4f}")
        print(f"   Trade-off: {'Excellent' if ap50_diff < 0.01 else 'Good' if ap50_diff < 0.02 else 'Acceptable'}")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        print("Results are still available in the results dictionary")
        
    print(f"\n{'='*60}")
    print("Benchmark evaluation completed!")
    print(f"{'='*60}")