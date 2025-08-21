"""
Utility functions for the EVEMASK Pipeline system.
Provides image preprocessing, postprocessing, bounding box utilities, mask operations, and blur effects for object detection and classification tasks.

Author: EVEMASK Team
Version: 1.0.0
"""

import torch
import numpy as np
import pycuda.driver as cuda
from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np
from numpy import ndarray
import torchvision.transforms as T
import torch.nn.functional as F
import random

# ========================================================================
# RANDOM SEED SETUP
# ========================================================================
def set_seed(seed=42):
    """
    Set the random seed for Python, numpy, and torch to ensure reproducibility.
    Args:
        seed (int): The seed value to use (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: The following line may cause errors if pycuda does not support 'init'.
    cuda.init()

# ========================================================================
# BOUNDING BOX CLASS
# ========================================================================
class BoundingBox:
    """
    Represents a bounding box for object detection results.
    Stores both absolute and normalized coordinates, class ID, and confidence score.
    """
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        """
        Initialize a bounding box with all required information.
        Args:
            classID (int): Class index
            confidence (float): Detection confidence
            x1, x2, y1, y2 (float): Absolute coordinates
            image_width, image_height (int): Original image dimensions
        """
        self.classID = classID
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        # Normalized coordinates
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height
    
    def box(self):
        """Return bounding box as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def width(self):
        """Return width of the bounding box."""
        return self.x2 - self.x1
    
    def height(self):
        """Return height of the bounding box."""
        return self.y2 - self.y1

    def center_absolute(self):
        """Return the absolute center coordinates of the bounding box."""
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))
    
    def center_normalized(self):
        """Return the normalized center coordinates of the bounding box."""
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))
    
    def size_absolute(self):
        """Return (width, height) in absolute pixel values."""
        return (self.x2 - self.x1, self.y2 - self.y1)
    
    def size_normalized(self):
        """Return (width, height) in normalized coordinates."""
        return (self.u2 - self.u1, self.v2 - self.v1)

# ========================================================================
# TENSORRT OUTPUT TO TORCH TENSOR
# ========================================================================
def copy_trt_output_to_torch_tensor(output_info):
    """
    Copy data from a TensorRT device pointer to a torch tensor on GPU.
    Args:
        output_info (dict): Contains device_ptr, shape, size, dtype
    Returns:
        torch.Tensor: Tensor on GPU with the specified shape and dtype
    """
    device_ptr = output_info['device_ptr']
    shape = output_info['shape']
    size = output_info['size']
    dtype = output_info['dtype'].type

    # Map numpy dtype to torch dtype
    dtype_map = {
        np.float32: torch.float32,
        np.float16: torch.float16,
        np.int32: torch.int32,
        np.int64: torch.int64,
    }
    torch_dtype = dtype_map.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    # Allocate empty tensor on GPU
    tensor_gpu = torch.empty(size, dtype=torch_dtype, device='cuda')
    # Copy data from TensorRT device pointer to torch tensor (may fail if pycuda does not support this)
    cuda.memcpy_dtod(tensor_gpu.data_ptr(), int(device_ptr), tensor_gpu.element_size() * size)
    # Reshape to original shape
    return tensor_gpu.view(*shape)

# ========================================================================
# TENSORRT OUTPUT POSTPROCESSING
# ========================================================================
def process_trt_output(tensor_pred0, number_of_class, conf_threshold=0.5):
    """
    Postprocess TensorRT output for batch inference, filter detections by confidence threshold.
    Args:
        tensor_pred0: Tensor [B, C, N] (batch, channel, num_boxes)
        number_of_class: Number of classes
        conf_threshold: Confidence threshold
    Returns:
        boxes_kept, scores_kept, class_ids_kept, mask_coeffs_kept, batch_idx, start_idx, end_idx
    """
    # Change from [B, C, N] to [B, N, C]
    tensor_pred0 = tensor_pred0.permute(0, 2, 1)
    boxes_cxcywh = tensor_pred0[:, :, :4]
    scores_all = tensor_pred0[:, :, 4:4 + number_of_class]
    mask_coeffs = tensor_pred0[:, :, 4 + number_of_class:]
    # Get max scores and class ids
    scores, class_ids = scores_all.max(dim=2)
    # Apply confidence threshold
    keep = scores > conf_threshold
    # Get indices of kept detections
    batch_idx, box_idx = keep.nonzero(as_tuple=True)
    if len(batch_idx) == 0:
        return [], [], [], [], [], [], []
    # Extract kept predictions
    boxes_kept = boxes_cxcywh[batch_idx, box_idx]
    scores_kept = scores[batch_idx, box_idx]
    class_ids_kept = class_ids[batch_idx, box_idx]
    mask_coeffs_kept = mask_coeffs[batch_idx, box_idx]
    # Convert cxcywh → xyxy
    cx, cy, w, h = boxes_kept.unbind(dim=1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_kept = torch.stack([x1, y1, x2, y2], dim=1)
    # Calculate start and end indices for each batch
    B = tensor_pred0.shape[0]
    valid_count = keep.sum(dim=1)
    start_idx = torch.cumsum(torch.cat([torch.tensor([0], device=valid_count.device), valid_count[:-1]], dim=0), dim=0)
    end_idx = start_idx + valid_count
    return boxes_kept, scores_kept, class_ids_kept, mask_coeffs_kept, batch_idx, start_idx, end_idx

# ========================================================================
# IMAGE RESIZING AND LETTERBOXING
# ========================================================================
def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) -> Tuple[ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image to the target shape, maintaining aspect ratio and adding border if needed.
    Args:
        im: Input image (ndarray)
        new_shape: Target shape (width, height)
        color: Border color
    Returns:
        im: Resized and padded image
        r: Scale ratio
        (dw, dh): Padding added (width, height)
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

# ========================================================================
# IMAGE TO BLOB CONVERSION
# ========================================================================
def blob(im: ndarray, return_seg: bool = False, half = True) -> Union[ndarray, Tuple]:
    """
    Convert image to blob (N, C, H, W) and normalize to [0, 1].
    Args:
        im: Input image (ndarray)
        return_seg: Whether to return normalized segmentation image
    Returns:
        im: Normalized blob [0, 1]
        imno255: Blob before division by 255
        seg: (optional) normalized segmentation image
    """
    seg = None
    if return_seg:
        seg = im.astype(np.float16) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    if half:
        imno255 = np.ascontiguousarray(im).astype(np.float16)
    else:
        imno255 = np.ascontiguousarray(im)
    im = imno255 / 255
    if return_seg:
        return im, imno255, seg
    else:
        return im, imno255

# ========================================================================
# POSTPROCESSING DETECTION RESULTS (CPU)
# ========================================================================
def postprocess_torch_cpu(boxes_cxcywh, scores, class_ids, masks, img_w, img_h, input_shape, ratio, dwdh):
    """
    Postprocess detection results: convert bounding boxes to original image, create masks, polygons, and BoundingBox objects.
    Args:
        boxes_cxcywh: [N, 4] tensor (cx, cy, w, h)
        scores: [N]
        class_ids: [N]
        masks: [N, 160, 160]
        img_w, img_h: Original image size
        input_shape: Model input size (width, height)
        ratio: Scale ratio from letterbox
        dwdh: Padding (dw, dh) from letterbox
    Returns:
        detected_objects: list[BoundingBox]
        polygons: list[list[tuple[int, int]]]
    """
    detected_objects = []
    polygons = []
    # Move to CPU and numpy
    boxes_cxcywh = boxes_cxcywh.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    class_ids = class_ids.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    # Convert cxcywh → xyxy
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    # Unpad and scale to original image
    dw, dh = dwdh
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy /= ratio
    # Clamp to image boundaries
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, img_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, img_h)
    for i in range(boxes_xyxy.shape[0]):
        box = boxes_xyxy[i]
        x1i, y1i, x2i, y2i = box
        w_box = int(round(x2i - x1i))
        h_box = int(round(y2i - y1i))
        if w_box <= 0 or h_box <= 0:
            continue
        # Get mask for this detection
        mask = masks[i]
        mask_h, mask_w = mask.shape
        # Calculate crop coordinates in mask space (160x160)
        crop_x1 = int(round(boxes_cxcywh[i, 0] - boxes_cxcywh[i, 2]/2) * mask_w / input_shape[0])
        crop_x2 = int(round(boxes_cxcywh[i, 0] + boxes_cxcywh[i, 2]/2) * mask_w / input_shape[0])
        crop_y1 = int(round(boxes_cxcywh[i, 1] - boxes_cxcywh[i, 3]/2) * mask_h / input_shape[1])
        crop_y2 = int(round(boxes_cxcywh[i, 1] + boxes_cxcywh[i, 3]/2) * mask_h / input_shape[1])
        # Clamp to mask bounds
        crop_x1 = max(0, min(crop_x1, mask_w))
        crop_x2 = max(0, min(crop_x2, mask_w))
        crop_y1 = max(0, min(crop_y1, mask_h))
        crop_y2 = max(0, min(crop_y2, mask_h))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue
        # Crop mask
        mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        if mask_crop.size == 0:
            continue
        # Resize to bounding box size in original image
        mask_resized = cv2.resize(mask_crop, (w_box, h_box), interpolation=cv2.INTER_LINEAR)
        # Binarize
        binary_mask = (mask_resized > 0.5).astype("uint8") * 255
        # Find contours and offset to correct position in original image
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = []
        for contour in contours:
            if len(contour) > 2:
                polygon_points = []
                for pt in contour:
                    x_offset = int(pt[0][0] + x1i)
                    y_offset = int(pt[0][1] + y1i)
                    polygon_points.append((x_offset, y_offset))
                if polygon_points:
                    polygon.append(polygon_points)
        bbox = BoundingBox(class_ids[i], scores[i], x1i, x2i, y1i, y2i, img_w, img_h)
        detected_objects.append(bbox)
        polygons.append(polygon)
    return detected_objects, polygons

# ========================================================================
# BLUR EFFECTS (RESIZE-BASED)
# ========================================================================
# def censored_options(image_tensor, downscale_factor=20):
#     if image_tensor.dim() == 3:
#         image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
    
#     _, _, h, w = image_tensor.shape
#     new_h = max(1, h // downscale_factor)
#     new_w = max(1, w // downscale_factor)

#     small_img = F.interpolate(image_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False, antialias=True)
#     blur_img = F.interpolate(small_img, size=(h, w), mode='bilinear', align_corners=False, antialias=True)

#     return blur_img.squeeze(0)

def censored_options(image_tensor, downscale_factor=20):
    """
    Applies a blur effect to the input image tensor by using a resize-down and resize-up approach.
    
    This method performs image blurring by first downsampling the image to a smaller resolution and 
    then upsampling it back to its original size. The downsampling removes high-frequency details, 
    and the upsampling smoothens the result, mimicking the effect of a blur filter. 

    Functionally, this technique is analogous to using `F.interpolate` with `mode='bilinear'` for 
    both the downscale and upscale operations. However, `T.Resize` from `torchvision.transforms` is 
    used here as it supports anti-aliasing via the `antialias=True` flag during downsampling. 
    Anti-aliasing helps reduce aliasing artifacts by applying a low-pass filter before resizing, 
    resulting in a more natural and visually smooth blur effect.

    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (C, H, W).
        downscale_factor (int): Factor by which the image is downscaled before being upscaled. 
                                A higher value results in stronger blur.

    Returns:
        torch.Tensor: Blurred image tensor of shape (C, H, W).
    """
    _, h, w = image_tensor.shape
    new_h = max(1, h // downscale_factor)
    new_w = max(1, w // downscale_factor)
    resize_down = T.Resize(size=(new_h, new_w), antialias=True)
    resize_up = T.Resize(size=(h, w), antialias=True)
    small_img = resize_down(image_tensor)
    blur_img = resize_up(small_img)
    return blur_img

# ========================================================================
# MASK RESIZING
# ========================================================================
def resize_mask_to_image(mask, target_h, target_w):
    """
    Resize mask to match the target image dimensions.
    Args:
        mask: torch.Tensor (H, W)
        target_h, target_w: Target dimensions
    Returns:
        resized_mask: torch.Tensor (H, W)
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    resized_mask = F.interpolate(mask, size=(target_h, target_w), mode='bilinear', align_corners=False)
    resized_mask = resized_mask.squeeze().clamp(0, 1)
    return resized_mask

# ========================================================================
# APPLY BLUR TO MASKED AREA
# ========================================================================
def apply_blur_to_masked_area(image_region, mask, downscale_factor=20):
    """
    Apply blur to the area of the image specified by the mask.
    Args:
        image_region: numpy array (H, W, C)
        mask: numpy array (H, W) with values 0-255
        downscale_factor: Blur strength
    Returns:
        processed_image: numpy array with blurred masked area
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = torch.from_numpy(image_region).permute(2, 0, 1).float().to(device) / 255.0
    blur_img = censored_options(image_tensor, downscale_factor)
    mask_tensor = torch.from_numpy(mask).float().to(device) / 255.0
    h, w = image_tensor.shape[1], image_tensor.shape[2]
    if mask_tensor.shape != (h, w):
        mask_tensor = resize_mask_to_image(mask_tensor, h, w)
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
        
    # Blend (1-mask)*image + mask*blur_image
    processed_image = (1 - mask_tensor) * image_tensor + mask_tensor * blur_img
    processed_image = processed_image.permute(1, 2, 0).cpu().numpy()
    processed_image = (processed_image * 255).astype(np.uint8)
    return processed_image

# ========================================================================
# CONDITIONAL BLUR BASED ON CLASS
# ========================================================================
def draw_masks_conditional_blur(frame, detected_objects, polygons, class_ids, downscale_factor=20, no_blur_classes=None):
    """
    Conditionally blur objects in the image except for specified classes.
    Args:
        frame: Original image (ndarray)
        detected_objects: List of BoundingBox
        polygons: List of polygons (list[tuple(x, y)])
        class_ids: Tensor containing class_id for each object
        downscale_factor: Blur strength
        no_blur_classes: List of class_ids NOT to blur
    Returns:
        frame: Image with conditional blur applied
    """
    if no_blur_classes is None:
        no_blur_classes = []
    result_frame = frame.copy()
    for i, (obj, poly_group) in enumerate(zip(detected_objects, polygons)):
        if not poly_group:
            continue
        current_class_id = class_ids[i].item() if torch.is_tensor(class_ids[i]) else class_ids[i]
        should_blur = current_class_id not in no_blur_classes
        if not should_blur:
            continue
        x1, y1, x2, y2 = int(obj.x1), int(obj.y1), int(obj.x2), int(obj.y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        bbox_region = result_frame[y1:y2, x1:x2].copy()
        bbox_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        for polygon in poly_group:
            if len(polygon) > 2:
                offset_polygon = [(max(0, min(x-x1, x2-x1-1)), max(0, min(y-y1, y2-y1-1))) for x, y in polygon]
                pts = np.array(offset_polygon, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(bbox_mask, [pts], 255)
        try:
            blurred_region = apply_blur_to_masked_area(bbox_region, bbox_mask, downscale_factor)
            result_frame[y1:y2, x1:x2] = blurred_region
        except Exception as e:
            print(f"Error applying blur: {e}")
            continue
    return result_frame

# ========================================================================
# DRAW BOUNDING BOXES AND LABELS
# ========================================================================
def draw_bboxes(frame, bboxes, class_names=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes and labels on the image.
    Args:
        frame: Original image (ndarray)
        bboxes: List of BoundingBox
        class_names: List of class names (optional)
        color: Bounding box color
        thickness: Bounding box thickness
    Returns:
        frame: Image with bounding boxes drawn
    """
    for bbox in bboxes:
        x1 = int(bbox.x1)
        y1 = int(bbox.y1)
        x2 = int(bbox.x2)
        y2 = int(bbox.y2)
        class_id = bbox.classID
        score = bbox.confidence
        label = f"{class_names[class_id]}: {score:.2f}" if class_names else f"ID:{class_id} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return frame

def apply_optimized_blur(frame, masks_gpu, boxes_gpu, class_ids_gpu, frame_ratio, frame_dwdh, CLASSES_NO_BLUR):
    """
    Apply optimized blur processing using the exact same logic as original draw_masks_conditional_blur.
    This maintains blur quality while minimizing GPU-CPU transfers.
    
    Args:
        frame: numpy array (H, W, C) - original frame
        masks_gpu: torch.Tensor (N, 160, 160) on GPU - segmentation masks
        boxes_gpu: torch.Tensor (N, 4) on GPU - bounding boxes in xyxy format
        class_ids_gpu: torch.Tensor (N,) on GPU - class IDs
        frame_ratio: float - scaling ratio from preprocessing
        frame_dwdh: tuple - padding offsets from preprocessing
        CLASSES_NO_BLUR: class is not blur
    
    Returns:
        frame: numpy array with blur applied
    """
    if masks_gpu is None or boxes_gpu is None:
        return frame
    
    # First, we need to create detected_objects and polygons using the same logic as original
    # Convert data to CPU for compatibility with existing postprocess_torch_cpu
    boxes_cpu = boxes_gpu.cpu()
    class_ids_cpu = class_ids_gpu.cpu()
    
    # Convert boxes from xyxy to cxcywh for postprocessing (same as original)
    x1, y1, x2, y2 = boxes_cpu.unbind(dim=1)
    cx = (x1 + x2) / 2  # Center x
    cy = (y1 + y2) / 2  # Center y
    w = x2 - x1  # Width
    h = y2 - y1  # Height
    boxes_cxcywh = torch.stack([cx, cy, w, h], dim=1)
    
    # Create dummy scores (we don't need them for blur, but postprocess_torch_cpu expects them)
    scores_dummy = torch.ones(len(boxes_cpu))
    
    # Use the exact same postprocessing as original to get detected_objects and polygons
    detected_objects, polygons = postprocess_torch_cpu(
        boxes_cxcywh=boxes_cxcywh,
        scores=scores_dummy,
        class_ids=class_ids_cpu,
        masks=masks_gpu,  # This handles GPU->CPU conversion internally
        img_w=frame.shape[1],
        img_h=frame.shape[0],
        input_shape=(640, 640),
        ratio=frame_ratio,
        dwdh=frame_dwdh
    )
    
    # Now use the EXACT same blur logic as original draw_masks_conditional_blur
    result_frame = frame.copy()
    
    for i, (obj, poly_group) in enumerate(zip(detected_objects, polygons)):
        if not poly_group:
            continue
        
        current_class_id = class_ids_cpu[i].item() if torch.is_tensor(class_ids_cpu[i]) else class_ids_cpu[i]
        should_blur = current_class_id not in CLASSES_NO_BLUR
        
        if not should_blur:
            continue
        
        # Use EXACT same coordinate logic as original
        x1, y1, x2, y2 = int(obj.x1), int(obj.y1), int(obj.x2), int(obj.y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        bbox_region = result_frame[y1:y2, x1:x2].copy()
        bbox_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        
        # Use EXACT same polygon filling logic as original
        for polygon in poly_group:
            if len(polygon) > 2:
                offset_polygon = [(max(0, min(x-x1, x2-x1-1)), max(0, min(y-y1, y2-y1-1))) for x, y in polygon]
                pts = np.array(offset_polygon, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(bbox_mask, [pts], 255)
        
        try:
            # Use EXACT same blur function as original
            blurred_region = apply_blur_to_masked_area(bbox_region, bbox_mask, downscale_factor=20)
            result_frame[y1:y2, x1:x2] = blurred_region
        except Exception as e:
            print(f"Error applying blur: {e}")
            continue
    
    return result_frame

