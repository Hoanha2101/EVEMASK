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


# Đảm bảo tính nhất quán
def set_seed(seed=42):
    """
    Đặt seed cho tất cả các random number generator
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Đảm bảo deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Đặt seed cho CUDA context
    cuda.init()

class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = classID
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height
    
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)
        
    def width(self):
        return self.x2 - self.x1
    
    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))
    
    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))
    
    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)
    
    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)
    
def copy_trt_output_to_torch_tensor(output_info):
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

    # Tạo tensor rỗng trên GPU
    tensor_gpu = torch.empty(size, dtype=torch_dtype, device='cuda')

    # Copy dữ liệu từ TensorRT device ptr vào torch tensor
    cuda.memcpy_dtod(tensor_gpu.data_ptr(), int(device_ptr), tensor_gpu.element_size() * size)

    # Reshape về shape ban đầu
    return tensor_gpu.view(*shape)

# === Hàm xử lý đầu ra TensorRT ===
def process_trt_output(tensor_pred0, number_of_class, conf_threshold=0.5):
    """
    Process TensorRT output for batch inference
    tensor_pred0: [B, C, N] format
    """
    # Chuyển từ [B, C, N] sang [B, N, C]
    tensor_pred0 = tensor_pred0.permute(0, 2, 1)  # [B, N, C]
    
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


def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)


def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    seg = None
    if return_seg:
        seg = im.astype(np.float16) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    imno255 = np.ascontiguousarray(im).astype(np.float16)
    im = imno255 / 255
    if return_seg:
        return im, imno255, seg
    else:
        return im, imno255

def postprocess_torch_cpu(boxes_cxcywh, scores, class_ids, masks, img_w, img_h, input_shape, ratio, dwdh):
    """
    Xử lý hậu xử lý (postprocess) hoàn toàn trên CPU với rescaling đúng.
    Args:
        boxes_cxcywh: [N, 4] tensor trên GPU hoặc CPU
        scores: [N]
        class_ids: [N]
        masks: [N, 160, 160]
        img_w, img_h: kích thước ảnh gốc
        input_shape: kích thước input của model (width, height)
        ratio: tỷ lệ scale từ letterbox
        dwdh: padding (dw, dh) từ letterbox
    Returns:
        detected_objects: list[BoundingBox]
        polygons: list[list[tuple[int, int]]]
    """

    detected_objects = []
    polygons = []

    # Đưa tất cả về CPU và numpy
    boxes_cxcywh = boxes_cxcywh.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    class_ids = class_ids.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()

    # Tính lại boxes: cxcywh → xyxy  
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Unpad và scale boxes về ảnh gốc (reverse letterbox transform)
    dw, dh = dwdh
    boxes_xyxy[:, [0, 2]] -= dw  # Remove width padding
    boxes_xyxy[:, [1, 3]] -= dh  # Remove height padding
    boxes_xyxy /= ratio  # Scale back to original size

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

        # Lấy mask tương ứng với detection thứ i
        mask = masks[i]  # shape [160,160]
        mask_h, mask_w = mask.shape

        # Tính toạ độ crop trong mask space (160x160)
        # boxes_cxcywh ở đây vẫn trong không gian input (640x640)
        # Cần map về mask space
        crop_x1 = int(round(boxes_cxcywh[i, 0] - boxes_cxcywh[i, 2]/2) * mask_w / input_shape[0])
        crop_x2 = int(round(boxes_cxcywh[i, 0] + boxes_cxcywh[i, 2]/2) * mask_w / input_shape[0])
        crop_y1 = int(round(boxes_cxcywh[i, 1] - boxes_cxcywh[i, 3]/2) * mask_h / input_shape[1])
        crop_y2 = int(round(boxes_cxcywh[i, 1] + boxes_cxcywh[i, 3]/2) * mask_h / input_shape[1])

        # Clamp để đảm bảo trong phạm vi mask
        crop_x1 = max(0, min(crop_x1, mask_w))
        crop_x2 = max(0, min(crop_x2, mask_w))
        crop_y1 = max(0, min(crop_y1, mask_h))
        crop_y2 = max(0, min(crop_y2, mask_h))

        # Đảm bảo crop_x2 > crop_x1 và crop_y2 > crop_y1
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        # Crop mask
        mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        if mask_crop.size == 0:
            continue

        # Resize về đúng kích thước bbox trong ảnh gốc
        mask_resized = cv2.resize(mask_crop, (w_box, h_box), interpolation=cv2.INTER_LINEAR)

        # Binarize
        binary_mask = (mask_resized > 0.5).astype("uint8") * 255

        # Contour to polygon - VÀ OFFSET ĐẾN VỊ TRÍ ĐÚNG
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = []
        for contour in contours:
            if len(contour) > 2:  # Cần ít nhất 3 điểm để tạo polygon
                # QUAN TRỌNG: Offset polygon về vị trí đúng trong ảnh gốc
                polygon_points = []
                for pt in contour:
                    x_offset = int(pt[0][0] + x1i)  # Cộng với x1 của bbox
                    y_offset = int(pt[0][1] + y1i)  # Cộng với y1 của bbox
                    polygon_points.append((x_offset, y_offset))
                if polygon_points:
                    polygon.append(polygon_points)

        # Tạo BoundingBox
        bbox = BoundingBox(class_ids[i], scores[i], x1i, x2i, y1i, y2i, img_w, img_h)
        detected_objects.append(bbox)
        polygons.append(polygon)

    return detected_objects, polygons

def censored_options(image_tensor, downscale_factor=20):
    """Apply resizing and blurring to the image tensor."""
    # Get original dimensions
    _, h, w = image_tensor.shape
    
    # Calculate new dimensions
    new_h = max(1, h // downscale_factor)
    new_w = max(1, w // downscale_factor)
    
    # Resize down then up for blur effect
    resize_down = T.Resize(size=(new_h, new_w), antialias=True)
    resize_up = T.Resize(size=(h, w), antialias=True)
    
    small_img = resize_down(image_tensor)
    blur_img = resize_up(small_img)
    
    return blur_img

def resize_mask_to_image(mask, target_h, target_w):
    """
    Resize mask to target image dimensions
    Args:
        mask: Torch tensor (H, W)
        target_h, target_w: Target dimensions
    Returns:
        Resized mask tensor
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)  # Add batch dim
    
    # Resize mask to target dimensions
    resized_mask = F.interpolate(mask, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    # Remove extra dimensions and clamp values
    resized_mask = resized_mask.squeeze().clamp(0, 1)
    
    return resized_mask

def apply_blur_to_masked_area(image_region, mask, downscale_factor=20):
    """
    Apply blur to the area of the image specified by the mask.
    Args:
        image_region: numpy array (H, W, C)
        mask: numpy array (H, W) with values 0-255
        downscale_factor: factor for blur effect
    Returns:
        processed image as numpy array
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert image to tensor (C, H, W) and normalize to [0, 1]
    image_tensor = torch.from_numpy(image_region).permute(2, 0, 1).float().to(device) / 255.0
    
    # Apply blur effect
    blur_img = censored_options(image_tensor, downscale_factor)
    
    # Convert mask to tensor and normalize to [0, 1]
    mask_tensor = torch.from_numpy(mask).float().to(device) / 255.0
    
    # Resize mask to match image dimensions if needed
    h, w = image_tensor.shape[1], image_tensor.shape[2]
    if mask_tensor.shape != (h, w):
        mask_tensor = resize_mask_to_image(mask_tensor, h, w)
    
    # Ensure mask is in correct shape for broadcasting
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dim
    
    # Apply blur to masked area: original * (1 - mask) + blurred * mask
    processed_image = (1 - mask_tensor) * image_tensor + mask_tensor * blur_img
    
    # Convert back to numpy (H, W, C) and scale to [0, 255]
    processed_image = processed_image.permute(1, 2, 0).cpu().numpy()
    processed_image = (processed_image * 255).astype(np.uint8)
    
    return processed_image

def draw_masks_conditional_blur(frame, detected_objects, polygons, class_ids, 
                               downscale_factor=20, no_blur_classes=None):
    """
    Conditional blur using resize method - blur everything EXCEPT specified classes
    Args:
        frame: ảnh gốc (ndarray)
        detected_objects: danh sách BoundingBox
        polygons: danh sách list[tuple(x, y)]
        class_ids: tensor chứa class_id cho từng object
        downscale_factor: factor for blur effect (higher = more blur)
        no_blur_classes: list các class_id KHÔNG cần blur (giữ nguyên)
    Returns:
        frame đã được blur có điều kiện
    """
    
    if no_blur_classes is None:
        no_blur_classes = []  # Blur tất cả nếu không chỉ định
    
    result_frame = frame.copy()
    
    for i, (obj, poly_group) in enumerate(zip(detected_objects, polygons)):
        if not poly_group:
            continue
        
        # Kiểm tra xem object này có cần blur không
        current_class_id = class_ids[i].item() if torch.is_tensor(class_ids[i]) else class_ids[i]
        should_blur = current_class_id not in no_blur_classes  # Blur if NOT in no_blur_classes
        
        if not should_blur:
            continue  # Skip nếu không cần blur (class = 0)
            
        # Get bounding box coordinates
        x1, y1, x2, y2 = int(obj.x1), int(obj.y1), int(obj.x2), int(obj.y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Extract region of interest
        bbox_region = result_frame[y1:y2, x1:x2].copy()
        bbox_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        
        # Create mask from polygons
        for polygon in poly_group:
            if len(polygon) > 2:
                # Offset polygon coordinates to bbox region
                offset_polygon = [(max(0, min(x-x1, x2-x1-1)), max(0, min(y-y1, y2-y1-1))) 
                                 for x, y in polygon]
                pts = np.array(offset_polygon, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(bbox_mask, [pts], 255)
        
        # Apply blur to masked area using the new method
        try:
            blurred_region = apply_blur_to_masked_area(bbox_region, bbox_mask, downscale_factor)
            result_frame[y1:y2, x1:x2] = blurred_region
        except Exception as e:
            print(f"Error applying blur: {e}")
            # Fallback to original region if blur fails
            continue
    
    return result_frame

def draw_bboxes(frame, bboxes, class_names=None, color=(0, 255, 0), thickness=2):
    for bbox in bboxes:
        x1 = int(bbox.x1)
        y1 = int(bbox.y1)
        x2 = int(bbox.x2)
        y2 = int(bbox.y2)
        class_id = bbox.classID
        score = bbox.confidence

        label = f"{class_names[class_id]}: {score:.2f}" if class_names else f"ID:{class_id} {score:.2f}"

        # Vẽ bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Vẽ nhãn nền
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 4), (x1 + w, y1), color, -1)

        # Vẽ chữ
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return frame

