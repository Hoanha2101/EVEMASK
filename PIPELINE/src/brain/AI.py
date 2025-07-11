from ..tools import *
from ..models.initNet import net1
from ..tools.vectorPrepare import VectorPrepare
import torchvision.ops as ops
import numpy as np
import torch
import time
from ..controllers import CircleQueue

class AI:
    def __init__(self, cfg, blurPlot = True , boxPlot = False ,FEmodel = True):
        
        self.circle_queue = CircleQueue.get_instance()
        
        self.batch_size = cfg['batch_size']
        self.number_of_class = cfg['nc']
        self.conf_threshold = cfg['conf_threshold']
        self.iou_threshold = cfg['iou_threshold']
        self.CLASS_NAMES = cfg['names']
        self.CLASSES_NO_BLUR = cfg['CLASSES_NO_BLUR']
        
        self.FEmodel = FEmodel
        self.blurPlot = blurPlot
        self.boxPlot = boxPlot
        self.current_frame = None

        self._instance_list_ = None

        if FEmodel:
            from ..models.initNet import net2
            self.net2 = net2
            orgFolderPath = cfg["recognizeData_path"]
            vectorizer = VectorPrepare(orgFolderPath=orgFolderPath, enginePlan=self.net2)
            self.recognizeDataVector, self.image_names = vectorizer.run()

    def _get_frames_from_queue(self):
        use_count = min(self.batch_size, self.circle_queue.queue_length())
        frames = self.circle_queue.get_frame_non_processed(use_count)
        self._instance_list_ = frames
        return frames

    def run(self):
        while True:
            if self.circle_queue.queue_length() > 0:
                frames = self._get_frames_from_queue()
                if frames:
                    processed_batch = [frame.framePreprocessing() for frame in frames]
                    self.inference(processed_batch)
            time.sleep(0.01)

    def inference(self, processed_batch):
        origin_imno255 = np.concatenate([item[4] for item in processed_batch], axis=0)
        batch_tensor = np.concatenate([item[1] for item in processed_batch], axis=0)
        x = torch.from_numpy(origin_imno255).to(device='cuda', dtype=torch.float16)
        
        # TensorRT Inference
        net1.cuda_ctx.push()
        results = net1.infer(batch_tensor)
        net1.cuda_ctx.pop()
        
        tensor_pred0 = copy_trt_output_to_torch_tensor(results[0])  # [B, C, N]
        tensor_pred1_2 = copy_trt_output_to_torch_tensor(results[1])  # [B, 32, 160, 160]
        
        # Process predictions
        (boxes_kept, 
        scores_kept, 
        class_ids_kept, 
        mask_coeffs_kept, 
        batch_indices, 
        start_idx, 
        end_idx
        ) = process_trt_output(tensor_pred0, 
                                self.number_of_class, 
                                self.conf_threshold
                                )
        
        # Process each frame in batch
        for b in range(min(self.batch_size, len(processed_batch))):
            current_frame, _, frame_ratio, frame_dwdh, _ = processed_batch[b]
            
            # Khởi tạo giá trị mặc định
            detected_objects = []
            polygons = []
            final_class_ids = torch.tensor([])
            
            # Get detections for current batch item
            if len(start_idx) > b and start_idx[b] < end_idx[b]:
                b_boxes = boxes_kept[start_idx[b]:end_idx[b]]
                b_scores = scores_kept[start_idx[b]:end_idx[b]]
                b_classes = class_ids_kept[start_idx[b]:end_idx[b]]
                b_masks = mask_coeffs_kept[start_idx[b]:end_idx[b]]
                
                # Sửa lỗi: không so sánh trực tiếp tensor với boolean
                if b_boxes.numel() > 0:
                    # Apply NMS for current batch item
                    nms_indices = ops.batched_nms(
                        b_boxes, b_scores, b_classes, self.iou_threshold
                    )

                    b_boxes_nms = b_boxes[nms_indices]
                    b_scores_nms = b_scores[nms_indices]
                    b_classes_nms = b_classes[nms_indices]
                    b_masks_nms = b_masks[nms_indices]
                    

                    if b_boxes_nms.numel() > 0: # Total element
                        # Separate objects by class_id
                        class_0_mask = (b_classes_nms == 0)
                        non_class_0_mask = (b_classes_nms != 0)
                    
                        # Generate masks for all objects first
                        protos = tensor_pred1_2[b]  # [32, 160, 160]
                        protos_reshaped = protos.view(32, -1)  # [32, 25600]
                        masks = torch.matmul(b_masks_nms, protos_reshaped).sigmoid()
                        masks = masks.view(-1, 160, 160)
                        
                        # Convert boxes back to cxcywh for postprocessing
                        x1, y1, x2, y2 = b_boxes_nms.unbind(dim=1)
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        boxes_cxcywh_nms = torch.stack([cx, cy, w, h], dim=1)

                        # Initialize final class predictions
                        final_class_ids = b_classes_nms.clone()
                        
                        # Use mode Feature extraction
                        if self.FEmodel:
                            class_0_indices = torch.nonzero(class_0_mask, as_tuple=False).squeeze(1)
        
                            # Process class_id == 0 objects with second model to verify
                            if class_0_mask.any():
                                class_0_boxes = b_boxes_nms[class_0_mask]
                                
                                # RoIAlign for class_id == 0 objects only
                                batch_index = torch.full((class_0_boxes.shape[0], 1), b, 
                                                    dtype=torch.float16, device=b_boxes.device)
                                
                                roi_boxes = torch.cat([batch_index, class_0_boxes], dim=1).half()
                                
                                roi_features = ops.roi_align(
                                    x, roi_boxes, output_size=(224, 224),
                                    spatial_scale=1.0, aligned=True
                                )

                                roi_features = roi_features.half().contiguous()
                                num_rois = roi_features.shape[0]

                                outputs_all = []

                                if num_rois <= self.batch_size:
                                    outputs = self.net2.infer(roi_features)
                                    outputs_all.append(outputs)
                                else:
                                    # Cắt thành nhiều batch nhỏ và truyền từng phần
                                    for i in range(0, num_rois, self.batch_size):
                                        chunk = roi_features[i:i+self.batch_size]
                                        output_chunk = self.net2.infer(chunk)
                                        outputs_all.append(output_chunk)
                                # Ghép tất cả output lại
                                outputs = torch.cat(outputs_all, dim=0)
                                outputs = outputs.cpu().numpy()

                                if np.isnan(outputs).any():
                                    print("[Warning] NaN detected in net2 outputs. Skipping similarity matching.")
                                    continue

                                recognizeDataVector_array = np.array(self.recognizeDataVector)  # shape (N, D)
                                
                                final_class_ids = SimilarityMethod(final_class_ids, class_0_indices, recognizeDataVector_array, outputs)
                            
                        # Postprocess all objects
                        detected_objects, polygons = postprocess_torch_cpu(
                                    boxes_cxcywh=boxes_cxcywh_nms,
                                    scores=b_scores_nms,
                                    class_ids=final_class_ids,
                                    masks=masks,
                                    img_w=current_frame.shape[1],
                                    img_h=current_frame.shape[0],
                                    input_shape=(640, 640),
                                    ratio=frame_ratio,
                                    dwdh=frame_dwdh)
            if self.boxPlot:           
                current_frame = draw_bboxes(
                    current_frame, 
                    detected_objects, 
                    class_names=self.CLASS_NAMES)
            
            if self.blurPlot:
                current_frame = draw_masks_conditional_blur(
                    current_frame, 
                    detected_objects, 
                    polygons, 
                    final_class_ids,
                    downscale_factor=20, 
                    no_blur_classes=self.CLASSES_NO_BLUR)
                    
            self._instance_list_[b].frame_data = current_frame
            self._instance_list_[b].processed = True
