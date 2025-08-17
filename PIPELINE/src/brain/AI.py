"""
AI Processing Module
Main AI engine for real-time video processing using YOLO segmentation and feature extraction.

This module implements:
- YOLO-based object detection and segmentation
- Feature extraction for object classification
- Batch processing with dynamic frame skipping
- Real-time performance optimization
- Conditional blurring based on object classes

Key Components:
- TensorRT inference for high performance
- RoI alignment for feature extraction
- Similarity-based classification
- FPS monitoring and optimization

Author: EVEMASK Team
"""

from ..tools import *
from ..models.initNet import net1
from ..tools.vectorPrepare import VectorPrepare
import torchvision.ops as ops
import numpy as np
import torch
import time
from ..controllers import CircleQueue
from ..logger import EveMaskLogger

class AI:
    """
    Main AI processing engine using singleton pattern.
    
    This class handles:
    - Real-time video frame processing
    - YOLO object detection and segmentation
    - Feature extraction for object classification
    - Batch processing optimization
    - Performance monitoring and FPS control
    
    Attributes:
        circle_queue: Circular queue for frame management
        batch_size: Number of frames to process in each batch
        number_of_class: Total number of object classes
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        CLASS_NAMES: Dictionary mapping class IDs to names
        CLASSES_NO_BLUR: List of class IDs that should not be blurred
        FEmodel: Whether to use feature extraction model
        blurPlot: Whether to apply conditional blurring
        boxPlot: Whether to draw bounding boxes
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(AI, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, cfg, blurPlot = True , boxPlot = False ,FEmodel = True):
        """
        Initialize AI processing engine.
        
        Args:
            cfg (dict): Configuration dictionary
            blurPlot (bool): Enable conditional blurring
            boxPlot (bool): Enable bounding box drawing
            FEmodel (bool): Enable feature extraction model
        """
        # Get singleton instance of circular queue
        self.circle_queue = CircleQueue.get_instance()
        self.logger = EveMaskLogger.get_instance()
        
        # Load configuration parameters
        self.batch_size = cfg['batch_size']
        self.number_of_class = cfg['nc']
        self.conf_threshold = cfg['conf_threshold']
        self.iou_threshold = cfg['iou_threshold']
        self.CLASS_NAMES = cfg['names']
        self.CLASSES_NO_BLUR = cfg['CLASSES_NO_BLUR']
        
        # Processing flags
        self.FEmodel = FEmodel
        self.blurPlot = blurPlot
        self.boxPlot = boxPlot
        self.current_frame = None

        # Frame processing list
        self._instance_list_ = None

        # FPS monitoring
        self._instream_fps_ = 25  # Default input FPS
        self._ai_fps_ = 10  # Default AI processing FPS
        self._processing_times = []  # Store processing times for FPS calculation
        self._last_fps_update = time.time()
        
        # Frame marking completed by AI
        self.mooc_processed_frames = 0

        # Initialize feature extraction if enabled
        if FEmodel:
            from ..models.initNet import net2
            self.net2 = net2
            
            # Load and prepare reference data for similarity matching
            orgFolderPath = cfg["recognizeData_path"]
            vectorizer = VectorPrepare(orgFolderPath=orgFolderPath, enginePlan=self.net2)
            self.recognizeDataVector, self.image_names = vectorizer.run()

    def _get_frames_from_queue(self):
        """
        Get frames from circular queue with dynamic frame skipping.
        
        This method implements intelligent frame skipping based on the ratio
        between input FPS and AI processing FPS to optimize performance.
        
        Returns:
            list: List of frames to process in current batch
        """
        # Get available frames (limited by batch size)
        use_count = min(self.batch_size, self.circle_queue.queue_length())
        
        # Calculate skip frames based on FPS ratio
        # Formula: n_skip = max(0, int((instream_fps / ai_fps) - 1))
        # If AI FPS = 10, Input FPS = 25 -> n_skip = 1 (process 1 frame, skip 1 frame)
        # If AI FPS = 5, Input FPS = 25 -> n_skip = 4 (process 1 frame, skip 4 frames)
        if self._ai_fps_ > 0:
            n_skip = max(0, int(((self._instream_fps_ / self._ai_fps_)* self.batch_size) - self.batch_size))
        else:
            n_skip = 0
        self.logger.update_n_skip_frames(n_skip)
        # Get frames from queue with skipping
        frames = self.circle_queue.get_frame_non_processed(use_count, n_skip)
        self._instance_list_ = frames
        return frames

    def run(self):
        """
        Main AI processing loop.
        
        This method runs continuously in a separate thread and:
        - Gets frames from the circular queue
        - Preprocesses frames for AI inference
        - Performs object detection and classification
        - Applies post-processing (blurring, drawing)
        - Updates frame data in the queue
        """
        while True:
            # Check if frames are available for processing
            if self.circle_queue.queue_length() > 0:
                # Get frames with dynamic skipping
                frames = self._get_frames_from_queue()
                if frames:
                    # Preprocess all frames in batch
                    processed_batch = [frame.framePreprocessing() for frame in frames]
                    # Perform AI inference on batch
                    self.inference(processed_batch)
            time.sleep(0.01)  # Short sleep to prevent busy waiting

    def _update_ai_fps(self, processing_time):
        """
        Update AI processing FPS based on actual processing time.
        
        This method tracks processing times and calculates the actual
        AI processing FPS to optimize frame skipping strategy.
        
        Args:
            processing_time (float): Time taken for current batch processing
        """
        self._processing_times.append(processing_time)
        
        # Only update FPS every 5 seconds to avoid fluctuations
        current_time = time.time()
        if current_time - self._last_fps_update > 5.0:
            if len(self._processing_times) > 0:
                # Calculate average processing time
                avg_processing_time = sum(self._processing_times) / len(self._processing_times)
                if avg_processing_time > 0:
                    # Update AI FPS based on average processing time
                    self._ai_fps_ = 1.0 / avg_processing_time
                    self.logger.update_ai_fps(round(self._ai_fps_, 2))
            
            # Reset tracking for next calculation
            self._processing_times = []
            self._last_fps_update = current_time
            
    # def inference(self, processed_batch):
    #     """
    #     Perform AI inference on a batch of preprocessed frames.
        
    #     This method handles the complete AI pipeline:
    #     1. YOLO object detection and segmentation
    #     2. Non-maximum suppression (NMS)
    #     3. Feature extraction for class 0 objects
    #     4. Similarity-based classification
    #     5. Post-processing and visualization
        
    #     Args:
    #         processed_batch (list): List of preprocessed frame data
    #     """
    #     start_time = time.time()
        
    #     # Prepare input tensors for inference
    #     origin_imno255 = np.concatenate([item[4] for item in processed_batch], axis=0)
    #     batch_tensor = np.concatenate([item[1] for item in processed_batch], axis=0)
    #     x = torch.from_numpy(origin_imno255).to(device='cuda', dtype=torch.float16)
        
    #     # Perform TensorRT inference on YOLO model
    #     net1.cuda_ctx.push()
    #     results = net1.infer(batch_tensor)
    #     net1.cuda_ctx.pop()
        
    #     # Extract prediction tensors
    #     tensor_pred0 = copy_trt_output_to_torch_tensor(results[0])  # [B, C, N] - detection outputs
    #     tensor_pred1_2 = copy_trt_output_to_torch_tensor(results[1])  # [B, 32, 160, 160] - mask prototypes
        
    #     # Process YOLO predictions to get detections
    #     (boxes_kept, 
    #     scores_kept, 
    #     class_ids_kept, 
    #     mask_coeffs_kept, 
    #     batch_indices, 
    #     start_idx, 
    #     end_idx
    #     ) = process_trt_output(tensor_pred0, 
    #                             self.number_of_class, 
    #                             self.conf_threshold
    #                             )
        
    #     # Process each frame in the batch
    #     for b in range(min(self.batch_size, len(processed_batch))):
    #         # Extract frame data and preprocessing parameters
    #         current_frame, _, frame_ratio, frame_dwdh, _ = processed_batch[b]
            
    #         # Initialize default values for this frame
    #         detected_objects = []
    #         polygons = []
    #         final_class_ids = torch.tensor([])
            
    #         # Get detections for current batch item
    #         if len(start_idx) > b and start_idx[b] < end_idx[b]:
    #             # Extract detections for this frame
    #             b_boxes = boxes_kept[start_idx[b]:end_idx[b]]
    #             b_scores = scores_kept[start_idx[b]:end_idx[b]]
    #             b_classes = class_ids_kept[start_idx[b]:end_idx[b]]
    #             b_masks = mask_coeffs_kept[start_idx[b]:end_idx[b]]
                
    #             # Check if any detections exist
    #             if b_boxes.numel() > 0:
    #                 # Apply Non-Maximum Suppression (NMS) to remove overlapping detections
    #                 nms_indices = ops.batched_nms(
    #                     b_boxes, b_scores, b_classes, self.iou_threshold
    #                 )

    #                 # Get NMS-filtered detections
    #                 b_boxes_nms = b_boxes[nms_indices]
    #                 b_scores_nms = b_scores[nms_indices]
    #                 b_classes_nms = b_classes[nms_indices]
    #                 b_masks_nms = b_masks[nms_indices]
                    
    #                 # Check if any detections remain after NMS
    #                 if b_boxes_nms.numel() > 0:
    #                     # Separate objects by class_id for different processing
    #                     class_0_mask = (b_classes_nms == 0)  # Objects that need feature extraction
    #                     non_class_0_mask = (b_classes_nms != 0)  # Objects with direct classification
                    
    #                     # Generate segmentation masks for all objects
    #                     protos = tensor_pred1_2[b]  # [32, 160, 160] - mask prototypes
    #                     protos_reshaped = protos.view(32, -1)  # [32, 25600] - flattened prototypes
    #                     masks = torch.matmul(b_masks_nms, protos_reshaped).sigmoid()  # Generate masks
    #                     masks = masks.view(-1, 160, 160)  # Reshape to [N, 160, 160]
                        
    #                     # Convert bounding boxes from xyxy to cxcywh format for postprocessing
    #                     x1, y1, x2, y2 = b_boxes_nms.unbind(dim=1)
    #                     cx = (x1 + x2) / 2  # Center x
    #                     cy = (y1 + y2) / 2  # Center y
    #                     w = x2 - x1  # Width
    #                     h = y2 - y1  # Height
    #                     boxes_cxcywh_nms = torch.stack([cx, cy, w, h], dim=1)

    #                     # Initialize final class predictions
    #                     final_class_ids = b_classes_nms.clone()
                        
    #                     # Feature extraction mode for class 0 objects
    #                     if self.FEmodel:
    #                         class_0_indices = torch.nonzero(class_0_mask, as_tuple=False).squeeze(1)
        
    #                         # Process class_id == 0 objects with second model for verification
    #                         if class_0_mask.any():
    #                             class_0_boxes = b_boxes_nms[class_0_mask]
                                
    #                             # RoI (Region of Interest) alignment for feature extraction
    #                             batch_index = torch.full((class_0_boxes.shape[0], 1), b, 
    #                                                 dtype=torch.float16, device=b_boxes.device)
                                
    #                             roi_boxes = torch.cat([batch_index, class_0_boxes], dim=1).half()
                                
    #                             # Extract features using RoI alignment
    #                             # roi_features = ops.roi_align(
    #                             #     x, roi_boxes, output_size=(224, 224),
    #                             #     spatial_scale=1.0, aligned=True
    #                             # )
                                
    #                             # Extract features using RoI po
    #                             roi_features = ops.roi_pool(
    #                                 x, roi_boxes, output_size=(224, 224),
    #                                 spatial_scale=1.0
    #                             )

    #                             roi_features = roi_features.half().contiguous()
    #                             num_rois = roi_features.shape[0]

    #                             outputs_all = []

    #                             # Process ROIs in batches if needed
    #                             if num_rois <= self.batch_size:
    #                                 outputs = self.net2.infer(roi_features)
    #                                 outputs_all.append(outputs)
    #                             else:
    #                                 # Split into smaller batches and process sequentially
    #                                 for i in range(0, num_rois, self.batch_size):
    #                                     chunk = roi_features[i:i+self.batch_size]
    #                                     output_chunk = self.net2.infer(chunk)
    #                                     outputs_all.append(output_chunk)
                                        
    #                             # Concatenate all outputs
    #                             outputs = torch.cat(outputs_all, dim=0)
    #                             outputs = outputs.cpu().numpy()

    #                             # Check for NaN values in outputs
    #                             if np.isnan(outputs).any():
    #                                 # print("[Warning] NaN detected in net2 outputs. Skipping similarity matching.")
    #                                 continue

    #                             # Prepare reference data for similarity matching
    #                             recognizeDataVector_array = np.array(self.recognizeDataVector)  # shape (N, D)
                                
    #                             # Perform similarity-based classification
    #                             final_class_ids = SimilarityMethod(final_class_ids, class_0_indices, recognizeDataVector_array, outputs)
                            
    #                     # Postprocess all objects to get final detections and masks
    #                     detected_objects, polygons = postprocess_torch_cpu(
    #                                 boxes_cxcywh=boxes_cxcywh_nms,
    #                                 scores=b_scores_nms,
    #                                 class_ids=final_class_ids,
    #                                 masks=masks,
    #                                 img_w=current_frame.shape[1],
    #                                 img_h=current_frame.shape[0],
    #                                 input_shape=(640, 640),
    #                                 ratio=frame_ratio,
    #                                 dwdh=frame_dwdh)
                                    
    #         # Apply visualization if enabled
    #         if self.boxPlot:           
    #             # Draw bounding boxes on frame
    #             current_frame = draw_bboxes(
    #                 current_frame, 
    #                 detected_objects, 
    #                 class_names=self.CLASS_NAMES)
            
    #         if self.blurPlot:
    #             # Apply conditional blurring based on object classes
    #             current_frame = draw_masks_conditional_blur(
    #                 current_frame, 
    #                 detected_objects, 
    #                 polygons, 
    #                 final_class_ids,
    #                 downscale_factor=20, 
    #                 no_blur_classes=self.CLASSES_NO_BLUR)
                    
    #         # Update frame data and mark as processed
    #         self._instance_list_[b].frame_data = current_frame
    #         self._instance_list_[b].processed = True
    #         self.mooc_processed_frames = self._instance_list_[b].frame_id
        
    #     # Calculate processing time and update AI FPS
    #     processing_time = time.time() - start_time
    #     self._update_ai_fps(processing_time)
    
    
    def inference(self, processed_batch):
        """
        Optimized inference method that minimizes GPU-CPU transfers while using existing blur functions.
        
        Key optimizations:
        1. Keep masks on GPU until final processing
        2. Batch convert masks to CPU only when needed for blur
        3. Process all objects at once instead of individual transfers
        
        Args:
            processed_batch (list): List of preprocessed frame data
        """
        start_time = time.time()
        
        # Prepare input tensors for inference
        origin_imno255 = np.concatenate([item[4] for item in processed_batch], axis=0)
        batch_tensor = np.concatenate([item[1] for item in processed_batch], axis=0)
        x = torch.from_numpy(origin_imno255).to(device='cuda', dtype=torch.float16)
        
        # Perform TensorRT inference on YOLO model
        net1.cuda_ctx.push()
        results = net1.infer(batch_tensor)
        net1.cuda_ctx.pop()
        
        # Extract prediction tensors
        tensor_pred0 = copy_trt_output_to_torch_tensor(results[0])  # [B, C, N] - detection outputs
        tensor_pred1_2 = copy_trt_output_to_torch_tensor(results[1])  # [B, 32, 160, 160] - mask prototypes
        
        # Process YOLO predictions to get detections
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
        
        # Process each frame in the batch
        for b in range(min(self.batch_size, len(processed_batch))):
            # Extract frame data and preprocessing parameters
            current_frame, _, frame_ratio, frame_dwdh, _ = processed_batch[b]
            
            # Initialize default values for this frame
            detected_objects = []
            polygons = []
            final_class_ids = torch.tensor([])
            
            # Variables to store GPU data for optimized blur processing
            masks_gpu = None
            boxes_gpu = None
            final_class_ids_gpu = None
            
            # Get detections for current batch item
            if len(start_idx) > b and start_idx[b] < end_idx[b]:
                # Extract detections for this frame
                b_boxes = boxes_kept[start_idx[b]:end_idx[b]]
                b_scores = scores_kept[start_idx[b]:end_idx[b]]
                b_classes = class_ids_kept[start_idx[b]:end_idx[b]]
                b_masks = mask_coeffs_kept[start_idx[b]:end_idx[b]]
                
                # Check if any detections exist
                if b_boxes.numel() > 0:
                    # Apply Non-Maximum Suppression (NMS) to remove overlapping detections
                    nms_indices = ops.batched_nms(
                        b_boxes, b_scores, b_classes, self.iou_threshold
                    )

                    # Get NMS-filtered detections
                    b_boxes_nms = b_boxes[nms_indices]
                    b_scores_nms = b_scores[nms_indices]
                    b_classes_nms = b_classes[nms_indices]
                    b_masks_nms = b_masks[nms_indices]
                    
                    # Check if any detections remain after NMS
                    if b_boxes_nms.numel() > 0:
                        # Separate objects by class_id for different processing
                        class_0_mask = (b_classes_nms == 0)  # Objects that need feature extraction
                        non_class_0_mask = (b_classes_nms != 0)  # Objects with direct classification
                    
                        # Generate segmentation masks for all objects (KEEP ON GPU)
                        protos = tensor_pred1_2[b]  # [32, 160, 160] - mask prototypes
                        protos_reshaped = protos.view(32, -1)  # [32, 25600] - flattened prototypes
                        masks_gpu = torch.matmul(b_masks_nms, protos_reshaped).sigmoid()  # Generate masks
                        masks_gpu = masks_gpu.view(-1, 160, 160)  # Reshape to [N, 160, 160] - STAY ON GPU
                        
                        # Store GPU data for optimized blur processing
                        boxes_gpu = b_boxes_nms.clone()  # Keep boxes on GPU
                        final_class_ids_gpu = b_classes_nms.clone()  # Keep class IDs on GPU

                        # Convert bounding boxes from xyxy to cxcywh format for postprocessing
                        x1, y1, x2, y2 = b_boxes_nms.unbind(dim=1)
                        cx = (x1 + x2) / 2  # Center x
                        cy = (y1 + y2) / 2  # Center y
                        w = x2 - x1  # Width
                        h = y2 - y1  # Height
                        boxes_cxcywh_nms = torch.stack([cx, cy, w, h], dim=1)

                        # Initialize final class predictions
                        final_class_ids = b_classes_nms.clone()
                        
                        # Feature extraction mode for class 0 objects
                        if self.FEmodel:
                            class_0_indices = torch.nonzero(class_0_mask, as_tuple=False).squeeze(1)
        
                            # Process class_id == 0 objects with second model for verification
                            if class_0_mask.any():
                                class_0_boxes = b_boxes_nms[class_0_mask]
                                
                                # RoI (Region of Interest) alignment for feature extraction
                                batch_index = torch.full((class_0_boxes.shape[0], 1), b, 
                                                    dtype=torch.float16, device=b_boxes.device)
                                
                                roi_boxes = torch.cat([batch_index, class_0_boxes], dim=1).half()
                                
                                # Extract features using RoI pooling
                                roi_features = ops.roi_pool(
                                    x, roi_boxes, output_size=(224, 224),
                                    spatial_scale=1.0
                                )

                                roi_features = roi_features.half().contiguous()
                                num_rois = roi_features.shape[0]

                                outputs_all = []

                                # Process ROIs in batches if needed
                                if num_rois <= self.batch_size:
                                    outputs = self.net2.infer(roi_features)
                                    outputs_all.append(outputs)
                                else:
                                    # Split into smaller batches and process sequentially
                                    for i in range(0, num_rois, self.batch_size):
                                        chunk = roi_features[i:i+self.batch_size]
                                        output_chunk = self.net2.infer(chunk)
                                        outputs_all.append(output_chunk)
                                        
                                # Concatenate all outputs
                                outputs = torch.cat(outputs_all, dim=0)
                                outputs = outputs.cpu().numpy()

                                # Check for NaN values in outputs
                                if np.isnan(outputs).any():
                                    # print("[Warning] NaN detected in net2 outputs. Skipping similarity matching.")
                                    continue

                                # Prepare reference data for similarity matching
                                recognizeDataVector_array = np.array(self.recognizeDataVector)  # shape (N, D)
                                
                                # Perform similarity-based classification
                                final_class_ids = SimilarityMethod(final_class_ids, class_0_indices, recognizeDataVector_array, outputs)
                                
                                # Update GPU class IDs after similarity matching
                                final_class_ids_gpu = final_class_ids.clone()
                            
                        # ===================================================================
                        # OPTIMIZED BLUR PROCESSING - SAME QUALITY AS ORIGINAL
                        # ===================================================================
                        if self.blurPlot:
                            # Apply blur using the exact same logic as original draw_masks_conditional_blur
                            current_frame = self._apply_optimized_blur(
                                current_frame, 
                                masks_gpu, 
                                boxes_gpu, 
                                final_class_ids_gpu,
                                frame_ratio, 
                                frame_dwdh
                            )
                        
                        # Only do postprocessing for box drawing if needed (and blur hasn't already done it)
                        if self.boxPlot and not self.blurPlot:
                            # Postprocess objects to get final detections and polygons for box drawing
                            detected_objects, polygons = postprocess_torch_cpu(
                                        boxes_cxcywh=boxes_cxcywh_nms,
                                        scores=b_scores_nms,
                                        class_ids=final_class_ids,
                                        masks=masks_gpu,  # This will be converted to CPU in postprocess_torch_cpu
                                        img_w=current_frame.shape[1],
                                        img_h=current_frame.shape[0],
                                        input_shape=(640, 640),
                                        ratio=frame_ratio,
                                        dwdh=frame_dwdh)
                                        
            # Apply bounding box visualization if enabled
            if self.boxPlot:           
                # Get detected_objects if not already processed during blur
                if not self.blurPlot and not detected_objects:
                    # Need to do postprocessing for box drawing
                    if masks_gpu is not None:
                        detected_objects, polygons = postprocess_torch_cpu(
                            boxes_cxcywh=boxes_cxcywh_nms,
                            scores=b_scores_nms,
                            class_ids=final_class_ids,
                            masks=masks_gpu,
                            img_w=current_frame.shape[1],
                            img_h=current_frame.shape[0],
                            input_shape=(640, 640),
                            ratio=frame_ratio,
                            dwdh=frame_dwdh)
                
                if detected_objects:
                    # Draw bounding boxes on frame
                    current_frame = draw_bboxes(
                        current_frame, 
                        detected_objects, 
                        class_names=self.CLASS_NAMES)
                    
            # Update frame data and mark as processed
            self._instance_list_[b].frame_data = current_frame
            self._instance_list_[b].processed = True
            self.mooc_processed_frames = self._instance_list_[b].frame_id
        
        # Calculate processing time and update AI FPS
        processing_time = time.time() - start_time
        self._update_ai_fps(processing_time)

    def _apply_optimized_blur(self, frame, masks_gpu, boxes_gpu, class_ids_gpu, frame_ratio, frame_dwdh):
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
            should_blur = current_class_id not in self.CLASSES_NO_BLUR
            
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

    def _scale_boxes_to_original(self, boxes_gpu, frame, frame_ratio, frame_dwdh):
        """
        Scale boxes from model input coordinates back to original image coordinates.
        
        Args:
            boxes_gpu: torch.Tensor (N, 4) on GPU - boxes in xyxy format
            frame: numpy array - original frame for boundary checking
            frame_ratio: float - scaling ratio
            frame_dwdh: tuple - padding offsets (dw, dh)
        
        Returns:
            scaled_boxes: torch.Tensor (N, 4) on GPU - scaled boxes
        """
        dw, dh = frame_dwdh
        
        # Clone to avoid modifying original
        scaled_boxes = boxes_gpu.clone()
        
        # Remove padding and scale back
        scaled_boxes[:, [0, 2]] -= dw  # x coordinates
        scaled_boxes[:, [1, 3]] -= dh  # y coordinates
        scaled_boxes /= frame_ratio
        
        # Clamp to image boundaries
        scaled_boxes[:, [0, 2]] = scaled_boxes[:, [0, 2]].clamp(0, frame.shape[1])  # width
        scaled_boxes[:, [1, 3]] = scaled_boxes[:, [1, 3]].clamp(0, frame.shape[0])  # height
        
        return scaled_boxes

    def get_skip_frame_info(self):
        """
        Get information about current frame skipping strategy.
        
        Returns:
            dict: Information about FPS ratios and skipping strategy
        """
        if self._ai_fps_ > 0:
            n_skip = max(0, int((self._instream_fps_ / self._ai_fps_) - 1))
            return {
                'input_fps': self._instream_fps_,
                'ai_fps': self._ai_fps_,
                'skip_frames': n_skip,
                'ratio': self._instream_fps_ / self._ai_fps_,
                'strategy': f"Process 1 frame, skip {n_skip} frames" if n_skip > 0 else "Process all frames"
            }
        return {
            'input_fps': self._instream_fps_,
            'ai_fps': self._ai_fps_,
            'skip_frames': 0,
            'ratio': 0,
            'strategy': "No FPS data available"
        }

    def update_input_fps(self, input_fps):
        """
        Update input FPS from stream controller.
        
        Args:
            input_fps (float): Current input stream FPS
        """
        self._instream_fps_ = input_fps

    @classmethod
    def get_instance(cls, cfg=None, **kwargs):
        if cls._instance is None:
            if cfg is None:
                raise ValueError("AI instance not created yet. Provide `cfg` to initialize.")
            cls._instance = cls(cfg, **kwargs)
        return cls._instance
