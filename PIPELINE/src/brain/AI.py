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
        
        if len(self._processing_times) > 0:
            # Calculate average processing time
            avg_processing_time = sum(self._processing_times) / len(self._processing_times)
            if avg_processing_time > 0:
                # Update AI FPS based on average processing time
                self._ai_fps_ = 1.0 / avg_processing_time
                self.logger.update_ai_fps(round(self._ai_fps_, 2))
            
            # Reset tracking for next calculation
            self._processing_times = []
            
    def inference(self, processed_batch):
        """
        Perform AI inference on a batch of preprocessed frames.
        
        This method handles the complete AI pipeline:
        1. YOLO object detection and segmentation
        2. Non-maximum suppression (NMS)
        3. Feature extraction for class 0 objects
        4. Similarity-based classification
        5. Post-processing and visualization
        
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
                    
                        # Generate segmentation masks for all objects
                        protos = tensor_pred1_2[b]  # [32, 160, 160] - mask prototypes
                        protos_reshaped = protos.view(32, -1)  # [32, 25600] - flattened prototypes
                        masks = torch.matmul(b_masks_nms, protos_reshaped).sigmoid()  # Generate masks
                        masks = masks.view(-1, 160, 160)  # Reshape to [N, 160, 160]
                        
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
                                
                                # Extract features using RoI alignment
                                # roi_features = ops.roi_align(
                                #     x, roi_boxes, output_size=(224, 224),
                                #     spatial_scale=1.0, aligned=True
                                # )
                                
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
                            
                        # Postprocess all objects to get final detections and masks
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
                                    
            # Apply visualization if enabled
            if self.boxPlot:           
                # Draw bounding boxes on frame
                current_frame = draw_bboxes(
                    current_frame, 
                    detected_objects, 
                    class_names=self.CLASS_NAMES)
            
            if self.blurPlot:
                # Apply conditional blurring based on object classes
                current_frame = draw_masks_conditional_blur(
                    current_frame, 
                    detected_objects, 
                    polygons, 
                    final_class_ids,
                    downscale_factor=20, 
                    no_blur_classes=self.CLASSES_NO_BLUR)
                    
            # Update frame data and mark as processed
            self._instance_list_[b].frame_data = current_frame
            self._instance_list_[b].processed = True
            self.mooc_processed_frames = self._instance_list_[b].frame_id
        
        # Calculate processing time and update AI FPS
        processing_time = time.time() - start_time
        self._update_ai_fps(processing_time)

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
