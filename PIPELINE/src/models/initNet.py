"""
Neural network initialization module for the EVEMASK Pipeline system.
Handles loading and configuration of TensorRT models for segmentation and feature extraction.
Manages model paths, input/output names, and TensorRT engine initialization.

Author: EVEMASK Team
Version: 1.0.0
"""

import yaml
from ..models import TensorrtBase, TensorrtBase_M2
import os

# ========================================================================
# CONFIGURATION LOADING
# ========================================================================
# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cfg', 'default.yaml')
with open(os.path.abspath(config_path), "r") as f:
    cfg = yaml.safe_load(f)

# ========================================================================
# SEGMENTATION MODEL INITIALIZATION
# ========================================================================
# Load segmentation model configuration
seg_cfg = cfg["segment_model"]
SegmentModel_path = seg_cfg["path"]
input_name = seg_cfg["input_names"]
all_outputs = seg_cfg["all_output_names"]

# Determine output names based on model type (trimmed vs full)
if "trimmed" in SegmentModel_path:
    # Use simplified output names for trimmed model
    output_names = ["pred0", "pred1_2"]
else:
    # Use all output names for full model
    output_names = all_outputs

# Initialize segmentation model with TensorRT engine
net1 = TensorrtBase(
    engine_file_path=SegmentModel_path,
    input_names=input_name,
    output_names=output_names,
    max_batch_size=seg_cfg["max_batch_size"],
    dynamic_factor=seg_cfg["dynamic_factor"],
    getTo=seg_cfg["get_to"]
)

# ========================================================================
# FEATURE EXTRACTION MODEL INITIALIZATION
# ========================================================================
# Load feature extraction model configuration
ext_cfg = cfg["extract_model"]
ExtractModel_path = ext_cfg["path"]
input_name_M2 = ext_cfg["input_names"]
output_names_M2 = ext_cfg["output_names"]

# Initialize feature extraction model with TensorRT engine
net2 = TensorrtBase_M2(
    ExtractModel_path,
    input_names=input_name_M2,
    output_names=output_names_M2,
    max_batch_size=ext_cfg["max_batch_size"],
    lenEmb=ext_cfg["len_emb"]
)
