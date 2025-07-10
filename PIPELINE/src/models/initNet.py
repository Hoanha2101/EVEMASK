import yaml
from ..models import TensorrtBase, TensorrtBase_M2
import os

# Load config
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cfg', 'default.yaml')
with open(os.path.abspath(config_path), "r") as f:
    cfg = yaml.safe_load(f)

# Load Segment Model config
seg_cfg = cfg["segment_model"]
SegmentModel_path = seg_cfg["path"]
input_name = seg_cfg["input_names"]
all_outputs = seg_cfg["all_output_names"]

# Determine output names based on filename
if "trimmed" in SegmentModel_path:
    output_names = ["pred0", "pred1_2"]
else:
    output_names = all_outputs

# Load Segment Model (TensorrtBase)
net1 = TensorrtBase(
    engine_file_path=SegmentModel_path,
    input_names=input_name,
    output_names=output_names,
    max_batch_size=seg_cfg["max_batch_size"],
    dynamic_factor=seg_cfg["dynamic_factor"],
    getTo=seg_cfg["get_to"]
)

# Load Extract Model config
ext_cfg = cfg["extract_model"]
ExtractModel_path = ext_cfg["path"]
input_name_M2 = ext_cfg["input_names"]
output_names_M2 = ext_cfg["output_names"]

# Load Extract Model (TensorrtBase_M2)
net2 = TensorrtBase_M2(
    ExtractModel_path,
    input_names=input_name_M2,
    output_names=output_names_M2,
    max_batch_size=ext_cfg["max_batch_size"],
    lenEmb=ext_cfg["len_emb"]
)
