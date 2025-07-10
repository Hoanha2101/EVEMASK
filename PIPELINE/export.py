
########################
### Sample CLI: python export.py --pth weights/pytorch/yolov8_seg_aug_best_l.pth --output weights/onnx/yolov8_seg_aug_best_l.onnx --input-shape 1 3 640 640 --input-name input --output-names pred0 pred1_0_0 pred1_0_1 pred1_0_2 pred1_1 pred1_2 --mode float32bit --device cuda --opset 19 --typeModel seg
### Sample CLI: python export.py --pth weights/pytorch/SupConLoss_BBVGG16.pth --output weights/onnx/SupConLoss_BBVGG16.onnx --input-shape 1 3 224 224 --input-name input --output-names output --mode float16bit --device cuda --opset 12 --typeModel fe
#######################

import torch
import torch.nn as nn
import argparse
from torchvision import models
import os

def smart_load_model(pth_path, model_type, emb_dim=256):
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("📂 Detected checkpoint with 'model_state_dict'")
        if model_type == "fe":
            model = Network(emb_dim=emb_dim)
            model.load_state_dict(checkpoint["model_state_dict"])
            return model
        else:
            raise ValueError("Segmentation/classification model should not contain 'model_state_dict'. Please check --typeModel or retrain properly.")
    
    elif isinstance(checkpoint, dict):
        print("📂 Detected raw state_dict")
        if model_type == "fe":
            model = Network(emb_dim=emb_dim)
            model.load_state_dict(checkpoint)
            return model
        else:
            raise ValueError("Segmentation/classification model should not be a raw state_dict. Please check --typeModel.")
    
    elif isinstance(checkpoint, torch.nn.Module):
        print("📂 Detected full nn.Module object")
        return checkpoint
    
    else:
        raise RuntimeError("Unsupported model format.")

# Feature extractor model (VGG16 → embedding)
class Network(nn.Module):
    def __init__(self, emb_dim=128):
        super(Network, self).__init__()
        model_bb = models.vgg16(pretrained=True)
        self.conv = model_bb.features
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def convert_pytorch_model_to_onnx(
    model,
    path_onnx,
    input_shape=(1, 3, 224, 224),
    input_name="input",
    output_names=["output"],
    mode='float32bit',
    device='cuda',
    opset_version=12
):
    dummy_input = torch.randn(*input_shape)

    if mode == 'float16bit':
        print("Converting model and inputs to float16")
        model = model.half()
        dummy_input = dummy_input.half()
    else:
        print("Converting model and inputs to float32")
        model = model.float()
        dummy_input = dummy_input.float()

    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    print(f"Exporting ONNX model on device: {device}")
    model.to(device).eval()
    dummy_input = dummy_input.to(device)

    dynamic_axes = {input_name: {0: 'batch_size'}}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}

    torch.onnx.export(
        model,
        dummy_input,
        path_onnx,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset_version,
        verbose=False
    )

    print(f"✅ ONNX model saved to {path_onnx}")

    return path_onnx  # Trả về đường dẫn để xử lý sau

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")

    parser.add_argument("--pth", type=str, required=True, help="Path to .pth PyTorch model file")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--input-shape", type=int, nargs=4, default=(1, 3, 224, 224), help="Input shape e.g. 1 3 224 224")
    parser.add_argument("--input-name", type=str, default="input", help="Name of ONNX input tensor")
    parser.add_argument("--output-names", type=str, nargs='+', default=["output"], help="List of output names")
    parser.add_argument("--mode", type=str, choices=["float32bit", "float16bit"], default="float32bit", help="Precision mode")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to use")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--typeModel", type=str, choices=["seg", "fe"], required=True, help="Model type: seg or fe")

    args = parser.parse_args()

    if args.typeModel == "seg":
        model = smart_load_model(args.pth, model_type="seg")
    elif args.typeModel == "fe":
        model = smart_load_model(args.pth, model_type="fe", emb_dim=256)

    # Export ONNX
    onnx_path = convert_pytorch_model_to_onnx(
        model=model,
        path_onnx=args.output,
        input_shape=tuple(args.input_shape),
        input_name=args.input_name,
        output_names=args.output_names,
        mode=args.mode,
        device=args.device,
        opset_version=args.opset
    )

    # Trim ONNX nếu là segmentation model
    if args.typeModel == "seg":
        print("✂️ Trimming unused outputs from ONNX model...")
        import onnx_graphsurgeon as gs
        import onnx

        graph = gs.import_onnx(onnx.load(onnx_path))
        # Giữ lại 2 output quan trọng
        graph.outputs = [o for o in graph.outputs if o.name in ["pred0", "pred1_2"]]
        graph.cleanup().toposort()

        trimmed_path = onnx_path.replace(".onnx", "_trimmed.onnx")
        onnx.save(gs.export_onnx(graph), trimmed_path)
        print(f"✅ Exported trimmed ONNX model to {trimmed_path}")

