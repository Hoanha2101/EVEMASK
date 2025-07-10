########################
### Sample CLI: python build.py --onnx weights/onnx/yolov8_seg_aug_best_l.onnx --engine weights/trtPlans/yolov8_seg_aug_best_l.trt --fp16 --dynamic --dynamic-shapes "{\"input\": ((1, 3, 640, 640), (2, 3, 640, 640), (3, 3, 640, 640))}"
### Sample CLI: python build.py --onnx weights/onnx/SupConLoss_BBVGG16.onnx --engine weights/trtPlans/SupConLoss_BBVGG16.trt --fp16 --dynamic --dynamic-shapes "{\"input\": ((1,3,224,224), (8,3,224,224), (32,3,224,224))}"

#######################

import tensorrt as trt
import argparse
import os
import ast

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_trt_engine(
    onnx_file_path,
    engine_file_path,
    use_fp16=False,
    dynamic=False,
    dynamic_shapes=None,
    fixed_batch_size=1
):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)
    config.max_workspace_size = 1 << 32  # 4 GB
    if builder.platform_has_fast_fp16 and use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    print("✅ Successfully parsed ONNX file")

    if dynamic:
        print(f"===> Using dynamic shapes: {dynamic_shapes}")
        profile = builder.create_optimization_profile()
        for name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    else:
        for i in range(network.num_inputs):
            shape = list(network.get_input(i).shape)
            shape[0] = fixed_batch_size
            network.get_input(i).shape = shape
        print(f"===> Using fixed batch size: {fixed_batch_size}")

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception as e:
            print(f"⚠️ Cannot remove existing file: {engine_file_path}. Error: {e}")

    print("Creating TensorRT Engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"✅ Engine saved at: {engine_file_path}")
    else:
        print("❌ Failed to build engine")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine")

    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX file")
    parser.add_argument("--engine", type=str, required=True, help="Path to save the TensorRT engine")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mode")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shape mode")
    parser.add_argument("--dynamic-shapes", type=str, default="", help="Dynamic shape dict, e.g. '{\"input\": [(1,3,224,224),(8,3,224,224),(32,3,224,224)]}'")
    parser.add_argument("--fixed-batch-size", type=int, default=1, help="Fixed batch size if not using dynamic shapes")

    args = parser.parse_args()

    dynamic_shapes = ast.literal_eval(args.dynamic_shapes) if args.dynamic_shapes else None

    build_trt_engine(
        onnx_file_path=args.onnx,
        engine_file_path=args.engine,
        use_fp16=args.fp16,
        dynamic=args.dynamic,
        dynamic_shapes=dynamic_shapes,
        fixed_batch_size=args.fixed_batch_size
    )
