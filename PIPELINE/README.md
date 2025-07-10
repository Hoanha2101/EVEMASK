# Pipline v2.0
### Tối ưu hàng đợi
### Chạy với pipeline không có skip frame, nhận full frame

# Run
```bash
python getpytorch.py --weights weights/yolo/yolov8_seg_aug_best_l.pt --output weights/pytorch/yolov8_seg_aug_best_l.pth
```
```bash
python export.py --pth weights/pytorch/yolov8_seg_aug_best_l.pth --output weights/onnx/yolov8_seg_aug_best_l.onnx --input-shape 1 3 640 640 --input-name input --output-names pred0 pred1_0_0 pred1_0_1 pred1_0_2 pred1_1 pred1_2 --mode float32bit --device cuda --opset 19 --typeModel seg
```
```bash
python export.py --pth weights/pytorch/SupConLoss_BBVGG16.pth --output weights/onnx/SupConLoss_BBVGG16.onnx --input-shape 1 3 224 224 --input-name input --output-names output --mode float16bit --device cuda --opset 12 --typeModel fe
```
```bash
python build.py --onnx weights/onnx/yolov8_seg_aug_best_l.onnx --engine weights/trtPlans/yolov8_seg_aug_best_l.trt --fp16 --dynamic --dynamic-shapes "{\"input\": ((1, 3, 640, 640), (2, 3, 640, 640), (3, 3, 640, 640))}"
```
```bash
python build.py --onnx weights/onnx/SupConLoss_BBVGG16.onnx --engine weights/trtPlans/SupConLoss_BBVGG16.trt --fp16 --dynamic --dynamic-shapes "{\"input\": ((1,3,224,224), (8,3,224,224), (32,3,224,224))}"
```
```bash
python main.py
```

# Folder map
```bash
PIPELINE/
│   .gitignore
│   build.py
│   export.py
│   getpytorch.py
│   main.py
│   monitor.py
│   README.md
│   requirements.txt
│   test_stream.py
│   
├───cfg
│       default.yaml
│
├───recognizeData
│   ├───bet365
│   │       000001.png
│   │       000002.png
│   │       000003.png
│   │
│   └───betano
│           000001.png
│           000002.png
│           000003.png
│
├───src
│   │   __init__.py
│   │
│   ├───brain
│   │   │   AI.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           AI.cpython-310.pyc
│   │           AI.cpython-39.pyc
│   │           __init__.cpython-310.pyc
│   │           __init__.cpython-39.pyc
│   │
│   ├───controllers
│   │   │   circle_queue.py
│   │   │   frame.py
│   │   │   stream.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           circle_queue.cpython-39.pyc
│   │           frame.cpython-39.pyc
│   │           stream.cpython-39.pyc
│   │           __init__.cpython-39.pyc
│   │
│   ├───models
│   │   │   engine.py
│   │   │   initNet.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           engine.cpython-39.pyc
│   │           initNet.cpython-39.pyc
│   │           __init__.cpython-39.pyc
│   │
│   ├───tools
│   │   │   NB_search.py
│   │   │   similarityBlock.py
│   │   │   utils.py
│   │   │   vectorPrepare.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │           frame.cpython-39.pyc
│   │           NB_search.cpython-39.pyc
│   │           similarityBlock.cpython-39.pyc
│   │           utils.cpython-310.pyc
│   │           utils.cpython-39.pyc
│   │           vectorPrepare.cpython-39.pyc
│   │           __init__.cpython-310.pyc
│   │           __init__.cpython-39.pyc
│   │
│   └───__pycache__
│           __init__.cpython-310.pyc
│           __init__.cpython-39.pyc
│
├───videos
│       1.mp4
│       2.mp4
│       3.mp4
│       test.mp4
│
├───weights
│   │   .gitkeep
│   │
│   ├───onnx
│   │       .gitkeep
│   │       ResNet_80.onnx
│   │       SupConLoss_BBVGG16.onnx
│   │       yolov8_seg_aug_best_l.onnx
│   │       yolov8_seg_aug_best_l_16.onnx
│   │       yolov8_seg_aug_best_l_trimmed.onnx
│   │       yolov8_seg_aug_best_s.onnx
│   │       yolov8_seg_aug_best_s_trimmed.onnx
│   │
│   ├───pytorch
│   │       .gitkeep
│   │       ResNet_80.pth
│   │       SupConLoss_BBVGG16.pth
│   │       yolov8_seg_aug_best_l.pth
│   │       yolov8_seg_aug_best_s.pth
│   │
│   ├───trtPlans
│   │       .gitkeep
│   │       ResNet.trt
│   │       ResNet_80.trt
│   │       seg_final_dynamic_FP16.trt
│   │       SupConLoss_BBVGG16.trt
│   │       yolov8_seg_aug_best_l_trimmed.trt
│   │       yolov8_seg_aug_best_s_trimmed.trt
│   │
│   └───yolo
│           .gitkeep
│           seg_v1.0.pt
│           yolov8_seg_aug_best_s.pt
│
└───__pycache__
        improved_classifier.cpython-39.pyc
        initNet.cpython-39.pyc
        vectorPrepare.cpython-39.pyc
```