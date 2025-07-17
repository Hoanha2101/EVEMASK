# EVEMASK Pipeline v1.0.0

<div align="center">

![EVEMASK Logo](https://img.shields.io/badge/EVEMASK-Pipeline%20v1.0.0-blue?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/Python-3.9%2B-green?style=for-the-badge&logo=python)
![TensorRT](https://img.shields.io/badge/TensorRT-8.6.1-orange?style=for-the-badge&logo=nvidia)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-red?style=for-the-badge&logo=ultralytics)

**Real-time AI Video Processing Pipeline with Object Detection, Segmentation, and Classification**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Configuration](#configuration) • [Team](#team)

</div>

---

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Team](#team)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 About

**EVEMASK Pipeline v1.0.0** is a high-performance, real-time video processing system designed for intelligent content analysis and automated content moderation. The system combines state-of-the-art AI models to detect, segment, and classify objects in video streams with exceptional accuracy and speed.

### Key Capabilities

- **Real-time Object Detection & Segmentation**: YOLOv8-based segmentation for precise object boundaries
- **Intelligent Classification**: Feature extraction and similarity matching for object classification
- **Content Moderation**: Conditional blurring based on object classes
- **High Performance**: TensorRT optimization for GPU-accelerated inference
- **Multi-stream Support**: Handle multiple input/output streams simultaneously
- **Dynamic Processing**: Adaptive frame skipping and batch processing

### Use Cases

- **Content Moderation**: Automated detection and blurring of sensitive content
- **Brand Recognition**: Identify and classify brand logos in video streams
- **Quality Control**: Monitor video content for compliance and standards
- **Real-time Analytics**: Extract insights from live video feeds

---

## ✨ Features

### 🚀 Performance Optimizations
- **TensorRT Engine**: GPU-accelerated inference with FP16 precision
- **Dynamic Batching**: Adaptive batch sizes for optimal throughput
- **Circular Queue**: Efficient frame buffering with automatic memory management
- **Multi-threading**: Parallel processing of capture, AI, and output streams

### 🎯 AI Capabilities
- **YOLOv8 Segmentation**: State-of-the-art object detection and segmentation
- **Feature Extraction**: VGG16-based embedding generation for classification
- **Similarity Matching**: FAISS-based nearest neighbor search
- **Conditional Blurring**: Intelligent content masking based on object classes

### 📡 Streaming Support
- **Multiple Protocols**: UDP, RTSP, RTMP input/output support
- **Real-time Processing**: Sub-second latency for live streams
- **FPS Control**: Configurable target frame rates
- **Stream Validation**: Built-in connectivity testing tools

### 🔧 System Monitoring
- **Resource Tracking**: CPU, GPU, memory, and network monitoring
- **Performance Metrics**: FPS, latency, and throughput measurement
- **Health Monitoring**: Automatic system status reporting
- **Logging**: Comprehensive logging with timestamps

---

## 👥 Team

**EVEMASK Team** - AI Research & Development Group

### Project Information
- **Project Name**: EVEMASK Pipeline v1.0.0
- **Version**: 1.0.0
- **Development Period**: 2024
- **Technology Stack**: Python, PyTorch, TensorRT, OpenCV, YOLOv8

### Contact
- **Email**: [team@evemask.com](mailto:team@evemask.com)
- **Repository**: [EVEMASK/PIPELINE](https://github.com/EVEMASK/PIPELINE)

---

## 🛠 Installation

### Prerequisites

- **Python**: 3.9 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **TensorRT**: 8.6.1
- **NVIDIA GPU**: Compatible with TensorRT 8.6.1

### System Requirements

- **OS**: Windows 10/11, Linux (Ubuntu 20.04+)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **Storage**: 10GB free space for models and data

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/EVEMASK/PIPELINE.git
cd PIPELINE
```

2. **Create virtual environment**
```bash
python -m venv evemask_env
source evemask_env/bin/activate  # Linux/Mac
# or
evemask_env\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python test_stream.py
```

---

## 🚀 Quick Start

### 1. Model Preparation

Convert your YOLOv8 models to TensorRT format:

```bash
# Extract PyTorch model from Ultralytics format
python getpytorch.py --weights weights/yolo/yolov8_seg_aug_best_l.pt --output weights/pytorch/yolov8_seg_aug_best_l.pth

# Export to ONNX format
python export.py --pth weights/pytorch/yolov8_seg_aug_best_l.pth --output weights/onnx/yolov8_seg_aug_best_l.onnx --input-shape 1 3 640 640 --input-name input --output-names pred0 pred1_0_0 pred1_0_1 pred1_0_2 pred1_1 pred1_2 --mode float32bit --device cuda --opset 19 --typeModel seg

# Build TensorRT engine
python build.py --onnx weights/onnx/yolov8_seg_aug_best_l.onnx --engine weights/trtPlans/yolov8_seg_aug_best_l.trt --fp16 --dynamic --dynamic-shapes "{\"input\": ((1, 3, 640, 640), (2, 3, 640, 640), (3, 3, 640, 640))}"
```

### 2. Configuration

Edit `cfg/default.yaml` to configure your input/output streams and model parameters.

### 3. Run the Pipeline

```bash
python main.py
```

---

## 📖 Usage

### Basic Usage

```bash
# Start the main pipeline
python main.py

# Test stream connectivity
python test_stream.py

# Monitor system resources
python monitor.py
```

### Advanced Configuration

#### Input Sources
- **UDP Stream**: `udp://224.1.1.1:30122?pkt_size=1316`
- **RTSP Stream**: `rtsp://username:password@ip:port/stream`
- **Local File**: `videos/test.mp4`
- **Webcam**: `0` (default camera)

#### Output Destinations
- **UDP Stream**: `udp://@225.1.9.254:30133?pkt_size=1316`
- **RTMP Stream**: `rtmp://server/live/stream`
- **Local File**: `output.mp4`

### Model Configuration

The system supports two main models:

1. **Segmentation Model** (YOLOv8)
   - Object detection and segmentation
   - Configurable confidence and IoU thresholds
   - Dynamic batch processing

2. **Feature Extraction Model** (VGG16)
   - Embedding generation for classification
   - Similarity-based matching
   - Reference data preparation

### Performance Tuning

- **Batch Size**: Adjust `batch_size` in config for optimal throughput
- **FPS Control**: Set `TARGET_FPS` for desired output frame rate
- **GPU Memory**: Configure `max_batch_size` based on available VRAM
- **Precision**: Use FP16 for faster inference with minimal accuracy loss

---

## 🏗 Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Stream  │───▶│  Capture Thread │───▶│  Circular Queue │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Output Stream  │◀───│  Output Thread  │◀───│   AI Thread     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                └───────────────────────┘
                                        │
                                ┌─────────────────┐
                                │  TensorRT       │
                                │  Inference      │
                                └─────────────────┘
```

### Core Components

#### 1. Stream Controller (`src/controllers/stream.py`)
- Manages video capture and output streaming
- Handles multiple input/output protocols
- Implements frame buffering and synchronization

#### 2. AI Engine (`src/brain/AI.py`)
- Coordinates YOLO segmentation and feature extraction
- Implements batch processing and frame skipping
- Manages conditional blurring and object classification

#### 3. Circular Queue (`src/controllers/circle_queue.py`)
- Thread-safe frame buffer with automatic overflow management
- Efficient frame retrieval with skipping capabilities
- Memory management for real-time processing

#### 4. TensorRT Models (`src/models/engine.py`)
- High-performance GPU inference engines
- Dynamic shape support for flexible batch sizes
- Memory optimization and workspace management

### Processing Pipeline

1. **Frame Capture**: Continuous frame reading from input source
2. **Preprocessing**: Image resizing, normalization, and tensor conversion
3. **Segmentation**: YOLOv8-based object detection and mask generation
4. **Feature Extraction**: VGG16-based embedding generation for detected objects
5. **Classification**: Similarity matching against reference data
6. **Post-processing**: Conditional blurring and mask application
7. **Output Streaming**: Processed frames sent to output destination

---

## ⚙️ Configuration

### Main Configuration (`cfg/default.yaml`)

```yaml
# Input/Output Configuration
INPUT_SOURCE: "udp://224.1.1.1:30122?pkt_size=1316"
OUTPUT_TYPE: "udp"
OUTPUT_STREAM_URL_UDP: "udp://@225.1.9.254:30133?pkt_size=1316"
TARGET_FPS: 25

# Processing Parameters
batch_size: 3
conf_threshold: 0.5
iou_threshold: 0.7
DELAY_TIME: 2

# Model Paths
segment_model:
  path: "weights/trtPlans/yolov8_seg_aug_best_l_trimmed.trt"
  max_batch_size: 3
  dynamic_factor: 3

extract_model:
  path: "weights/trtPlans/SupConLoss_BBVGG16.trt"
  max_batch_size: 32
  len_emb: 256

# Classification Classes
names:
  0: "unbet"
  1: "betrivers"
  2: "fanduel"
  # ... additional classes

# Blur Configuration
CLASSES_NO_BLUR: [0]  # Classes that should not be blurred
```

### Environment Variables

```bash
# CUDA Configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# TensorRT Configuration
export TENSORRT_ROOT=/usr/local/tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
```

---

## 📁 Project Structure

```
PIPELINE/
├── 📄 main.py                 # Main entry point
├── 📄 monitor.py              # System monitoring
├── 📄 build.py                # TensorRT engine builder
├── 📄 export.py               # PyTorch to ONNX exporter
├── 📄 getpytorch.py           # YOLOv8 model extractor
├── 📄 test_stream.py          # Stream testing utility
├── 📄 requirements.txt        # Python dependencies
├── 📄 README.md              # This file
│
├── 📁 cfg/
│   └── 📄 default.yaml       # Main configuration file
│
├── 📁 src/
│   ├── 📁 brain/
│   │   └── 📄 AI.py          # Main AI processing engine
│   │
│   ├── 📁 controllers/
│   │   ├── 📄 stream.py      # Stream management
│   │   ├── 📄 frame.py       # Frame processing
│   │   └── 📄 circle_queue.py # Circular buffer
│   │
│   ├── 📁 logger/
│   │   ├── 📄 logger.py      # EveMaskLogger
│   │   └── 📄 initNet.py     
│   │
│   ├── 📁 models/
│   │   ├── 📄 engine.py      # TensorRT inference
│   │   └── 📄 initNet.py     # Model initialization
│   │
│   └── 📁 tools/
│       ├── 📄 utils.py       # Utility functions
│       ├── 📄 vectorPrepare.py # Feature preparation
│       ├── 📄 similarityBlock.py # Similarity matching
│       └── 📄 NB_search.py   # Neighbor search
│
├── 📁 weights/
│   ├── 📁 yolo/              # YOLOv8 weights
│   ├── 📁 pytorch/           # PyTorch models
│   ├── 📁 onnx/              # ONNX models
│   └── 📁 trtPlans/          # TensorRT engines
│
├── 📁 recognizeData/         # Reference data for classification
├── 📁 videos/                # Test video files
└── 📁 block/                 # Additional resources
```

---

## 📊 Performance

### Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Inference Speed** | 25 FPS | Real-time processing |
| **Latency** | <40ms | End-to-end processing |
| **GPU Memory** | 4-6GB | TensorRT optimized |
| **CPU Usage** | 15-25% | Multi-threaded |
| **Accuracy** | 95%+ | YOLOv8 segmentation |

### Optimization Features

- **TensorRT FP16**: 2x speed improvement with minimal accuracy loss
- **Dynamic Batching**: Adaptive batch sizes for optimal throughput
- **Frame Skipping**: Intelligent frame selection for real-time processing
- **Memory Management**: Automatic cleanup and efficient buffer usage

### Scaling Considerations

- **Multi-GPU**: Support for multiple GPU configurations
- **Distributed Processing**: Horizontal scaling across multiple nodes
- **Load Balancing**: Automatic workload distribution
- **Failover**: Graceful handling of hardware failures

---

## 🤝 Contributing

We welcome contributions from the community! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Code Style

- Follow PEP 8 Python style guidelines
- Add comprehensive docstrings
- Include type hints where appropriate
- Write unit tests for new features

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_ai.py

# Run with coverage
python -m pytest --cov=src tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- **Ultralytics**: YOLOv8 implementation
- **NVIDIA**: TensorRT optimization framework
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

---

## 📞 Support

For support and questions:

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions on GitHub
- **Email**: Contact the EVEMASK team directly

---

<div align="center">

**Made with ❤️ by the EVEMASK Team**

[Back to Top](#evemask-pipeline-v20)

</div>