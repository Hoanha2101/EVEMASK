"""
This script benchmarks two main scenarios:
1. AI Inference Only Benchmark:
   - Measures the raw inference speed of the AI module on synthetic data.
   - Does not include data capture or output stages.

2. Full Pipeline Benchmark:
   - Simulates the real-world pipeline including:
     a. Frame capture (from video/camera)
     b. AI inference (object detection, segmentation, etc.)
     c. Output processing (e.g., writing results)
   - Measures end-to-end throughput (AI FPS) for different batch sizes.

Pipeline Flow:
    [Frame Capture] --> [AI Inference] --> [Output/Write]
         |                 |                  |
      (Thread)          (Thread)           (Thread)

For each batch size, the script:
- Initializes all pipeline components and threads.
- Processes a fixed number of frames.
- Reports the average AI FPS for that configuration.

This helps evaluate both the raw AI speed and the practical throughput of the full system.

Author: EVEMASK Team
Version: 1.0.0
"""
import psutil
import GPUtil
import platform
from datetime import datetime
import sys
import os
import numpy as np
import time
import yaml
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.brain.AI import AI
from src.controllers.frame import Frame
from src.models.initNet import net1
from src.controllers import CircleQueue, StreamController
from src.logger import EveMaskLogger

config_path = os.path.join(os.path.dirname(__file__), "..", "cfg", "default.yaml")
with open(os.path.abspath(config_path), "r") as f:
    cfg = yaml.safe_load(f)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

def get_cpu_info():
    print("===== CPU Info =====")
    print(f"Logical CPUs   : {psutil.cpu_count(logical=True)}")
    print(f"Physical CPUs  : {psutil.cpu_count(logical=False)}")
    print(f"CPU Usage (%)  : {psutil.cpu_percent(interval=1)}%")
    print(f"CPU Frequency  : {psutil.cpu_freq().current:.2f} MHz")
    print()

def get_memory_info():
    print("===== Memory Info =====")
    virtual_mem = psutil.virtual_memory()
    print(f"Total RAM      : {virtual_mem.total / (1024**3):.2f} GB")
    print(f"Available RAM  : {virtual_mem.available / (1024**3):.2f} GB")
    print(f"Used RAM       : {virtual_mem.used / (1024**3):.2f} GB")
    print(f"Memory Usage   : {virtual_mem.percent}%")
    print()

def get_gpu_info():
    print("===== GPU Info =====")
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPU found.")
    for gpu in gpus:
        print(f"Name           : {gpu.name}")
        print(f"ID             : {gpu.id}")
        print(f"Memory Total   : {gpu.memoryTotal} MB")
        print(f"Memory Used    : {gpu.memoryUsed} MB")
        print(f"Memory Free    : {gpu.memoryFree} MB")
        print(f"GPU Load       : {gpu.load * 100:.1f}%")
        print(f"Temperature    : {gpu.temperature} °C")
        print()

def show_system_info():
    print("===== System Info =====")
    print(f"Platform       : {platform.system()} {platform.release()}")
    print(f"Processor      : {platform.processor()}")
    print(f"Python Version : {platform.python_version()}")
    print(f"Time           : {datetime.now()}")
    print()

def AI_Inference_Only_Benchmark(times_avg = 100, warm_up_times = 10):
    results = []
    ai_instance = AI.get_instance(cfg=cfg, FEmodel=True)
    MAX_BATCH_SIZE = cfg['MAX_BATCH_SIZE']
    for max_batch_size in range(1, MAX_BATCH_SIZE + 1):
        processed_batch = []
        ai_instance._instance_list_ = []
        for i in range(max_batch_size):
            original_frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
            img_tensor = np.random.rand(1, 3, 640, 640).astype(np.float16)
            ratio = 0.3333333333333333
            dwdh = (0.0, 140.0)
            img_no_255 = np.random.rand(1, 3, 640, 640).astype(np.float16)
            ai_instance._instance_list_.append(Frame(i, original_frame))
            processed_batch.append((original_frame, img_tensor, ratio, dwdh, img_no_255))
        # WARM UP
        for _ in range(warm_up_times):
            ai_instance.inference(processed_batch = processed_batch)
        start_time = time.time()
        for _ in range(times_avg):
            ai_instance.inference(processed_batch = processed_batch) 
        end_time = time.time()
        ai_fps = round(1/((end_time - start_time) / times_avg), 4)
        print(f"Batch size: {max_batch_size} - AI FPS: {ai_fps}")
        results.append((max_batch_size, ai_fps))
    return results

def AI_Inference_Pipeline_Benchmark(times_avg=10, warm_up_times=2):
    results = []
    MAX_BATCH_SIZE = cfg['MAX_BATCH_SIZE']
    cfg["INPUT_SOURCE"] = "videos/1.mp4"

    for max_batch_size in range(1, MAX_BATCH_SIZE + 1):
        # Set configuration for current batch size and warm-up delay
        cfg["DELAY_TIME"] = warm_up_times
        cfg["batch_size"] = max_batch_size

        # Initialize logger, stream controller, and AI engine
        logger = EveMaskLogger.get_instance()
        streamController = StreamController(cfg)
        useAI = AI(cfg, FEmodel=True)
        
        # Start the video source (e.g., ffmpeg or camera)
        streamController._start_ffmpeg()

        # Create and start threads for each pipeline stage
        capture_thread = threading.Thread(target=streamController.source_capture, name="CaptureThread", daemon=True)
        ai_thread = threading.Thread(target=useAI.run, name="AIThread", daemon=True)
        output_thread = threading.Thread(target=streamController.out_stream, name="OutputThread", daemon=True)

        capture_thread.start()   # Thread for capturing frames from the source
        ai_thread.start()        # Thread for running AI inference on frames
        logger.waiting_bar(cfg)  # Optional: show progress bar or wait for warm-up
        output_thread.start()    # Thread for outputting processed frames
        # Record the starting frame index to measure progress
        anchor_count = streamController._write_frame_index + times_avg

        start_time = time.time()
        while True:
            time.sleep(0.01)
            if streamController._write_frame_index > anchor_count:
                break
        end_time = time.time()
        # Calculate and print the average AI FPS for this batch size
        ai_fps_now = times_avg / (end_time - start_time) if (end_time - start_time) > 0 else 0
        print(f"Batch size: {max_batch_size} - AI FPS: {ai_fps_now:.4f}")
        print("--------------------------------")
        results.append((max_batch_size, round(ai_fps_now, 4)))
    return results

if __name__ == "__main__":
    # Run the AI inference-only benchmark (single module, no pipeline)
    only_results = AI_Inference_Only_Benchmark()
    # Run the full pipeline benchmark (capture, AI, output)
    pipeline_results = AI_Inference_Pipeline_Benchmark()
    

    # ========================================================================
    # CLEAR SCREEN
    # ========================================================================
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # ========================================================================
    # LOGGER INITIALIZATION
    # ========================================================================
    logger = EveMaskLogger.get_instance()

    # ========================================================================
    # DISPLAY LOGO
    # ========================================================================
    logger.display_logo()

    # Print system and hardware information for benchmarking context
    show_system_info()
    get_cpu_info()
    get_memory_info()
    get_gpu_info()

    print("\n===== BENCHMARK SUMMARY =====")
    headers = ["Batch Size", "AI Inference Only FPS", "Full Pipeline FPS"]
    table = []
    for i in range(len(only_results)):
        batch_size = only_results[i][0]
        only_fps = only_results[i][1]
        pipeline_fps = pipeline_results[i][1] if i < len(pipeline_results) else "-"
        table.append([batch_size, only_fps, pipeline_fps])
    if tabulate:
        print(tabulate(table, headers=headers, tablefmt="grid"))
    else:
        # Fallback: print as plain text
        print(headers)
        for row in table:
            print(row)