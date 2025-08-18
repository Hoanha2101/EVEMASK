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
import csv
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.brain.AI import AI
from src.controllers.frame import Frame
from src.models.initNet import net1
from src.controllers import CircleQueue, StreamController
from src.logger import EveMaskLogger
import pycuda.driver as cuda
cuda.Context.synchronize()
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

def AI_Inference_Only_Benchmark(times_avg = 20, warm_up_times = 10):
    results = []
    ai_instance = AI.get_instance(cfg=cfg, FEmodel=True)
    MAX_BATCH_SIZE = cfg['MAX_BATCH_SIZE']
    cfg["INPUT_SOURCE"] = "videos/demo1.mp4"
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
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return results

def AI_Inference_Pipeline_Benchmark(times_avg=200, warm_up_times=2):
    results = []
    MAX_BATCH_SIZE = cfg['MAX_BATCH_SIZE']
    cfg["INPUT_SOURCE"] = "videos/demo1.mp4"

    for max_batch_size in range(1, MAX_BATCH_SIZE + 1):
        FPS_LIST = []
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

        while True:
            time.sleep(0.01)
            if streamController._write_frame_index > 100:
                if logger.ai_fps > 0:
                    FPS_LIST.append(logger.ai_fps)
            if streamController._write_frame_index > anchor_count:
                break
        # Calculate and print the average AI FPS for this batch size
        ai_fps_now = sum(FPS_LIST) / len(FPS_LIST)
        print(f"Batch size: {max_batch_size} - AI FPS: {ai_fps_now:.4f}")
        print("--------------------------------")
        results.append((max_batch_size, round(ai_fps_now, 4)))

        streamController.stop()
        useAI.stop() if hasattr(useAI, 'stop') else None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    return results

def save_benchmark_results_to_csv(only_results, pipeline_results, csv_path):
    """
    Save combined benchmark results to a CSV file.
    Columns: batch_size, ai_inference_only_fps, full_pipeline_fps
    """
    # Create a mapping for quick lookup of pipeline FPS by batch size
    pipeline_map = {batch: fps for batch, fps in pipeline_results}
    fieldnames = ["batch_size", "ai_inference_only_fps", "full_pipeline_fps"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for batch_size, only_fps in only_results:
            writer.writerow({
                "batch_size": batch_size,
                "ai_inference_only_fps": only_fps,
                "full_pipeline_fps": pipeline_map.get(batch_size, "-")
            })

def visualize_benchmark_results(only_results, pipeline_results, save_path):
    """
    Visualize FPS vs Batch Size for both scenarios and save as PNG.
    """
    if not only_results:
        return
    only_batches = [b for b, _ in only_results]
    only_fps = [f for _, f in only_results]
    pipeline_map = {b: f for b, f in pipeline_results}
    pipeline_fps_aligned = [pipeline_map.get(b, np.nan) for b in only_batches]

    plt.figure(figsize=(10, 6))
    plt.plot(only_batches, only_fps, marker='o', label='AI Inference Only FPS')
    plt.plot(only_batches, pipeline_fps_aligned, marker='s', label='Full Pipeline FPS')
    plt.title('Benchmark: FPS vs Batch Size', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('FPS', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def collect_system_info_text():
    """
    Collect system, CPU, memory, and GPU information as text for saving.
    """
    lines = []
    lines.append("===== System Info =====")
    lines.append(f"Platform       : {platform.system()} {platform.release()}")
    lines.append(f"Processor      : {platform.processor()}")
    lines.append(f"Python Version : {platform.python_version()}")
    lines.append(f"Time           : {datetime.now()}")
    lines.append("")

    lines.append("===== CPU Info =====")
    lines.append(f"Logical CPUs   : {psutil.cpu_count(logical=True)}")
    lines.append(f"Physical CPUs  : {psutil.cpu_count(logical=False)}")
    lines.append(f"CPU Usage (%)  : {psutil.cpu_percent(interval=1)}%")
    freq = psutil.cpu_freq()
    if freq:
        lines.append(f"CPU Frequency  : {freq.current:.2f} MHz")
    lines.append("")

    lines.append("===== Memory Info =====")
    vm = psutil.virtual_memory()
    lines.append(f"Total RAM      : {vm.total / (1024**3):.2f} GB")
    lines.append(f"Available RAM  : {vm.available / (1024**3):.2f} GB")
    lines.append(f"Used RAM       : {vm.used / (1024**3):.2f} GB")
    lines.append(f"Memory Usage   : {vm.percent}%")
    lines.append("")

    lines.append("===== GPU Info =====")
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            lines.append("No GPU found.")
        for gpu in gpus:
            lines.append(f"Name           : {gpu.name}")
            lines.append(f"ID             : {gpu.id}")
            lines.append(f"Memory Total   : {gpu.memoryTotal} MB")
            lines.append(f"Memory Used    : {gpu.memoryUsed} MB")
            lines.append(f"Memory Free    : {gpu.memoryFree} MB")
            lines.append(f"GPU Load       : {gpu.load * 100:.1f}%")
            lines.append(f"Temperature    : {gpu.temperature} °C")
            lines.append("")
    except Exception as e:
        lines.append(f"GPU info error: {e}")
    return "\n".join(lines)

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

    # ========================================================================
    # SAVE RESULTS: CSV AND FIGURES
    # ========================================================================
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'ai_infer_evaluator')
    os.makedirs(results_dir, exist_ok=True)

    # Save combined results to CSV
    csv_path = os.path.join(results_dir, 'benchmark_summary.csv')
    save_benchmark_results_to_csv(only_results, pipeline_results, csv_path)

    # Save line chart figure comparing scenarios
    fig_path = os.path.join(results_dir, 'fps_vs_batch.png')
    visualize_benchmark_results(only_results, pipeline_results, fig_path)

    # Save system information to a text file
    sysinfo_path = os.path.join(results_dir, 'system_info.txt')
    with open(sysinfo_path, 'w', encoding='utf-8') as f:
        f.write(collect_system_info_text())

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved figure to: {fig_path}")
    print(f"Saved system info to: {sysinfo_path}")