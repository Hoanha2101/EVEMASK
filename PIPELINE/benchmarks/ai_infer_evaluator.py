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
import json

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

def AI_Inference_Only_Benchmark(batch_size, times_avg = 20, warm_up_times = 10):
    """
    Run AI inference benchmark for a specific batch size.
    Returns the result for this batch size only.
    """
    print(f"\n=== Running AI Inference Only Benchmark for Batch Size: {batch_size} ===")
    
    ai_instance = AI.get_instance(cfg=cfg, FEmodel=True)
    cfg["INPUT_SOURCE"] = "videos/1.mp4"
    
    processed_batch = []
    ai_instance._instance_list_ = []
    
    for i in range(batch_size):
        original_frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        img_tensor = np.random.rand(1, 3, 640, 640).astype(np.float16)
        ratio = 0.3333333333333333
        dwdh = (0.0, 140.0)
        img_no_255 = np.random.rand(1, 3, 640, 640).astype(np.float16)
        ai_instance._instance_list_.append(Frame(i, original_frame))
        processed_batch.append((original_frame, img_tensor, ratio, dwdh, img_no_255))
    
    # WARM UP
    print(f"Warming up with {warm_up_times} iterations...")
    for _ in range(warm_up_times):
        ai_instance.inference(processed_batch = processed_batch)
    
    # ACTUAL BENCHMARK
    print(f"Running benchmark with {times_avg} iterations...")
    start_time = time.time()
    for _ in range(times_avg):
        ai_instance.inference(processed_batch = processed_batch) 
    end_time = time.time()
    
    ai_fps = round(batch_size/((end_time - start_time) / times_avg), 4)
    print(f"Batch size: {batch_size} - AI FPS: {ai_fps}")
    
    # Clean up
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return (batch_size, ai_fps)

def AI_Inference_Pipeline_Benchmark(batch_size, times_avg=200, warm_up_times=2):
    """
    Run full pipeline benchmark for a specific batch size.
    Returns the result for this batch size only.
    """
    print(f"\n=== Running Full Pipeline Benchmark for Batch Size: {batch_size} ===")
    
    cfg["INPUT_SOURCE"] = "videos/1.mp4"
    cfg["DELAY_TIME"] = warm_up_times
    cfg["batch_size"] = batch_size

    # Initialize logger, stream controller, and AI engine
    logger = EveMaskLogger.get_instance()
    streamController = StreamController(cfg)
    useAI = AI(cfg, FEmodel=True)
    
    try:
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
        FPS_LIST = []

        while True:
            time.sleep(0.01)
            if streamController._write_frame_index > 20:
                if logger.ai_fps > 0:
                    FPS_LIST.append(logger.ai_fps)
            if streamController._write_frame_index > anchor_count:
                break
                
        # Calculate and print the average AI FPS for this batch size
        if FPS_LIST:
            ai_fps_now = sum(FPS_LIST) / len(FPS_LIST)
            print(f"Batch size: {batch_size} - AI FPS: {ai_fps_now:.4f}")
            result = (batch_size, round(ai_fps_now, 4))
        else:
            print(f"Batch size: {batch_size} - No FPS data collected")
            result = (batch_size, 0.0)
            
    except Exception as e:
        print(f"Error in pipeline benchmark for batch size {batch_size}: {e}")
        result = (batch_size, 0.0)
    finally:
        # Clean up
        streamController.stop()
        if hasattr(useAI, 'stop'):
            useAI.stop()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    return result

def save_temporary_result(batch_size, ai_only_fps, pipeline_fps, temp_dir):
    """
    Save temporary result for a specific batch size.
    """
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f"batch_{batch_size}_result.json")
    
    result_data = {
        "batch_size": batch_size,
        "ai_inference_only_fps": ai_only_fps,
        "full_pipeline_fps": pipeline_fps,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Saved temporary result for batch size {batch_size} to {temp_file}")

def load_temporary_results(temp_dir):
    """
    Load all temporary results and combine them.
    """
    results = []
    if not os.path.exists(temp_dir):
        return results
    
    for filename in os.listdir(temp_dir):
        if filename.startswith("batch_") and filename.endswith("_result.json"):
            filepath = os.path.join(temp_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append((
                        data["batch_size"],
                        data["ai_inference_only_fps"],
                        data["full_pipeline_fps"]
                    ))
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    # Sort by batch size
    results.sort(key=lambda x: x[0])
    return results

def run_single_batch_benchmark(batch_size, temp_dir):
    """
    Run both benchmarks for a single batch size and save temporary result.
    """
    print(f"\n{'='*60}")
    print(f"STARTING BENCHMARK FOR BATCH SIZE: {batch_size}")
    print(f"{'='*60}")
    
    try:
        # Run AI inference only benchmark
        ai_only_result = AI_Inference_Only_Benchmark(batch_size)
        
        # Run full pipeline benchmark
        pipeline_result = AI_Inference_Pipeline_Benchmark(batch_size)
        
        # Save temporary result
        save_temporary_result(
            batch_size, 
            ai_only_result[1], 
            pipeline_result[1], 
            temp_dir
        )
        
        print(f"\n✓ Completed benchmark for batch size {batch_size}")
        print(f"AI Inference Only FPS: {ai_only_result[1]}")
        print(f"Full Pipeline FPS: {pipeline_result[1]}")
        
    except Exception as e:
        print(f"✗ Error in benchmark for batch size {batch_size}: {e}")
        # Save error result
        save_temporary_result(batch_size, 0.0, 0.0, temp_dir)

def save_benchmark_results_to_csv(combined_results, csv_path):
    """
    Save combined benchmark results to a CSV file.
    Columns: batch_size, ai_inference_only_fps, full_pipeline_fps
    """
    fieldnames = ["batch_size", "ai_inference_only_fps", "full_pipeline_fps"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for batch_size, ai_fps, pipeline_fps in combined_results:
            writer.writerow({
                "batch_size": batch_size,
                "ai_inference_only_fps": ai_fps,
                "full_pipeline_fps": pipeline_fps
            })

def visualize_benchmark_results(combined_results, save_path):
    """
    Visualize FPS vs Batch Size for both scenarios and save as PNG.
    """
    if not combined_results:
        return
    
    batch_sizes = [r[0] for r in combined_results]
    ai_fps = [r[1] for r in combined_results]
    pipeline_fps = [r[2] for r in combined_results]

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, ai_fps, marker='o', label='AI Inference Only FPS')
    plt.plot(batch_sizes, pipeline_fps, marker='s', label='Full Pipeline FPS')
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

def main():
    """
    Main function to run benchmarks for each batch size separately.
    """
    print("TensorRT Batch Size Benchmark Tool")
    print("This tool will run benchmarks for each batch size separately")
    print("to avoid TensorRT batch size change issues.")
    print()
    
    # Get configuration
    MAX_BATCH_SIZE = cfg['MAX_BATCH_SIZE']
    print(f"Maximum batch size from config: {MAX_BATCH_SIZE}")
    
    # Create temporary directory for storing results
    temp_dir = os.path.join(os.path.dirname(__file__), 'results', 'temp_results')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Check if we have existing temporary results
    existing_results = load_temporary_results(temp_dir)
    completed_batches = [r[0] for r in existing_results]
    
    if existing_results:
        print(f"Found existing results for batch sizes: {completed_batches}")
        print("You can:")
        print("1. Continue from where you left off")
        print("2. Start fresh (delete existing results)")
        print("3. Just combine existing results")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "2":
            # Delete existing results
            import shutil
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            existing_results = []
            completed_batches = []
            print("Deleted existing results. Starting fresh.")
        elif choice == "3":
            # Just combine existing results
            print("Combining existing results...")
            combined_results = existing_results
            save_final_results(combined_results)
            return
        # Choice 1: continue from where left off
        else:
            print("Continuing from where you left off...")
    
    # Run benchmarks for each batch size
    for batch_size in range(1, MAX_BATCH_SIZE + 1):
        if batch_size in completed_batches:
            print(f"Batch size {batch_size} already completed, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"READY TO RUN BENCHMARK FOR BATCH SIZE: {batch_size}")
        print(f"{'='*60}")
        print("Press Enter when ready to continue, or 'q' to quit...")
        
        user_input = input().strip().lower()
        if user_input == 'q':
            print("Benchmark stopped by user.")
            break
            
        # Run benchmark for this batch size
        run_single_batch_benchmark(batch_size, temp_dir)
        
        print(f"\nBatch size {batch_size} completed!")
        print("You can now:")
        print("1. Continue to next batch size")
        print("2. Stop here and combine results later")
        print("3. Quit")
        
        continue_choice = input("Enter choice (1/2/3): ").strip()
        if continue_choice == "2":
            print("Stopping here. You can run the script again later to continue.")
            break
        elif continue_choice == "3":
            print("Quitting...")
            break
    
    # Combine all results
    print("\nCombining all results...")
    combined_results = load_temporary_results(temp_dir)
    
    if combined_results:
        save_final_results(combined_results)
    else:
        print("No results to combine.")

def save_final_results(combined_results):
    """
    Save final combined results and display summary.
    """
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
    for batch_size, ai_fps, pipeline_fps in combined_results:
        table.append([batch_size, ai_fps, pipeline_fps])
    
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
    save_benchmark_results_to_csv(combined_results, csv_path)

    # Save line chart figure comparing scenarios
    fig_path = os.path.join(results_dir, 'fps_vs_batch.png')
    visualize_benchmark_results(combined_results, fig_path)

    # Save system information to a text file
    sysinfo_path = os.path.join(results_dir, 'system_info.txt')
    with open(sysinfo_path, 'w', encoding='utf-8') as f:
        f.write(collect_system_info_text())

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved figure to: {fig_path}")
    print(f"Saved system info to: {sysinfo_path}")

if __name__ == "__main__":
    main()