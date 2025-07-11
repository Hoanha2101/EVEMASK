import yaml
import os
import cv2
import threading
from src.controllers import CircleQueue, StreamController
from src.brain import AI
import time
import sys


if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "cfg", "default.yaml")
    with open(os.path.abspath(config_path), "r") as f:
        cfg = yaml.safe_load(f)
        
    
    # Initialize components
    streamController = StreamController(cfg)
    useAI = AI(cfg, FEmodel=False)

    print("Configuration loaded successfully")
    print(f"Input source: {cfg.get('INPUT_SOURCE', 'Not specified')}")
    print(f"Output type: {cfg.get('OUTPUT_TYPE', 'Not specified')}")
    print(f"Batch size: {cfg.get('batch_size', 'Not specified')}")
    print(f"Target FPS: {cfg.get('TARGET_FPS', 'Not specified')}")
    print("Components initialized successfully")

    # --- Start threads ---
    print("Starting threads...")
    
    capture_thread = threading.Thread(target=streamController.source_capture, name="CaptureThread")
    ai_thread = threading.Thread(target=useAI.run, name="AIThread")
    output_thread = threading.Thread(target=streamController.out_stream, name="OutputThread")

    # Set as daemon threads
    capture_thread.daemon = True
    ai_thread.daemon = True
    output_thread.daemon = True

    # Start threads with proper order
    capture_thread.start()
    print("Capture thread started")
    
    ai_thread.start()
    print("AI thread started")
    
    # Wait before starting output to ensure some frames are processed
    delay_time = cfg.get('DELAY_TIME', 0)
    if delay_time > 0:
        print(f"Waiting {delay_time} seconds before starting output thread...")
        time.sleep(delay_time)
    
    output_thread.start()
    print("Output thread started")
    
    print("All threads started successfully. Press Ctrl+C to stop.")
    
    # Main loop with monitoring
    last_check = time.time()
    while True:
        time.sleep(1)