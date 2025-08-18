"""
Main entry point for the EVEMASK Pipeline system.
This system processes video streams in real-time using AI models for object detection and classification.
The pipeline consists of three main threads: capture, AI processing, and output streaming.

Author: EVEMASK Team
Version: 1.0.0
"""

import yaml
import os
import threading
from src.controllers import CircleQueue, StreamController
from src.brain import AI
import time
from src.logger import EveMaskLogger
import sys

if __name__ == "__main__":
    # ========================================================================
    # LOGGER INITIALIZATION
    # ========================================================================
    logger = EveMaskLogger.get_instance()

    # ========================================================================
    # DISPLAY LOGO AND START MESSAGE
    # ========================================================================
    logger.display_logo()
    logger.display_start_message()

    # ========================================================================
    # LOAD CONFIG
    # ========================================================================
    print("🔧 Loading configuration...")
    config_path = os.path.join(os.path.dirname(__file__), "cfg", "default.yaml")
    with open(os.path.abspath(config_path), "r") as f:
        cfg = yaml.safe_load(f)
    logger.show_config(cfg)

    # ========================================================================
    # INITIALIZE COMPONENTS
    # ========================================================================
    print("⚙️  Initializing pipeline components...")
    streamController = StreamController.get_instance(cfg)
    useAI = AI(cfg, FEmodel=cfg["USE_FEATURE_EXTRACTION"], boxPlot= cfg["USE_BOX_PLOT"])
    print("✅ Components initialized successfully.\n")

    # ========================================================================
    # THREAD MANAGEMENT
    # ========================================================================
    print("🔁 Starting system threads...\n")

    capture_thread = threading.Thread(target=streamController.source_capture, name="CaptureThread", daemon=True)
    ai_thread = threading.Thread(target=useAI.run, name="AIThread", daemon=True)
    output_thread = threading.Thread(target=streamController.out_stream, name="OutputThread", daemon=True)

    capture_thread.start()
    print("🎥 Capture thread ........... ✅ started")

    ai_thread.start()
    print("🧠 AI thread ................ ✅ started")

    # ========================================================================
    # DELAY TIME (with progress bar)
    # ========================================================================
    logger.waiting_bar(cfg)

    output_thread.start()
    print("📤 Output thread ............ ✅ started")

    print("\n🚀 All threads are now running.")
    print("📡 Real-time processing has begun. Press Ctrl+C to exit.\n")

    # ========================================================================
    # MAIN MONITORING LOOP
    # ========================================================================
    
    logger.start_live_display(cfg, capture_thread.is_alive(), output_thread.is_alive(), ai_thread.is_alive())
    try:
        while True:
            input_alive = capture_thread.is_alive()
            output_alive = output_thread.is_alive()
            ai_alive = ai_thread.is_alive()
            
            # Check if StreamController's running status is False (stream ended)
            if not streamController.running:
                print("\nStream has ended. Shutting down entire system...")
                logger.stop_live_display()
                
                # Stop all components gracefully
                streamController.stop()
                useAI.stop() if hasattr(useAI, 'stop') else None
                
                # Display logo and end message
                logger.display_logo()
                logger.display_end_message()
                sys.exit(0)
            
            logger.update_live_display(cfg, input_alive, output_alive, ai_alive)
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down system...")
        logger.stop_live_display()
        
        # Stop all components gracefully
        streamController.stop()
        useAI.stop() if hasattr(useAI, 'stop') else None
        
        # Display logo and end message
        logger.display_logo()
        logger.display_end_message()
        sys.exit(0)