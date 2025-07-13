"""
Stream Testing Module
Simple test script to verify video stream connectivity and performance.

This script provides:
- Stream connectivity testing
- Real-time FPS monitoring
- Visual frame display with FPS overlay
- Stream validation before main pipeline execution

Key Features:
- Configuration-based stream source selection
- Real-time FPS calculation and display
- Frame counting and statistics
- Interactive display with quit functionality

Usage:
    python test_stream.py
    
Author: EVEMASK Team
"""

import cv2
import yaml
import os
import time

# ============================================================================
# CONFIGURATION LOADING
# ============================================================================
# Load configuration from YAML file
# This uses the same configuration as the main pipeline
config_path = os.path.join(os.path.dirname(__file__), "cfg", "default.yaml")
with open(os.path.abspath(config_path), "r") as f:
    cfg = yaml.safe_load(f)

# Get input source from configuration
# Can be modified to test different streams
input_source = cfg.get('INPUT_SOURCE', 0)
# input_source = cfg.get('OUTPUT_STREAM_URL_UDP', 0)  # Alternative: test output stream

# ============================================================================
# STREAM INITIALIZATION
# ============================================================================
# Initialize video capture
cap = cv2.VideoCapture(input_source)
if not cap.isOpened():
    print(f"Không thể mở stream: {input_source}")
    exit()

# Get theoretical FPS from stream metadata
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Stream FPS (theoretical): {fps}")

# ============================================================================
# STREAM TESTING LOOP
# ============================================================================
# Initialize counters and timers for FPS calculation
frame_count = 0
fps_counter = 0
fps_display = 0
fps_timer = time.time()

while True:
    # Read frame from stream
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được frame từ stream hoặc stream đã kết thúc.")
        break
    
    # Update frame counters
    frame_count += 1
    fps_counter += 1
    
    # Calculate real-time FPS every second
    now = time.time()
    if now - fps_timer >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_timer = now

    # ============================================================================
    # VISUAL DISPLAY
    # ============================================================================
    # Create copy of frame for display (to avoid modifying original)
    show_frame = frame.copy()
    
    # Draw FPS information on frame
    cv2.putText(show_frame, f"FPS: {fps_display}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Display frame in window
    cv2.imshow("Test Stream Input", show_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Thoát chương trình...")
        break

# ============================================================================
# CLEANUP
# ============================================================================
# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Hoàn thành! Đã nhận {frame_count} frames từ stream.") 