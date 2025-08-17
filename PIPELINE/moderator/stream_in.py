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
    python stream_in.py
    
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
config_path = os.path.join(os.path.dirname(__file__), "..", "cfg", "default.yaml")
with open(os.path.abspath(config_path), "r") as f:
    cfg = yaml.safe_load(f)

# Get input source from configuration
# Can be modified to test different streams
input_source = cfg.get('INPUT_SOURCE', 0)

# ============================================================================
# STREAM INITIALIZATION
# ============================================================================
# Initialize video capture
cap = cv2.VideoCapture(input_source)
if not cap.isOpened():
    print(f"Could not open stream: {input_source}")
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
        print("Could not receive frame from stream or stream has ended.")
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
    
    # Get frame size and number of color channels information
    height, width = frame.shape[:2]
    channels = frame.shape[2] if len(frame.shape) == 3 else 1

    # --- Draw info box background ---
    # Box parameters
    box_x, box_y = 10, 10
    box_w, box_h = 350, 130  # width, height of the box
    box_color = (30, 30, 30)  # dark gray
    box_alpha = 0.6  # transparency
    # Draw filled rectangle with alpha blending
    overlay = show_frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, -1)
    cv2.addWeighted(overlay, box_alpha, show_frame, 1 - box_alpha, 0, show_frame)

    # Draw information on the frame (inside the box)
    cv2.putText(show_frame, f"FPS: {fps_display}", (box_x + 10, box_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(show_frame, f"Size: {width}x{height}", (box_x + 10, box_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(show_frame, f"Channels: {channels}", (box_x + 10, box_y + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display frame in window
    cv2.imshow("Test Stream Input", show_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exit the program...")
        break

# ============================================================================
# CLEANUP
# ============================================================================
# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Done! Received {frame_count} frames from stream.") 