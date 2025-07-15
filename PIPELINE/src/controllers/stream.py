"""
Stream Controller Module
Handles video stream capture, processing, and output streaming using FFmpeg.
This module manages the input/output video streams and coordinates with the AI processing pipeline.

Key Features:
- Real-time video capture from various sources (UDP, RTSP, RTMP)
- Frame buffering using circular queue
- FFmpeg-based output streaming with audio synchronization
- FPS monitoring and optimization
- Thread-safe operations

Author: EVEMASK Team
"""

from .circle_queue import CircleQueue
from .frame import Frame
import cv2
import time
import subprocess
import threading
import numpy as np
from collections import deque


class StreamController:
    """
    Main controller for video stream operations.
    
    This class manages:
    - Input video capture from various sources
    - Output streaming using FFmpeg
    - Frame buffering and synchronization
    - FPS monitoring and optimization
    
    Attributes:
        circle_queue: Circular queue for frame buffering
        ai_instance: Reference to AI processing instance
        cfg: Configuration dictionary
        INPUT_SOURCE: Input stream URL
        target_fps: Target output FPS
        width/height: Video dimensions
        running: Thread control flag
        ffmpeg_process: FFmpeg subprocess handle
        cap: OpenCV video capture object
    """
    _global_instance: "StreamController" = None
    
    def __init__(self, cfg):
        """
        Initialize the stream controller with configuration.
        
        Args:
            cfg (dict): Configuration dictionary containing stream settings
        """
        # Get singleton instances
        self.circle_queue = CircleQueue.get_instance()
        from ..brain.AI import AI
        from ..logger import EveMaskLogger
        self.logger = EveMaskLogger.get_instance()
        self.ai_instance = AI.get_instance(cfg=cfg)
        
        # Store configuration
        self.cfg = cfg
        self.INPUT_SOURCE = cfg['INPUT_SOURCE']
        self.target_fps = cfg['TARGET_FPS']
        self.batch_size = cfg['batch_size']
        
        # Video properties
        self.width = None
        self.height = None
        
        # Control flags
        self.running = True
        
        # Process handles
        self.ffmpeg_process = None
        self.cap = None
        
        # Frame tracking
        self._frame_index = 0
        self._write_frame_index = 0
        
        # FPS monitoring
        self._frame_times = []  # Store frame timestamps for FPS calculation
        self._last_fps_calc = time.time()
        
        self.out_timestamps = deque(maxlen=100)  # Store output timestamps for FPS calculation
        self.last_fps_update = time.time()
        
        # Initialize video capture
        self._init_capture()

    def _init_capture(self):
        """
        Initialize video capture from input source.
        
        This method continuously attempts to connect to the input stream
        until successful. It handles various input formats (UDP, RTSP, etc.)
        and sets up optimal buffer settings for real-time processing.
        """
        print(f"Waiting for input stream: {self.INPUT_SOURCE}")
        while True:
            try:
                # Create video capture object
                cap = cv2.VideoCapture(self.INPUT_SOURCE)
                
                # Optimize buffer settings for real-time processing
                # Smaller buffer reduces latency but may cause frame drops
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test capture by reading a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Input stream: {self.INPUT_SOURCE} is ready")
                    
                    # Store video dimensions
                    self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.cap = cap
                    self.begin_time = time.time()
                    
                    print(f"Input resolution: {self.width}x{self.height}")
                    break
                else:
                    print(f"Failed to read from {self.INPUT_SOURCE}")
                    cap.release()
                    
            except Exception as e:
                print(f"Error initializing capture: {e}")
                if 'cap' in locals():
                    cap.release()
                    
            # Wait before retrying connection
            print(f"[{time.time()}] Waiting for input stream: {self.INPUT_SOURCE}")
            time.sleep(1)

    def _start_ffmpeg(self):
        """
        Start FFmpeg process for output streaming.
        
        This method creates an FFmpeg subprocess that:
        - Takes raw video frames from stdin
        - Synchronizes with original audio stream
        - Outputs to specified destination (UDP, RTSP, RTMP)
        - Applies video processing and encoding
        """
        # Calculate audio delay to synchronize with video
        delay = time.time() - self.begin_time if self.begin_time else 0
        
        # Build FFmpeg command for real-time streaming
        ffmpeg_command = [
            "ffmpeg",
            "-re",  # Real time input - read input at native frame rate
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",  # Input pixel format
            "-s", "{}x{}".format(self.width, self.height),  # Input resolution
            "-r", str(self.target_fps),  # Input frame rate
            "-i", "-",  # Read from stdin
            "-i", self.INPUT_SOURCE,  # Original stream for audio
            "-af", f"adelay={delay * 1000}|{delay * 1000}",  # Audio delay for sync
            "-async", "1",  # Audio sync
            "-vsync", "1",  # Video sync
            "-q:v", "1",  # Video quality (1 = best)
            "-map", "1:a",  # Use audio from second input
            "-map", "0:v",  # Use video from first input
            "-c:v", "libx264",  # Video codec
            "-pix_fmt", "yuv420p",  # Output pixel format
            "-preset", "ultrafast",  # Encoding preset for low latency
            "-color_primaries", "bt709",  # Color space settings
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-vf", "yadif",  # Deinterlace filter
            "-f",  # Output format
        ]
        
        # Configure output based on stream type
        if self.cfg['OUTPUT_TYPE'] == "rtsp":
            ffmpeg_command.extend([
                "rtsp", "-rtsp_transport", "tcp", self.cfg['OUTPUT_STREAM_URL_RTSP']
            ])
            output_stream_url = self.cfg['OUTPUT_STREAM_URL_RTSP']
        elif self.cfg['OUTPUT_TYPE'] == "rtmp":
            ffmpeg_command.extend([
                "flv", self.cfg['OUTPUT_STREAM_URL_RTMP']
            ])
            output_stream_url = self.cfg['OUTPUT_STREAM_URL_RTMP']
        else:  # Default to UDP
            ffmpeg_command.extend([
                "mpegts", self.cfg['OUTPUT_STREAM_URL_UDP']
            ])
            output_stream_url = self.cfg['OUTPUT_STREAM_URL_UDP']
            
        print(f"Output URL: {output_stream_url}")
        
        try:
            # Start FFmpeg subprocess
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_command, 
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL  # Reduce logging overhead
            )
            print("FFmpeg process started successfully")
        except Exception as e:
            print(f"Error starting FFmpeg: {e}")
            self.ffmpeg_process = None

    def _calculate_input_fps(self):
        """
        Calculate input FPS based on frame arrival times.
        
        This method tracks frame timestamps and calculates the actual
        input frame rate to help optimize AI processing.
        
        Returns:
            float or None: Calculated FPS if enough data, None otherwise
        """
        current_time = time.time()
        self._frame_times.append(current_time)
        
        # Only calculate FPS every 3 seconds to avoid fluctuations
        if current_time - self._last_fps_calc > 3.0:
            if len(self._frame_times) > 1:
                # Calculate FPS based on frame count over time span
                time_span = self._frame_times[-1] - self._frame_times[0]
                if time_span > 0:
                    input_fps = (len(self._frame_times) - 1) / time_span
                    return input_fps
            
            # Reset tracking for next calculation
            self._frame_times = [current_time]
            self._last_fps_calc = current_time
        
        return None

    def source_capture(self):
        """
        Main capture loop that reads frames from input stream.
        
        This method runs in a separate thread and continuously:
        - Reads frames from the input source
        - Adds frames to the circular queue for processing
        - Tracks frame timing for FPS calculation
        - Updates AI instance with input FPS information
        """
        print("Starting source capture...")
        
        while self.running:
            try:
                # Read frame from input source
                ret, data = self.cap.read()
                if ret and data is not None:
                    # Create frame object and add to queue
                    frame = Frame(frame_id=self._frame_index, frame_data=data)
                    self.circle_queue.add_frame(frame=frame)
                    self._frame_index += 1
                    
                    # Calculate and update input FPS
                    input_fps = self._calculate_input_fps()
                    self.logger.update_in_stream_fps(input_fps)
                    if input_fps is not None:
                        # Update AI instance with current input FPS
                        try:     
                            if self.ai_instance:
                                self.ai_instance.update_input_fps(input_fps)
                        except Exception as e:
                            print(f"Error updating AI FPS: {e}")
                else:
                    # Short sleep if no frame available
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error in capture: {e}")
                time.sleep(0.1)

    def out_stream(self):
        """
        Main output streaming loop.
        
        This method runs in a separate thread and continuously:
        - Reads processed frames from the circular queue
        - Writes frames to FFmpeg for output streaming
        - Maintains output frame rate
        - Cleans up frame resources after streaming
        """
        print("Starting output stream...")
        
        # Start FFmpeg process
        self._start_ffmpeg()
        
        if self.ffmpeg_process is None:
            print("Failed to start FFmpeg")
            return
            
        while self.running:
            # Check if frame is available for output
            start = time.time()
            if (self._write_frame_index in self.circle_queue.frames.keys()) and (self.ai_instance.mooc_processed_frames >= self._write_frame_index):
                # Get frame from queue
                frame_out = self.circle_queue.get_by_id(self._write_frame_index)
                self.logger.update_number_out_frames(self._write_frame_index)
                if frame_out is not None:
                    # Convert frame to bytes and write to FFmpeg
                    frame_bytes = frame_out.frame_data.tobytes()
                    self.ffmpeg_process.stdin.write(frame_bytes)
                    
                    # Clean up frame resources
                    frame_out.destroy()
                    # Update timestamp
                    self.out_timestamps.append(time.time())
                    # Move to next frame
                    self._write_frame_index += 1
                
                # Control output rate
                time.sleep(0.01)
                # time.sleep(max(0,1/self.target_fps - time.time() + start - 0.001))
            else:
                time.sleep(0.01)
                # Wait if frame not available
                # time.sleep(max(0,1/self.target_fps - time.time() + start - 0.001))
            
            # Calculate output FPS
            now = time.time()
            if now - self.last_fps_update >= 1.0:
                if len(self.out_timestamps) >= 2:
                    time_deltas = [t2 - t1 for t1, t2 in zip(self.out_timestamps, list(self.out_timestamps)[1:])]
                    avg_delta = sum(time_deltas) / len(time_deltas)
                    fps = 1.0 / avg_delta if avg_delta > 0 else 0.0
                    self.logger.update_out_stream_fps(fps)
                self.last_fps_update = now
                    
        print("Output stream stopped")
        self._cleanup_ffmpeg()

    def _cleanup_ffmpeg(self):
        """
        Clean up FFmpeg process and resources.
        
        This method properly terminates the FFmpeg subprocess
        and handles any cleanup errors gracefully.
        """
        if self.ffmpeg_process is not None:
            try:
                # Close stdin and wait for process to finish
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=5)
            except Exception as e:
                print(f"Error closing ffmpeg: {e}")
                try:
                    # Force kill if graceful shutdown fails
                    self.ffmpeg_process.kill()
                except:
                    pass
            self.ffmpeg_process = None

    def stop(self):
        """
        Stop all streaming operations and clean up resources.
        
        This method should be called when shutting down the system
        to ensure proper cleanup of all resources.
        """
        self.running = False
        if self.cap:
            self.cap.release()
        self._cleanup_ffmpeg() 
    
    @classmethod
    def get_instance(cls) -> "StreamController":
        """
        Get the singleton instance of StreamController.
        """
        if cls._global_instance is None:
            cls._global_instance = StreamController(cfg=cfg)
        return cls._global_instance