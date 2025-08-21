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
from src.controllers import circle_queue
import os
import datetime
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
    
    # ========================================================================
    # __init__ StreamController
    # ========================================================================
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
        self.input_source_type = self.detect_input_stream(self.INPUT_SOURCE)
        self.target_fps = cfg['TARGET_FPS']
        self.batch_size = cfg['batch_size']
        self.application = cfg["APPLICATION"]
        self.save_stream = cfg["SAVE_STREAM_TO_VIDEO"]
        self.path_save_stream = cfg['FOLDER_SAVE_STREAM_TO_VIDEO']
        
        # Auto set batch_size for APPLICATION
        if self.application == "VIDEO":
            self.batch_size = cfg["MAX_BATCH_SIZE"]
        
        # Video properties
        self.width = None
        self.height = None
        
        # Control flags
        self.running = True
        
        # Process handles
        self.ffmpeg_process = None
        self.cap = None
        
        #delay time
        self.delay = 0
        
        # Frame tracking
        self._frame_index = 0
        self._write_frame_index = 0
        
        # Track consecutive failed frame reads
        self._consecutive_failed_reads = 0 
        
        # FPS monitoring
        # Store frame timestamps for FPS calculation
        self._frame_times = []  
        self._last_fps_calc = time.time()
        
        # Store output timestamps for FPS calculation
        self.out_timestamps = deque(maxlen=100)
        self.last_fps_update = time.time()
        
        # Initialize video capture
        self._init_capture()
        
        if self.save_stream:
            self._init_save_stream_to_video()
            self.video_writer_stream = None
            self.audio_extraction_started = False
            
    # ========================================================================
    # __init__ Path, Name - Save stream to video
    # ========================================================================    
    def _init_save_stream_to_video(self):
        """Initialize paths for saving video and audio separately."""
        # Create save directory
        os.makedirs(self.path_save_stream, exist_ok=True)
        # Generate base filename with timestamp
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"EVEMASK@streamProcessed_{time_str}"
        # Separate file paths
        self.video_only_path = os.path.join(self.path_save_stream, f"{base_name}_video.mp4")
        self.audio_only_path = os.path.join(self.path_save_stream, f"{base_name}_audio.aac")
        self.final_video_path = os.path.join(self.path_save_stream, f"{base_name}.mp4")
        
    # ========================================================================
    # __init__ Read function to get frames from input source
    # ========================================================================
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

    # ========================================================================
    # Start audio extraction function
    # ========================================================================
    def _start_audio_extraction(self):
        """Start FFmpeg process to extract audio from input source."""
        if self.audio_extraction_started:
            return 
        try:
            # FFmpeg command to extract audio only from an input video source
            audio_extract_command = [
                "ffmpeg",            # Call the ffmpeg program
                "-y",                # Overwrite output file if it already exists (yes - no prompt)
                "-i", self.INPUT_SOURCE,  # Specify the input source (stream)
                "-vn",               # Disable video recording (ignore the video stream)
                "-acodec", "aac",    # Set audio codec to AAC (Advanced Audio Codec)
                "-b:a", "128k",      # Set audio bitrate to 128 kbps
                self.audio_only_path # Path to the output file that will contain audio only
            ]

            # Start audio extraction in background
            self.audio_process = subprocess.Popen(
                audio_extract_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            self.audio_extraction_started = True
            
        except Exception as e:
            print(f"Error starting audio extraction: {e}")
            self.audio_process = None
    
    # ========================================================================
    # Start FFmpeg tool
    # ========================================================================
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
        self.delay = time.time() - self.begin_time if self.begin_time else 0
        
        # Build FFmpeg command for real-time streaming
        ffmpeg_command = [
            "ffmpeg",                         # Call the ffmpeg program
            "-re",                            # Read input at real-time speed (simulate live input)
            "-f", "rawvideo",                 # First input format: raw video frames
            "-pix_fmt", "bgr24",              # Pixel format of the input: BGR 24-bit (3 channels, 8-bit each)
            "-s", "{}x{}".format(self.width, self.height),  # Frame size: width x height
            "-r", str(self.target_fps),       # Frame rate (fps)
            "-i", "-",                        # Input video from stdin (piped raw frames)
            "-i", self.INPUT_SOURCE,          # Second input source (file or stream containing original audio)
            "-af", f"adelay={self.delay * 1000}|{self.delay * 1000}", # Apply audio delay (milliseconds) for both left/right channels
            "-async", "1",                    # Audio sync (compensates drift/delay)
            "-vsync", "1",                    # Video sync mode (1 = frame duplication/drop to sync)
            "-q:v", "1",                      # Video quality (1 = highest quality, mainly for mjpeg-like codecs)
            "-map", "1:a",                    # Select audio stream from second input
            "-map", "0:v:0",                  # Select video stream from first input (stdin raw video)
            "-c:v", "libx264",                # Encode video using H.264 (x264 encoder)
            "-pix_fmt", "yuv420p",            # Output pixel format yuv420p (widely compatible)
            "-preset", "ultrafast",           # Encoder preset: fastest speed (larger output size)
            "-color_primaries", "bt709",      # Set color primaries to BT.709 (HD video standard)
            "-color_trc", "bt709",            # Set transfer characteristics (gamma) to BT.709
            "-colorspace", "bt709",           # Set colorspace to BT.709
            "-vf", "yadif",                   # Apply video filter: deinterlace (yadif = Yet Another DeInterlacing Filter)
            "-f"                              # Specify output format (to be defined later, e.g., "rtsp", "flv", "mp4")
        ]
        
        # Configure output based on stream type
        # OUTPUT_TYPE can be "rtsp", "rtmp", or "udp"
        # Each protocol has its own purpose and use case:
        # - RTSP (Real Time Streaming Protocol): Common for low-latency IP camera streams and live monitoring.
        # - RTMP (Real-Time Messaging Protocol): Originally developed by Macromedia/Adobe for Flash, now widely used for
        #   pushing live streams to platforms like YouTube, Facebook, Twitch. Uses "flv" container format in FFmpeg.
        # - UDP (User Datagram Protocol): Sends video in MPEG-TS format over the network; very fast, often used for multicast
        #   or broadcast streaming in local networks, but without guaranteed packet delivery.

        if self.cfg['OUTPUT_TYPE'] == "rtsp":
            # Output to an RTSP stream using TCP transport for reliability
            ffmpeg_command.extend([
                "rtsp", "-rtsp_transport", "tcp", self.cfg['OUTPUT_STREAM_URL_RTSP']
            ])
            output_stream_url = self.cfg['OUTPUT_STREAM_URL_RTSP']

        elif self.cfg['OUTPUT_TYPE'] == "rtmp":
            # Output to an RTMP server; FFmpeg uses "flv" format for RTMP streaming
            ffmpeg_command.extend([
                "flv", self.cfg['OUTPUT_STREAM_URL_RTMP']
            ])
            output_stream_url = self.cfg['OUTPUT_STREAM_URL_RTMP']

        else:  # Default to UDP
            # Output over UDP in MPEG-TS format; often used for fast local streaming or multicast
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

    # ========================================================================
    # __init__ video_writer
    # ========================================================================
    def _setup_video_writer(self, height, width):
        """Setup OpenCV VideoWriter for video-only recording."""
        try:
            # Try different codecs for compatibility
            codecs_to_try = [
                ('mp4v', '.mp4'),
                ('XVID', '.avi'),
                ('MJPG', '.avi'),
            ]
            
            for codec, ext in codecs_to_try:
                try:
                    # Update file extension if needed
                    if not self.video_only_path.endswith(ext):
                        base_name = os.path.splitext(self.video_only_path)[0]
                        self.video_only_path = base_name + ext
                    
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    self.video_writer_stream = cv2.VideoWriter(
                        self.video_only_path,
                        fourcc,
                        self.target_fps,
                        (width, height)
                    )
                    
                    if self.video_writer_stream.isOpened():
                        return True
                    else:
                        self.video_writer_stream.release()
                        
                except Exception as e:
                    print(f"Failed to initialize with codec {codec}: {e}")
                    if self.video_writer_stream:
                        self.video_writer_stream.release()
                    continue
            
            print("Failed to initialize video writer")
            self.save_stream = False
            return False
            
        except Exception as e:
            print(f"Error setting up video writer: {e}")
            self.save_stream = False
            return False
    
    # ========================================================================
    # Merge video vs audio tool
    # ========================================================================
    def _merge_video_audio(self):
        """Merge video and audio files into final output."""
        try:
            # Check if both files exist
            if not os.path.exists(self.video_only_path):
                print("Video file not found, cannot merge")
                return False
                
            if not os.path.exists(self.audio_only_path):
                print("Audio file not found, cannot merge")
                return False
            
            # FFmpeg command to merge video + audio
            merge_command = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-i", self.video_only_path,  # Video input
                "-i", self.audio_only_path,  # Audio input
                "-c:v", "copy",  # Copy video stream (no re-encoding)
                "-c:a", "aac",   # Audio codec
                "-shortest",     # End when shortest stream ends
                self.final_video_path
            ]
            
            # Run merge process
            merge_process = subprocess.run(
                merge_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=300  # 5 minute timeout
            )
            
            if merge_process.returncode == 0:
                
                # Get file size
                if os.path.exists(self.final_video_path):
                    file_size = os.path.getsize(self.final_video_path)
                    print(f"Final video size: {file_size / (1024*1024):.2f} MB")
                
                # Clean up temporary files
                try:
                    os.remove(self.video_only_path)
                    os.remove(self.audio_only_path)
                except:
                    pass
                
                return True
            else:
                error_msg = merge_process.stderr.decode() if merge_process.stderr else "Unknown error"
                print(f"Merge failed: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Merge process timed out")
            return False
        except Exception as e:
            print(f"Error merging video and audio: {e}")
            return False
    
    # ========================================================================
    # Calculate FPS for Input stream block
    # ========================================================================
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
        
        # Only calculate FPS every 3 seconds
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
    
    # ========================================================================
    # Check input stream type
    # ========================================================================
    def detect_input_stream(self, url: str) -> str:
        """
        Specify output stream type (RTMP, RTSP, UDP) or video file
        """
        url = url.strip().lower()

        if url.startswith("rtmp://"):
            return "RTMP"
        elif url.startswith("rtsp://"):
            return "RTSP"
        elif url.startswith("udp://"):
            return "UDP"
        else:
            return "FILE"

    # ========================================================================
    # Input stream block
    # ======================================================================== 
    def source_capture(self):
        """
        Main capture loop that reads frames from input stream.
        
        - Reads frames from input source
        - Adds frames to queue
        - Tracks frame timing for FPS calculation
        - Updates AI instance with input FPS information
        - Stops after too many failed reads
        - Controls input FPS according to self.target_fps
        """
        print("Starting source capture...")
        
        frame_interval = 1.0 / self.target_fps
        next_frame_time = time.time()
        
        while self.running:
            
            # Read frame from input source
            ret, data = self.cap.read()
            if ret and data is not None:
                # Reset failed read counter
                self._consecutive_failed_reads = 0
                
                # Create frame object and add to queue
                frame = Frame(frame_id=self._frame_index, frame_data=data)
                self.circle_queue.add_frame(frame=frame)
                self._frame_index += 1
                
                if self.application == "STREAM" and self.save_stream:
                    self._start_audio_extraction()
                    
                # Calculate and update input FPS
                input_fps = self._calculate_input_fps()
                self.logger.update_in_stream_fps(input_fps)
                if self.application == "STREAM" and self.input_source_type == "FILE":
                    next_frame_time += frame_interval
                    sleep_time = next_frame_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_frame_time = time.time()
                        
                if input_fps is not None:
                    try:     
                        if self.ai_instance:
                            self.ai_instance.update_input_fps(input_fps)
                    except Exception as e:
                        print(f"Error updating AI FPS: {e}")
                        
                if self.application == "VIDEO":
                    while self.circle_queue.queue_length() == 1000:
                        time.sleep(0.01)
                        if self.circle_queue.count_processed_frames() == 1000:
                            self.circle_queue.clear_queue()
                            break
            else:
                # Increment failed read counter
                if self.application == "VIDEO":  
                    self._consecutive_failed_reads += 1
                    if self._consecutive_failed_reads >= 1000 and (self.circle_queue.last_frame_id - 1 == self.ai_instance.mooc_processed_frames):
                        self.running = False
                        self.ai_instance.video_writer.release()
                        print(f"Stream ended: No frames received for {self._consecutive_failed_reads} consecutive attempts")
                        break
                    time.sleep(0.01)
                    
                else:
                    self._consecutive_failed_reads += 1
                    if self._consecutive_failed_reads >= 1000:
                        self.running = False
                        print(f"Stream ended: No frames received for {self._consecutive_failed_reads} consecutive attempts")
                        break
                    time.sleep(0.01)

    # ========================================================================
    # Output stream block
    # ========================================================================
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
        
        if self.application == "STREAM":
            # Start FFmpeg process
            self._start_ffmpeg()
        
        if self.ffmpeg_process is None and self.application == "STREAM":
            print("Failed to start FFmpeg")
            return
            
        while self.running:
            if (self.ai_instance.mooc_processed_frames > self._write_frame_index):
                # Get frame from queue
                frame_out = self.circle_queue.get_by_id(self._write_frame_index)
                self.logger.update_number_out_frames(self._write_frame_index)
                if frame_out is not None:
                    # Output resolution show
                    if self._write_frame_index == 0:
                        h, w = frame_out.frame_data.shape[:2]
                        print(f"Output resolution: {w}x{h}")
                        
                        # Start audio extraction and setup video writer
                        if self.save_stream:
                            self._setup_video_writer(h, w)
                            
                    if self.application == "STREAM":
                        # Save processed frame to video file (without audio)
                        if self.save_stream and self.video_writer_stream:
                            if len(frame_out.frame_data.shape) == 3:
                                self.video_writer_stream.write(frame_out.frame_data)
                                
                        # Convert frame to bytes and write to FFmpeg
                        frame_bytes = frame_out.frame_data.tobytes()
                        self.ffmpeg_process.stdin.write(frame_bytes)
                        self.ffmpeg_process.stdin.flush()
                        
                    time.sleep(0.001)
                    # Clean up frame resources
                    frame_out.destroy()
                    # Update timestamp
                    self.out_timestamps.append(time.time())
                    # Move to next frame
                    self._write_frame_index += 1
                
            # Calculate output FPS
            now = time.time()
            if now - self.last_fps_update >= 1.0:
                if len(self.out_timestamps) >= 2:
                    time_deltas = [t2 - t1 for t1, t2 in zip(self.out_timestamps, list(self.out_timestamps)[1:])]
                    avg_delta = sum(time_deltas) / len(time_deltas)
                    fps = 1.0 / avg_delta if avg_delta > 0 else 0.0
                    self.logger.update_out_stream_fps(fps)
                self.last_fps_update = now
            
        if self.application == "STREAM":   
            print("Output stream stopped")
            self._cleanup_ffmpeg()
            if self.save_stream:
                self._cleanup_video_saving()

    # ========================================================================
    # Clear cache and exit FFmpeg
    # ========================================================================        
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
            
    # ========================================================================
    # Video saving cleanup routine
    # ========================================================================
    def _cleanup_video_saving(self):
        """
        Release video/audio resources and finalize output.

        This method performs the following:
        - Releases the video writer resource
        - Terminates the background audio extraction process
        - Cleans up any active FFmpeg save process
        - Merges the video-only and audio-only files into a final output
        """
        try:
            # 1. Release the video writer if it was initialized
            if hasattr(self, 'video_writer_stream') and self.video_writer_stream:
                self.video_writer_stream.release()
                self.video_writer_stream = None
            
            # 2. Stop the audio extraction process if it's running
            if hasattr(self, 'audio_process') and self.audio_process:
                try:
                    self.audio_process.terminate()           # Try graceful termination
                    self.audio_process.wait(timeout=10)      # Wait up to 10s for exit
                except subprocess.TimeoutExpired:
                    self.audio_process.kill()                # Force kill if unresponsive
                    self.audio_process.wait(timeout=5)
                except Exception as e:
                    print(f"Error stopping audio extraction: {e}")
                finally:
                    self.audio_process = None
            
            # 3. Clean up FFmpeg save process if one exists
            if hasattr(self, 'ffmpeg_save_process'):
                self._cleanup_ffmpeg_save()
            
            # 4. Merge audio and video files into a final output
            #    (only if both paths are available and the files exist)
            if self.save_stream and hasattr(self, 'video_only_path') and hasattr(self, 'audio_only_path'):
                if os.path.exists(self.video_only_path) or os.path.exists(self.audio_only_path):
                    success = self._merge_video_audio()
                    if success:
                        print("Video and audio merged successfully")
                    else:
                        print("Failed to merge video and audio")
        
        except Exception as e:
            print(f"Error during video saving cleanup: {e}")

    # ========================================================================
    # Stop StreamController
    # ========================================================================
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
        
        if self.save_stream:
            self._cleanup_video_saving()
    
    # ========================================================================
    # Singleton accessor for StreamController
    # ========================================================================
    @classmethod
    def get_instance(cls, cfg=None) -> "StreamController":
        """
        Get the singleton instance of StreamController.
        
        Args:
            cfg (dict, optional): Configuration dictionary. 
                                Required only for first initialization.
        """
        if cls._global_instance is None:
            if cfg is None:
                raise ValueError("StreamController not initialized yet")
            cls._global_instance = StreamController(cfg)
        return cls._global_instance