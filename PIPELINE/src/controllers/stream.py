from .circle_queue import CircleQueue
from .frame import Frame
import cv2
import time
from ..brain import AI
import subprocess
import threading
import numpy as np
from collections import OrderedDict

class StreamController:
    def __init__(self, cfg):
        self.circle_queue = CircleQueue.get_instance()
        self.cfg = cfg
        self.INPUT_SOURCE = cfg['INPUT_SOURCE']
        self.target_fps = cfg['TARGET_FPS']
        self.width = None
        self.height = None
        self.running = True
        self.ffmpeg_process = None
        self.cap = None
        self._frame_index = 0
        self._write_frame_index = 0
        
        # Thêm lock để đồng bộ hóa
        self.lock = threading.Lock()
        
        self._init_capture()

    def _init_capture(self):
        while True:
            try:
                cap = cv2.VideoCapture(self.INPUT_SOURCE)
                # Thêm timeout và buffer size config
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Input stream: {self.INPUT_SOURCE} is ready")
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
                    
            print(f"[{time.time()}] Waiting for input stream: {self.INPUT_SOURCE}")
            time.sleep(1)  # Tăng delay để tránh spam

    def _restart_ffmpeg(self):
        with self.lock:
            if self.ffmpeg_process is not None:
                try:
                    self.ffmpeg_process.stdin.close()
                    self.ffmpeg_process.wait(timeout=5)
                except Exception as e:
                    print(f"Error closing ffmpeg: {e}")
                    try:
                        self.ffmpeg_process.kill()
                    except:
                        pass
                self.ffmpeg_process = None
            self._start_ffmpeg()

    def _start_ffmpeg(self):

        
        delay = time.time() - self.begin_time if self.begin_time else 0
        
        ffmpeg_command = [
            "ffmpeg",
            "-re",  # Real time input
            "-f",
            "rawvideo",  # Input format
            "-pix_fmt",
            "bgr24",  # Pixel format
            "-s",
            "{}x{}".format(self.width, self.height),  # Size of one frame
            "-r",
            str(self.target_fps),  # Frame rate
            "-i",
            "-",  # The input comes from a pipe
            "-i",
            self.INPUT_SOURCE,
            "-af",
            f"adelay={delay * 1000}|{delay * 1000}",
            "-async",
            "1",
            "-vsync",
            "1",
            "-q:v",
            "1",
            "-map",
            "1:a",
            "-map",
            "0:v",
            "-c:v",
            "libx264",  # Codec
            "-pix_fmt",
            "yuv420p",  # Pixel format for output
            "-preset",
            "medium",  # Encoding speed medium
            "-color_primaries",
            "bt709",  # Color primaries as per BT.709
            "-color_trc",
            "bt709",  # Transfer characteristics as per BT.709
            "-colorspace",
            "bt709",  # Color space as per BT.709
            "-vf",
            "yadif",
            "-f",
        ]
        
        # Đơn giản hóa audio handling
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
        else:
            ffmpeg_command.extend([
                "mpegts",
                self.cfg['OUTPUT_STREAM_URL_UDP']
            ])
            output_stream_url = self.cfg['OUTPUT_STREAM_URL_UDP']
            
        print(f"Output URL: {output_stream_url}")
        print(f"FFmpeg command: {' '.join(ffmpeg_command)}")
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_command, 
                stdin=subprocess.PIPE
            )
            print("FFmpeg process started successfully")
        except Exception as e:
            print(f"Error starting FFmpeg: {e}")
            self.ffmpeg_process = None

    def source_capture(self):
        print("Starting source capture...")
        frame_time = 1.0 / self.target_fps
        
        while self.running:
            try:
                start_time = time.time()
                ret, data = self.cap.read()
                
                if ret and data is not None:
                    frame = Frame(frame_id=self._frame_index, frame_data=data)
                    self.circle_queue.add_frame(frame=frame)
                    print(self.circle_queue.queue_length())
                    self._frame_index += 1
                    
                    # Frame rate control
                    elapsed = time.time() - start_time
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                else:
                    print("Failed to read frame, attempting to reconnect...")
                    self._init_capture()
                    
            except Exception as e:
                print(f"Error in source capture: {e}")
                time.sleep(0.1)

    def out_stream(self):
        print("Starting output stream...")
        self._start_ffmpeg()
        
        if self.ffmpeg_process is None:
            print("Failed to start FFmpeg")
            return
            
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                frame_out = self.circle_queue.get_by_id(self._write_frame_index)
                
                if frame_out and frame_out.processed:
                    # Kiểm tra FFmpeg process
                    if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
                        print("FFmpeg process died. Restarting...")
                        self._restart_ffmpeg()
                        if self.ffmpeg_process is None:
                            consecutive_failures += 1
                            if consecutive_failures >= max_failures:
                                print("Max FFmpeg restart failures reached")
                                break
                            time.sleep(1)
                            continue
                    
                    try:
                        frame_bytes = frame_out.frame_data.tobytes()
                        self.ffmpeg_process.stdin.write(frame_bytes)
                        self.ffmpeg_process.stdin.flush()
                        
                        # Cleanup
                        self.circle_queue.remove_by_id(self._write_frame_index)
                        self._write_frame_index += 1
                        frame_out.destroy()
                        
                        consecutive_failures = 0  # Reset counter on success
                        
                    except (BrokenPipeError, OSError) as e:
                        print(f"FFmpeg pipe error: {e}")
                        self._restart_ffmpeg()
                        consecutive_failures += 1
                        
                elif frame_out and not frame_out.processed:
                    # Frame chưa được xử lý, chờ
                    time.sleep(0.005)
                    
                else:
                    # Không có frame, chờ
                    time.sleep(0.005)
                    
            except Exception as e:
                print(f"Error in output stream: {e}")
                time.sleep(0.1)
                
        print("Output stream stopped")
        self._cleanup_ffmpeg()

    def _cleanup_ffmpeg(self):
        with self.lock:
            if self.ffmpeg_process is not None:
                try:
                    self.ffmpeg_process.stdin.close()
                    self.ffmpeg_process.wait(timeout=5)
                except Exception:
                    try:
                        self.ffmpeg_process.kill()
                        self.ffmpeg_process.wait(timeout=2)
                    except:
                        pass
                self.ffmpeg_process = None

    def stop(self):
        print("Stopping stream controller...")
        self.running = False
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            
        self._cleanup_ffmpeg()
        print("Stream controller stopped")