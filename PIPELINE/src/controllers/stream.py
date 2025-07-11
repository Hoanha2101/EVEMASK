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
        
        while self.running:
            ret, data = self.cap.read()
            frame = Frame(frame_id=self._frame_index, frame_data=data)
            if ret and frame is not None:
                self.circle_queue.add_frame(frame=frame)
                self._frame_index += 1

    def out_stream(self):
        print("Starting output stream...")
        self._start_ffmpeg()
        
        if self.ffmpeg_process is None:
            print("Failed to start FFmpeg")
            return
            
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            if self._write_frame_index in self.circle_queue.frames.keys():
                frame_out = self.circle_queue.get_by_id(self._write_frame_index)
                
                if frame_out is not None:
                    if not frame_out.processed:
                        print("Frame not processed")
                    else:
                        print("Frame processed")
                    frame_bytes = frame_out.frame_data.tobytes()
                    self.ffmpeg_process.stdin.write(frame_bytes)
                    # self.ffmpeg_process.stdin.flush()
                    frame_out.destroy()
                else:
                    time.sleep(0.001)
                
                self._write_frame_index += 1
            else:
                time.sleep(0.001)
                    
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