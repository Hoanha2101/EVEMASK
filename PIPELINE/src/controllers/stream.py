from .circle_queue import CircleQueue
from .frame import Frame
import cv2
import time
import subprocess
import threading
import numpy as np


class StreamController:
    def __init__(self, cfg):
        self.circle_queue = CircleQueue.get_instance()
        from ..brain.AI import AI
        self.ai_instance = AI.get_instance()
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
        self._frame_times = []  # Store frame timestamps for FPS calculation
        self._last_fps_calc = time.time()
        
        # Initialize capture
        self._init_capture()

    def _init_capture(self):
        print(f"Waiting for input stream: {self.INPUT_SOURCE}")
        while True:
            try:
                cap = cv2.VideoCapture(self.INPUT_SOURCE)
                # Optimize buffer settings
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
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
            time.sleep(1)

    def _start_ffmpeg(self):
        delay = time.time() - self.begin_time if self.begin_time else 0
        
        ffmpeg_command = [
            "ffmpeg",
            "-re",  # Real time input
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", "{}x{}".format(self.width, self.height),
            "-r", str(self.target_fps),
            "-i", "-",
            "-i", self.INPUT_SOURCE,
            "-af", f"adelay={delay * 1000}|{delay * 1000}",
            "-async", "1",
            "-vsync", "1",
            "-q:v", "1",
            "-map", "1:a",
            "-map", "0:v",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",  # Use ultrafast for better performance
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-vf", "yadif",
            "-f",
        ]
        
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
                "mpegts", self.cfg['OUTPUT_STREAM_URL_UDP']
            ])
            output_stream_url = self.cfg['OUTPUT_STREAM_URL_UDP']
            
        print(f"Output URL: {output_stream_url}")
        
        try:
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
        """Tính toán input FPS dựa trên thời gian nhận frame"""
        current_time = time.time()
        self._frame_times.append(current_time)
        
        # Chỉ tính FPS mỗi 3 giây để tránh dao động
        if current_time - self._last_fps_calc > 3.0:
            if len(self._frame_times) > 1:
                # Tính FPS dựa trên số frame trong khoảng thời gian
                time_span = self._frame_times[-1] - self._frame_times[0]
                if time_span > 0:
                    input_fps = (len(self._frame_times) - 1) / time_span
                    return input_fps
            
            # Reset tracking
            self._frame_times = [current_time]
            self._last_fps_calc = current_time
        
        return None

    def source_capture(self):
        print("Starting source capture...")
        
        while self.running:
            try:
                ret, data = self.cap.read()
                if ret and data is not None:
                    frame = Frame(frame_id=self._frame_index, frame_data=data)
                    self.circle_queue.add_frame(frame=frame)
                    self._frame_index += 1
                    
                    # Tính toán và cập nhật input FPS
                    input_fps = self._calculate_input_fps()
                    if input_fps is not None:
                        # Cập nhật AI FPS nếu có thể
                        try:     
                            if self.ai_instance:
                                self.ai_instance.update_input_fps(input_fps)
                        except Exception as e:
                            print(f"Error updating AI FPS: {e}")
                else:
                    time.sleep(0.01)  # Short sleep if no frame
            except Exception as e:
                print(f"Error in capture: {e}")
                time.sleep(0.1)

    def out_stream(self):
        print("Starting output stream...")
        self._start_ffmpeg()
        
        if self.ffmpeg_process is None:
            print("Failed to start FFmpeg")
            return
            
        while self.running:
            # start = time.time()
            if self._write_frame_index in self.circle_queue.frames.keys():
                frame_out = self.circle_queue.get_by_id(self._write_frame_index)
                if frame_out is not None:
                    # if frame_out.processed:
                    #     print("AI has processed")
                    # else:
                    #     print("AI has not processed")
                    frame_bytes = frame_out.frame_data.tobytes()
                    self.ffmpeg_process.stdin.write(frame_bytes)
                    frame_out.destroy()
                self._write_frame_index += 1
                # print("lenght circle:", self.circle_queue.queue_length())
                # sl = max(0,1/self.target_fps - time.time() + start - 0.001)
                # time.sleep(sl)
                time.sleep(0.01)
            else:
                time.sleep(0.1)  # Reduced sleep time for better responsiveness
                    
        print("Output stream stopped")
        self._cleanup_ffmpeg()

    def _cleanup_ffmpeg(self):
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

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self._cleanup_ffmpeg() 