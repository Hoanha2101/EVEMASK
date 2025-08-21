"""
FFmpeg Video Writer Utility
Writes processed video frames to a file while muxing the original audio.

Author: EVEMASK Team
"""

import subprocess

class FFmpegVideoWriter:
    """
    FFmpeg-based file writer that muxes processed video with the original audio.

    This writer accepts raw video frames in BGR24 format via stdin and calls
    FFmpeg to encode them as H.264 while pulling the audio track directly from
    the input source. It is intended for offline VIDEO mode where we process a
    file and need to preserve the source audio without re-encoding video in
    OpenCV. The design aims to minimize frame drops and duration discrepancies.

    Key behaviors and rationale:
    - Synchronization: Uses `-vsync 0` to disable frame duplication/drop so the
      number of output frames matches what we feed via stdin.
    - Audio drift handling: Uses `-af aresample=async=1:first_pts=0` to make the
      audio resampler tolerant to small clock drifts, improving A/V alignment.
    - Audio selection: `-map 1:a?` selects audio from the input source if it
      exists (`?` makes it optional, so run still succeeds if there is no audio).
    - Container duration: We do NOT pass `-shortest` to avoid truncating output
      to the shorter of audio/video; this helps prevent a slightly shorter video.

    Expected input:
    - Frames must be numpy arrays of shape (H, W, 3) in BGR order, dtype=uint8.
    - Frames should be fed at the desired output cadence; this class does not
      rate-limit writes. The caller controls cadence by when it calls `write`.

    Limitations:
    - This class is not thread-safe; write from a single thread.
    - If upstream changes resolution mid-run, a new writer must be created.
    """
    def __init__(self, input_source_path: str, output_path: str, width: int, height: int, fps: float):
        """
        Create and start the FFmpeg subprocess for muxing video and audio.

        Args:
            input_source_path: Path/URL to the original input (used for audio).
            output_path: Destination file path for the processed result.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Target output frames per second (metadata and encoder pacing).
        """
        self.input_source_path = input_source_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self._start()

    def _start(self):
        """
        Build and launch the FFmpeg command.

        Command highlights:
        - `-f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i -` reads raw frames via stdin.
        - Second `-i` is the input source used only for audio.
        - `-map 0:v:0` chooses our encoded video; `-map 1:a?` pulls audio track if present.
        - `-vsync 0` prevents FFmpeg from dropping/duplicating frames.
        - `-af aresample=async=1:first_pts=0` mitigates small A/V drift.
        - H.264 video with yuv420p for broad compatibility; AAC for audio.
        """
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-i", self.input_source_path,
            "-map", "0:v:0",
            "-map", "1:a?",
            "-vsync", "0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "128k",
            "-af", "aresample=async=1:first_pts=0",
            self.output_path,
        ]
        self.process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame_np):
        """
        Write one frame to FFmpeg.

        The caller should call this at the desired cadence. This method is non-blocking
        aside from the underlying pipe write; if FFmpeg back-pressures (rare for local
        files), the OS pipe may briefly block. Exceptions are swallowed to avoid bringing
        down the processing pipeline due to transient IO issues.

        Args:
            frame_np: Numpy uint8 array of shape (H, W, 3) in BGR order.
        """
        if self.process is None or self.process.stdin is None:
            return
        try:
            self.process.stdin.write(frame_np.tobytes())
        except Exception:
            pass

    def release(self):
        """
        Finalize the file: flush stdin, close it, and wait for FFmpeg to exit.

        This ensures the container is properly finalized and all buffered data
        is written. If FFmpeg does not exit within a short timeout, it will be
        force-killed to avoid hanging shutdown logic.
        """
        if self.process is None:
            return
        try:
            if self.process.stdin:
                try:
                    self.process.stdin.flush()
                except Exception:
                    pass
                self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        finally:
            self.process = None