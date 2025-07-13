"""
Logger Module
Real-time logging and display system for EVEMASK video processing pipeline.

This module provides:
- Real-time FPS monitoring for input/output streams and AI processing
- Configuration display and validation feedback
- Progress indicators and loading animations
- ASCII art logo display with color support
- Singleton pattern for global access

Key Features:
- Thread-safe FPS tracking and updates
- Cross-platform screen clearing and cursor control
- Colorized output with fallback to plain text
- Progress bars with customizable duration
- Real-time stream statistics display

Author: EVEMASK Team
"""

import os
import sys
import time
from datetime import datetime

class EveMaskLogger:
    """
    Real-time logger and display system for EVEMASK pipeline.
    
    This class provides comprehensive logging capabilities for the video processing
    pipeline, including FPS monitoring, configuration display, progress indicators,
    and real-time statistics visualization.
    
    Attributes:
        version (str): Current EVEMASK version
        year (int): Current year for copyright display
        in_stream_fps (float): Input stream FPS
        out_stream_fps (float): Output stream FPS
        ai_fps (float): AI processing FPS
        number_out_frames (int): Total output frames processed
        n_skip_frames (int): Number of frames skipped for optimization
    """
    
    _global_instance: "EveMaskLogger" = None
    
    def __init__(self, version: str = "2.0", year: int = datetime.now().year):
        """
        Initialize the EVEMASK logger with version information.
        
        Args:
            version (str): EVEMASK version string
            year (int): Current year for copyright display
        """
        self.version = version
        self.year = datetime.now().year
        
        # Initialize FPS tracking variables
        self.in_stream_fps = 0      # Input stream FPS
        self.out_stream_fps = 0     # Output stream FPS
        self.ai_fps = 0             # AI processing FPS
        
        # Initialize frame counting variables
        self.number_out_frames = 0  # Total output frames processed
        self.n_skip_frames = 0      # Number of frames skipped for optimization
        
    def update_in_stream_fps(self, fps: float):
        """
        Update input stream FPS value.
        
        Args:
            fps (float): Current input stream FPS
        """
        self.in_stream_fps = fps
        
    def update_out_stream_fps(self, fps: float):
        """
        Update output stream FPS value.
        
        Args:
            fps (float): Current output stream FPS
        """
        self.out_stream_fps = fps
        
    def update_ai_fps(self, fps: float):
        """
        Update AI processing FPS value.
        
        Args:
            fps (float): Current AI processing FPS
        """
        self.ai_fps = fps
        
    def update_number_out_frames(self, number: int):
        """
        Update total number of output frames processed.
        
        Args:
            number (int): Total number of output frames
        """
        self.number_out_frames = number
        
    def update_n_skip_frames(self, n: int):
        """
        Update number of frames skipped for optimization.
        
        Args:
            n (int): Number of frames skipped
        """
        self.n_skip_frames = n
        
    def show_config(self, cfg: dict):
        """
        Display configuration information with visual indicators.
        
        This method prints the loaded configuration in a user-friendly format
        with emoji indicators for different configuration categories.
        
        Args:
            cfg (dict): Configuration dictionary containing pipeline settings
        """
        print("✅ Configuration loaded")
        print(f"📥 Input source : {cfg.get('INPUT_SOURCE', 'Not specified')}")
        print(f"📤 Output type  : {cfg.get('OUTPUT_TYPE', 'Not specified')}")
        print(f"📦 Batch size   : {cfg.get('batch_size', 'Not specified')}")
        print(f"🎯 Target FPS   : {cfg.get('TARGET_FPS', 'Not specified')}")
        print("✅ All components initialized successfully")

    def waiting_bar(self, cfg: dict):
        """
        Display a progress bar during initialization.
        
        This method creates an animated progress bar to indicate initialization
        progress, typically used when setting up output streams.
        
        Args:
            cfg (dict): Configuration dictionary containing delay settings
        """
        # Get delay duration from configuration
        duration = cfg.get('DELAY_TIME', 0)
        
        # Initialize progress bar display
        sys.stdout.write("\n[⏳] Initializing output stream: ")
        sys.stdout.flush()
        
        # Progress bar parameters
        steps = 30  # Number of progress bar segments
        
        # Animate progress bar
        for i in range(steps + 1):
            # Create progress bar visualization
            bar = '█' * i + '-' * (steps - i)
            percent = int((i / steps) * 100)
            
            # Update progress bar display
            sys.stdout.write(f"\r[⏳] Initializing output stream: |{bar}| {percent}%")
            sys.stdout.flush()
            
            # Sleep for proportional duration
            time.sleep(duration / steps)
        
        print()  # new line after complete
        
    def display_stream(self):
        """
        Display real-time stream statistics in a formatted box.
        
        This method clears the screen and displays current FPS and frame
        statistics in a visually appealing ASCII box format.
        """
        # Clear screen and move cursor to top-left
        sys.stdout.write("\033[2J\033[H")
        
        # Display statistics in formatted box
        print("╔══════════════════════════════════════╗")
        print(f"║       🚀 EVEMASK STREAM LOGGER       ║")
        print("╠══════════════════════════════════════╣")
        print(f"║ 🎥 Input Stream FPS     : {self.in_stream_fps:6.0f}     ║")
        print(f"║ 📤 Output Stream FPS    : {self.out_stream_fps:6.0f}     ║")
        print(f"║ 🧠 AI Processing FPS    : {self.ai_fps:6.0f}     ║")
        print(f"║ 🖼️  Output Frames Count  : {self.number_out_frames:6d}     ║")
        print(f"║ 🕳️  Skipped Frames Count : {self.n_skip_frames:6d}     ║")
        print("╚══════════════════════════════════════╝")
        sys.stdout.flush()
    
    def display_logo(self):
        """
        Display EVEMASK logo and startup information.
        
        This method clears the screen and displays the EVEMASK ASCII art logo
        with version information and startup messages. It attempts to use
        colorized output if colorama is available, otherwise falls back to
        plain text.
        """
        # ASCII art logo with version information
        logo = f"""
        ╔════════════════════════════════════════════════════════════════════════════════════╗
        ║                                                                                    ║
        ║            ███████╗██╗   ██╗███████╗███╗   ███╗ █████╗ ███████╗██╗  ██╗            ║
        ║            ██╔════╝██║   ██║██╔════╝████╗ ████║██╔══██╗██╔════╝██║ ██╔╝            ║
        ║            █████╗  ██║   ██║█████╗  ██╔████╔██║███████║███████╗█████╔╝             ║
        ║            ██╔══╝  ╚██╗ ██╔╝██╔══╝  ██║╚██╔╝██║██╔══██║╚════██║██╔═██╗             ║
        ║            ███████╗ ╚████╔╝ ███████╗██║ ╚═╝ ██║██║  ██║███████║██║  ██╗            ║
        ║            ╚══════╝  ╚═══╝  ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝            ║
        ║                                                                                    ║
        ║                                                                                    ║
        ║                ╔══════════════════════════════════════════════════╗                ║
        ║                ║                   EVEMASK v{self.version}                   ║                ║
        ║                ║        Real-time AI Video Processing System      ║                ║
        ║                ╚══════════════════════════════════════════════════╝                ║
        ║                                                                                    ║
        ║            🚀High-Performance • 🎯Intelligent • ⚡Real-time • 🔒Secure             ║
        ║                                                                                    ║
        ║    Developed by: EVEMASK Team                                                      ║
        ║    Version: {self.version} | {self.year}                                                             ║
        ║                                                                                    ║
        ╚════════════════════════════════════════════════════════════════════════════════════╝
        """
        
        # Clear screen (works on Windows and Unix-like systems)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Try to use colorized output with colorama
        try:
            from colorama import init, Fore, Back, Style
            init(autoreset=True)
            
            # Print logo with colors
            print(Fore.CYAN + logo)
            print(Fore.YELLOW + f"🚀 Starting EVEMASK Pipeline {self.version}...")
            print(Fore.GREEN + "=" * 80)
            print()
            
        except ImportError:
            # Fallback to plain text if colorama is not available
            print(logo)
            print(f"🚀 Starting EVEMASK Pipeline {self.version}...")
            print("=" * 80)
            print()
            
    @classmethod
    def get_instance(cls) -> "EveMaskLogger":
        """
        Get singleton instance of EveMaskLogger.
        
        This method ensures only one instance of EveMaskLogger exists
        throughout the application, providing global access to the logger.
        
        Returns:
            EveMaskLogger: Singleton instance
        """
        if cls._global_instance is None:
            cls._global_instance = EveMaskLogger()
        return cls._global_instance