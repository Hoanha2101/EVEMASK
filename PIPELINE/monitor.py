#!/usr/bin/env python3
"""
System Monitoring Module
Real-time monitoring of system resources for AI pipeline performance tracking.

This module provides:
- CPU and memory usage monitoring
- GPU memory tracking (if available)
- Disk and network I/O monitoring
- Logging and console output
- Thread-safe monitoring operations

Key Features:
- Continuous monitoring in background thread
- Detailed system resource tracking
- File-based logging with timestamps
- Graceful error handling and recovery
- GPU memory monitoring via NVIDIA Management Library

Usage:
    python monitor.py [log_file_path]
    
Author: EVEMASK Team
"""

import psutil
import time
import threading
import os
import sys
from datetime import datetime
import pynvml

class SystemMonitor:
    """
    System resource monitoring class.
    
    This class provides comprehensive system monitoring capabilities:
    - Real-time resource usage tracking
    - Background monitoring with logging
    - GPU memory monitoring (NVIDIA GPUs)
    - Network and disk I/O tracking
    
    Attributes:
        log_file (str): Path to log file for storing monitoring data
        running (bool): Control flag for monitoring loop
        monitoring_thread (threading.Thread): Background monitoring thread
    """
    
    def __init__(self, log_file="monitor.log"):
        """
        Initialize system monitor.
        
        Args:
            log_file (str): Path to log file for storing monitoring data
        """
        self.log_file = log_file
        self.running = True
        self.monitoring_thread = None
        
    def get_system_info(self):
        """
        Collect comprehensive system information.
        
        This method gathers information about:
        - CPU usage percentage
        - Memory usage (used/total/percentage)
        - GPU memory usage (if NVIDIA GPU available)
        - Disk usage percentage
        - Network I/O statistics
        
        Returns:
            dict: Dictionary containing system information with timestamps
        """
        try:
            # CPU usage measurement (1-second interval for accuracy)
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage information
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)  # Convert to GB
            memory_total_gb = memory.total / (1024**3)  # Convert to GB
            
            # GPU memory monitoring (NVIDIA GPUs only)
            gpu_info = "N/A"
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_used_gb = gpu_memory.used / (1024**3)  # Convert to GB
                gpu_total_gb = gpu_memory.total / (1024**3)  # Convert to GB
                gpu_percent = (gpu_memory.used / gpu_memory.total) * 100
                gpu_info = f"GPU: {gpu_used_gb:.2f}GB/{gpu_total_gb:.2f}GB ({gpu_percent:.1f}%)"
            except:
                # GPU monitoring not available (no NVIDIA GPU or pynvml not installed)
                pass
            
            # Disk usage on root filesystem
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O statistics
            network = psutil.net_io_counters()
            network_info = f"↑{network.bytes_sent/1024/1024:.1f}MB ↓{network.bytes_recv/1024/1024:.1f}MB"
            
            # Return comprehensive system information
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'gpu_info': gpu_info,
                'disk_percent': disk_percent,
                'network_info': network_info
            }
        except Exception as e:
            # Return error information if monitoring fails
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
    
    def log_system_info(self, info):
        """
        Log system information to file.
        
        This method writes formatted system information to the log file
        with timestamps for historical tracking.
        
        Args:
            info (dict): System information dictionary from get_system_info()
        """
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                if 'error' in info:
                    # Log error information
                    f.write(f"{info['timestamp']} - ERROR: {info['error']}\n")
                else:
                    # Log comprehensive system information
                    f.write(f"{info['timestamp']} - CPU: {info['cpu_percent']:.1f}% | "
                           f"RAM: {info['memory_used_gb']:.2f}GB/{info['memory_total_gb']:.2f}GB ({info['memory_percent']:.1f}%) | "
                           f"{info['gpu_info']} | "
                           f"Disk: {info['disk_percent']:.1f}% | "
                           f"Network: {info['network_info']}\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def print_system_info(self, info):
        """
        Print system information to console.
        
        This method displays formatted system information in a readable
        format for real-time monitoring.
        
        Args:
            info (dict): System information dictionary from get_system_info()
        """
        if 'error' in info:
            # Display error information
            print(f"{info['timestamp']} - ERROR: {info['error']}")
        else:
            # Display comprehensive system information
            print(f"{info['timestamp']}")
            print(f"   CPU: {info['cpu_percent']:.1f}%")
            print(f"   RAM: {info['memory_used_gb']:.2f}GB/{info['memory_total_gb']:.2f}GB ({info['memory_percent']:.1f}%)")
            print(f"   {info['gpu_info']}")
            print(f"   Disk: {info['disk_percent']:.1f}%")
            print(f"   Network: {info['network_info']}")
            print("-" * 80)
    
    def monitor_loop(self):
        """
        Main monitoring loop.
        
        This method runs continuously in a background thread and:
        1. Collects system information
        2. Logs data to file
        3. Displays information to console
        4. Handles errors gracefully
        """
        while self.running:
            try:
                # Collect and process system information
                info = self.get_system_info()
                self.log_system_info(info)
                self.print_system_info(info)
                time.sleep(10)  # Monitor every 10 seconds
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Shorter sleep on error for faster recovery
    
    def start(self):
        """
        Start monitoring in a separate thread.
        
        This method creates and starts a daemon thread for background monitoring,
        allowing the main application to continue running while monitoring.
        """
        self.monitoring_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitoring_thread.start()
        print(f"System monitoring started. Log file: {self.log_file}")
    
    def stop(self):
        """
        Stop monitoring gracefully.
        
        This method stops the monitoring loop and waits for the thread
        to finish, ensuring clean shutdown.
        """
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)  # Wait up to 5 seconds
        print("System monitoring stopped.")

def main():
    """
    Main function for standalone monitoring script.
    
    This function handles command line arguments and starts the monitoring
    system. It can be run independently or integrated into other applications.
    """
    # Parse command line arguments for log file path
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "monitor.log"
    
    # Create and start monitoring system
    monitor = SystemMonitor(log_file)
    
    try:
        monitor.start()
        
        # Keep main thread alive while monitoring runs
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        print("\nStopping monitor...")
        monitor.stop()

if __name__ == "__main__":
    main() 