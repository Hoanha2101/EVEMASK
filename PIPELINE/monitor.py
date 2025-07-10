#!/usr/bin/env python3
"""
Monitoring script for the AI pipeline
Monitors memory usage, CPU usage, and system performance
"""

import psutil
import time
import threading
import os
import sys
from datetime import datetime

class SystemMonitor:
    def __init__(self, log_file="monitor.log"):
        self.log_file = log_file
        self.running = True
        self.monitoring_thread = None
        
    def get_system_info(self):
        """Get current system information"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # GPU memory (if available)
            gpu_info = "N/A"
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_used_gb = gpu_memory.used / (1024**3)
                gpu_total_gb = gpu_memory.total / (1024**3)
                gpu_percent = (gpu_memory.used / gpu_memory.total) * 100
                gpu_info = f"GPU: {gpu_used_gb:.2f}GB/{gpu_total_gb:.2f}GB ({gpu_percent:.1f}%)"
            except:
                pass
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            network_info = f"↑{network.bytes_sent/1024/1024:.1f}MB ↓{network.bytes_recv/1024/1024:.1f}MB"
            
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
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
    
    def log_system_info(self, info):
        """Log system information to file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                if 'error' in info:
                    f.write(f"{info['timestamp']} - ERROR: {info['error']}\n")
                else:
                    f.write(f"{info['timestamp']} - CPU: {info['cpu_percent']:.1f}% | "
                           f"RAM: {info['memory_used_gb']:.2f}GB/{info['memory_total_gb']:.2f}GB ({info['memory_percent']:.1f}%) | "
                           f"{info['gpu_info']} | "
                           f"Disk: {info['disk_percent']:.1f}% | "
                           f"Network: {info['network_info']}\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def print_system_info(self, info):
        """Print system information to console"""
        if 'error' in info:
            print(f"❌ {info['timestamp']} - ERROR: {info['error']}")
        else:
            print(f"📊 {info['timestamp']}")
            print(f"   CPU: {info['cpu_percent']:.1f}%")
            print(f"   RAM: {info['memory_used_gb']:.2f}GB/{info['memory_total_gb']:.2f}GB ({info['memory_percent']:.1f}%)")
            print(f"   {info['gpu_info']}")
            print(f"   Disk: {info['disk_percent']:.1f}%")
            print(f"   Network: {info['network_info']}")
            print("-" * 80)
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                info = self.get_system_info()
                self.log_system_info(info)
                self.print_system_info(info)
                time.sleep(10)  # Monitor every 10 seconds
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def start(self):
        """Start monitoring in a separate thread"""
        self.monitoring_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitoring_thread.start()
        print(f"System monitoring started. Log file: {self.log_file}")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("System monitoring stopped.")

def main():
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "monitor.log"
    
    monitor = SystemMonitor(log_file)
    
    try:
        monitor.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()

if __name__ == "__main__":
    main() 