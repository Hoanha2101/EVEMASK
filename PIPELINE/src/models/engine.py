"""
TensorRT engine implementation for the EVEMASK Pipeline system.
Provides TensorRT-based inference engines for segmentation and feature extraction models.
Handles CUDA context management, memory allocation, and efficient GPU inference.

Author: EVEMASK Team
Version: 1.0.0
"""

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import numpy as np
import atexit
import weakref

# ========================================================================
# CUDA MEMORY MANAGEMENT
# ========================================================================
# Clear CUDA cache to free up memory
torch.cuda.empty_cache()

# Global context manager to track all CUDA contexts for proper cleanup
_active_contexts = weakref.WeakSet()

def cleanup_all_contexts():
    """
    Cleanup all active CUDA contexts at program exit.
    Ensures proper resource deallocation to prevent memory leaks.
    """
    for ctx in list(_active_contexts):
        try:
            if hasattr(ctx, 'cuda_ctx') and ctx.cuda_ctx:
                ctx.cuda_ctx.pop()
                ctx.cuda_ctx = None
        except:
            pass

# Register cleanup function to run at program exit
atexit.register(cleanup_all_contexts)

# ========================================================================
# HOST-DEVICE MEMORY CLASS
# ========================================================================
class HostDeviceMem(object):
    """
    Container for host and device memory pairs used in TensorRT inference.
    Manages both CPU (host) and GPU (device) memory buffers.
    """
    def __init__(self, host_mem, device_mem) -> None:
        """
        Initialize host-device memory pair.
        Args:
            host_mem: CPU memory buffer
            device_mem: GPU memory buffer
        """
        self.host = host_mem
        self.device = device_mem
    
    def __str__(self) -> str:
        """String representation showing both host and device memory."""
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    
    def __repr__(self):
        """Detailed string representation."""
        return self.__str__()

# ========================================================================
# TENSORRT BASE ENGINE CLASS (SEGMENTATION)
# ========================================================================
class TensorrtBase:
    """
    Base TensorRT engine class for segmentation model inference.
    Handles dynamic shape inference, memory management, and CUDA context management.
    """
    def __init__(self, engine_file_path, input_names, output_names, *, gpu_id=0, dynamic_factor=2, max_batch_size=1, getTo="cpu") -> None:
        """
        Initialize TensorRT engine for segmentation model.
        Args:
            engine_file_path: Path to TensorRT engine file
            input_names: List of input tensor names
            output_names: List of output tensor names
            gpu_id: GPU device ID to use
            dynamic_factor: Memory allocation factor for dynamic shapes
            max_batch_size: Maximum batch size for inference
            getTo: Output destination ("cpu" or "cuda")
        """
        self.input_names = input_names
        self.output_names = output_names
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.cuda_ctx = None
        self.max_batch_size = max_batch_size
        self.getTo = getTo
        
        # Initialize CUDA context for this engine
        self._init_cuda_context(gpu_id)
        
        # Add to global context tracker for cleanup
        _active_contexts.add(self)
        
        # Load TensorRT engine and create execution context
        self.engine = self._load_engine(engine_file_path)
        self.binding_names = self.input_names + self.output_names
        self.context = self.engine.create_execution_context()
        self.buffers = self._allocate_buffer(dynamic_factor)
    
    def _init_cuda_context(self, gpu_id):
        """
        Initialize CUDA context with proper error handling.
        Args:
            gpu_id: GPU device ID to create context on
        """
        try:
            device = cuda.Device(gpu_id)
            self.cuda_ctx = device.make_context()
        except cuda.Error as e:
            print(f"Failed to create CUDA context: {e}")
            raise
        
    def _load_engine(self, engine_file_path):
        """
        Load TensorRT engine from file.
        Args:
            engine_file_path: Path to the TensorRT engine file
        Returns:
            TensorRT engine object
        """
        # Force initialization of TensorRT plugins
        trt.init_libnvinfer_plugins(None, '')
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(self.trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
    
    def _allocate_buffer(self, dynamic_factor):
        """
        Allocate host and device buffers for TensorRT inference.
        Args:
            dynamic_factor: Memory allocation factor for dynamic shapes
        Returns:
            tuple: (inputs, outputs, bindings, stream)
        """
        inputs = []
        outputs = []
        bindings = [None] * len(self.binding_names)
        stream = cuda.Stream()
        
        for i, binding in enumerate(self.binding_names):
            binding_idx = self.engine.get_binding_index(binding)
            if binding_idx == -1:
                print(f"Binding '{binding}' not found!")
                continue
            
            # Get binding shape - handle dynamic shapes properly
            binding_shape = self.engine.get_binding_shape(binding_idx)
            
            # Calculate buffer size safely
            if -1 in binding_shape:
                # Dynamic shape - use reasonable defaults
                if self.engine.binding_is_input(binding):
                    size = 1 * 640 * 640 * 3  # Input size for segmentation
                else:
                    # For outputs, allocate larger buffer for safety
                    size = 1000000  # 1M elements as safe default
            else:
                size = int(np.prod(binding_shape)) * self.max_batch_size
            
            # Apply dynamic factor for additional memory
            size *= dynamic_factor
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            # Allocate host and device buffers with error handling
            try:
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            except cuda.MemoryError as e:
                print(f"Memory allocation failed for {binding}: {e}")
                raise
            
            # Store device buffer pointer in bindings
            bindings[binding_idx] = int(device_mem)
            
            # Categorize as input or output buffer
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
    def infer(self, input_np):
        """
        Perform inference with the TensorRT engine.
        Args:
            input_np: Input numpy array
        Returns:
            List of output arrays or GPU pointers depending on getTo setting
        """
        # Ensure input is contiguous in memory
        if not input_np.flags['C_CONTIGUOUS']:
            input_np = np.ascontiguousarray(input_np)
            print("Made input contiguous")
        
        # Set dynamic input shape for the engine
        input_binding_idx = self.engine.get_binding_index(self.input_names[0])
        self.context.set_binding_shape(input_binding_idx, input_np.shape)
        
        # Validate context state after setting shape
        if not self.context.all_binding_shapes_specified:
            print("Not all binding shapes are specified")
            return None
        
        if not self.context.all_shape_inputs_specified:
            print("Not all shape inputs are specified")
            return None
        
        # Get current buffer configuration
        inputs, outputs, bindings, stream = self.buffers
        
        # Get actual output shapes after setting input shape
        output_shapes = []
        for i, output_name in enumerate(self.output_names):
            output_idx = self.engine.get_binding_index(output_name)
            if output_idx != -1:
                shape = self.context.get_binding_shape(output_idx)
                output_shapes.append(shape)
            else:
                print(f"Output binding {output_name} not found")
                return None
        
        # Reallocate output buffers if needed for dynamic shapes
        for i, (out, expected_shape) in enumerate(zip(outputs, output_shapes)):
            expected_size = int(np.prod(expected_shape))
            if out.host.size < expected_size:
                dtype = out.host.dtype
                
                # Free old memory
                out.device.free()
                
                # Allocate new memory with padding for safety
                new_size = int(expected_size * 1.2)  # 20% padding
                new_host = cuda.pagelocked_empty(new_size, dtype)
                new_device = cuda.mem_alloc(new_host.nbytes)
                outputs[i] = HostDeviceMem(new_host, new_device)
                
                # Update binding pointer
                output_idx = self.engine.get_binding_index(self.output_names[i])
                bindings[output_idx] = int(new_device)
        
        # Validate input size against allocated buffer
        input_size = int(np.prod(input_np.shape))
        if input_size > inputs[0].host.size:
            print(f"Input too large: {input_size} > {inputs[0].host.size}")
            return None
        
        # Handle CUDA output (return GPU pointers)
        if self.getTo == "cuda":
            try:
                # Flatten input array and copy to host buffer
                input_flat = input_np.ravel()
                np.copyto(inputs[0].host[:input_flat.size], input_flat)
                
                # Copy data from host to device asynchronously
                cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
                
                # Execute inference
                success = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                if not success:
                    print("Inference execution failed")
                    return None
                
                # Synchronize stream to ensure completion
                stream.synchronize()
                
                # Return GPU device pointers for further processing
                gpu_results = []
                for i, (out, shape) in enumerate(zip(outputs, output_shapes)):
                    gpu_results.append({
                        'device_ptr': out.device,  # GPU pointer
                        'shape': shape,
                        'size': int(np.prod(shape)),
                        'dtype': out.host.dtype
                    })
                
                return gpu_results
            except cuda.Error as e:
                print(f"CUDA error during inference: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error during inference: {e}")
                return None
            
        # Handle CPU output (return numpy arrays)
        if self.getTo == "cpu":
            try:
                # Flatten input array and copy to host buffer
                input_flat = input_np.ravel()
                np.copyto(inputs[0].host[:input_flat.size], input_flat)
                
                # Copy data from host to device asynchronously
                cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
                
                # Execute inference
                success = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                if not success:
                    print("Inference execution failed")
                    return None
                
                # Copy outputs back to host memory
                for out in outputs:
                    cuda.memcpy_dtoh_async(out.host, out.device, stream)
                
                # Synchronize stream to ensure completion
                stream.synchronize()
                
                # Reshape results to proper dimensions
                results = []
                for i, (out, shape) in enumerate(zip(outputs, output_shapes)):
                    result_size = int(np.prod(shape))
                    result = out.host[:result_size].reshape(shape)
                    results.append(result)
                
                return results
            
            except cuda.Error as e:
                print(f"CPU error during inference: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error during inference: {e}")
                return None
    
    def cleanup(self):
        """
        Explicit cleanup method to free GPU memory and CUDA context.
        Should be called when the engine is no longer needed.
        """
        try:
            if hasattr(self, 'buffers'):
                inputs, outputs, bindings, stream = self.buffers
                # Free device memory for all buffers
                for inp in inputs:
                    if hasattr(inp, 'device'):
                        inp.device.free()
                for out in outputs:
                    if hasattr(out, 'device'):
                        out.device.free()
            
            # Pop CUDA context
            if hasattr(self, 'cuda_ctx') and self.cuda_ctx:
                self.cuda_ctx.pop()
                self.cuda_ctx = None
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup when object is garbage collected."""
        self.cleanup()

# ========================================================================
# TENSORRT M2 ENGINE CLASS (FEATURE EXTRACTION)
# ========================================================================
class TensorrtBase_M2:
    """
    TensorRT engine class for feature extraction model inference.
    Optimized for batch processing of feature vectors with fixed input size.
    """
    def __init__(self, engine_file_path, input_names, output_names, *, gpu_id=0, dynamic_factor=1, max_batch_size=32, lenEmb=256) -> None:
        """
        Initialize TensorRT engine for feature extraction model.
        Args:
            engine_file_path: Path to TensorRT engine file
            input_names: List of input tensor names
            output_names: List of output tensor names
            gpu_id: GPU device ID to use
            dynamic_factor: Memory allocation factor
            max_batch_size: Maximum batch size for inference
            lenEmb: Length of embedding vectors
        """
        self.input_names = input_names
        self.output_names = output_names
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.cuda_ctx = None
        self.max_batch_size = max_batch_size
        self.lenEmb = lenEmb
        
        # Initialize CUDA context for this engine
        self._init_cuda_context(gpu_id)
        
        # Add to global context tracker for cleanup
        _active_contexts.add(self)
        
        # Load TensorRT engine and create execution context
        self.engine = self._load_engine(engine_file_path)
        self.binding_names = self.input_names + self.output_names
        self.context = self.engine.create_execution_context()
        self.buffers = self._allocate_buffer(dynamic_factor)
    
    def _init_cuda_context(self, gpu_id):
        """
        Initialize CUDA context with proper error handling.
        Args:
            gpu_id: GPU device ID to create context on
        """
        try:
            device = cuda.Device(gpu_id)
            self.cuda_ctx = device.make_context()
        except cuda.Error as e:
            print(f"Failed to create CUDA context: {e}")
            raise
    
    def _load_engine(self, engine_file_path):
        """
        Load TensorRT engine from file.
        Args:
            engine_file_path: Path to the TensorRT engine file
        Returns:
            TensorRT engine object
        """
        # Initialize TensorRT plugins
        trt.init_libnvinfer_plugins(None, '')
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(self.trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
        
    def _allocate_buffer(self, dynamic_factor):
        """
        Allocate host and device buffers for feature extraction inference.
        Args:
            dynamic_factor: Memory allocation factor
        Returns:
            tuple: (inputs, outputs, bindings, stream)
        """
        inputs = []
        outputs = []
        bindings = [None] * len(self.binding_names)
        stream = cuda.Stream()
        
        for binding in self.binding_names:
            binding_idx = self.engine.get_binding_index(binding)
            if binding_idx == -1:
                print(f"Binding '{binding}' not found!")
                continue
            
            # Calculate buffer size based on binding type
            binding_shape = self.engine.get_binding_shape(binding_idx)
            if -1 in binding_shape:
                # Dynamic shape - use expected sizes
                if self.engine.binding_is_input(binding_idx):
                    size = abs(trt.volume([self.max_batch_size, 3, 224, 224]))  # Input size
                else:
                    size = self.max_batch_size * self.lenEmb  # Output size
            else:
                size = abs(trt.volume(binding_shape)) * self.max_batch_size * dynamic_factor
            
            # Allocate memory
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings[binding_idx] = int(device_mem)
            
            # Categorize as input or output
            if self.engine.binding_is_input(binding_idx):
                inputs.append(HostDeviceMem_M2(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem_M2(host_mem, device_mem))
        return inputs, outputs, bindings, stream
    
    def infer(self, tensor: torch.Tensor):
        """
        Perform inference with torch tensor input.
        Args:
            tensor: Input torch tensor (must be on CUDA and float16)
        Returns:
            torch.Tensor: Output feature vectors
        """
        # Validate input tensor requirements
        assert tensor.is_cuda, "Tensor must be on CUDA"
        assert tensor.dtype == torch.float16, "Tensor must be float16"
        
        # Push CUDA context to current thread
        self.cuda_ctx.push()
        
        try:
            inputs, outputs, bindings, stream = self.buffers
            shape = tuple(tensor.shape)
            
            # Clear output buffers
            cuda.memset_d32(outputs[0].device, 0, outputs[0].host.nbytes // 4)
            
            # Set optimization profile and input shape
            self.context.set_optimization_profile_async(0, stream.handle)
            input_binding_name = self.input_names[0]
            self.context.set_input_shape(input_binding_name, shape)
            
            # Copy input tensor to TensorRT buffer
            src_device_ptr = tensor.data_ptr()
            dst_device_ptr = inputs[0].device
            tensor_size_bytes = tensor.numel() * tensor.element_size()
            cuda.memcpy_dtod_async(dst_device_ptr, src_device_ptr, tensor_size_bytes, stream)
            
            # Execute inference
            self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            
            # Create output tensor and copy results
            output_shape = (shape[0], self.lenEmb)
            output_tensor = torch.empty(output_shape, dtype=torch.float16, device='cuda')
            
            # Copy results from TensorRT buffer to output tensor
            cuda.memcpy_dtod_async(
                dest=output_tensor.data_ptr(),
                src=outputs[0].device,
                size=output_tensor.numel() * output_tensor.element_size(),
                stream=stream
            )
            
            # Synchronize stream to ensure completion
            stream.synchronize()
            return output_tensor
            
        finally:
            # Always pop context to restore previous state
            self.cuda_ctx.pop()

    def cleanup(self):
        """
        Explicit cleanup method to free GPU memory and CUDA context.
        Should be called when the engine is no longer needed.
        """
        try:
            if hasattr(self, 'buffers'):
                inputs, outputs, bindings, stream = self.buffers
                # Free device memory for all buffers
                for inp in inputs:
                    if hasattr(inp, 'device'):
                        inp.device.free()
                for out in outputs:
                    if hasattr(out, 'device'):
                        out.device.free()
            
            # Pop CUDA context
            if hasattr(self, 'cuda_ctx') and self.cuda_ctx:
                self.cuda_ctx.pop()
                self.cuda_ctx = None
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup when object is garbage collected."""
        self.cleanup()

# ========================================================================
# HOST-DEVICE MEMORY CLASS FOR M2
# ========================================================================
class HostDeviceMem_M2:
    """
    Container for host and device memory pairs used in M2 TensorRT inference.
    Similar to HostDeviceMem but specifically for feature extraction model.
    """
    def __init__(self, host_mem, device_mem):
        """
        Initialize host-device memory pair.
        Args:
            host_mem: CPU memory buffer
            device_mem: GPU memory buffer
        """
        self.host = host_mem
        self.device = device_mem
    
    def __str__(self) -> str:
        """String representation showing both host and device memory."""
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    
    def __repr__(self):
        """Detailed string representation."""
        return self.__str__()