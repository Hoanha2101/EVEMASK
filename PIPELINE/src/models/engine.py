import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import numpy as np
import atexit

# Clear CUDA cache
torch.cuda.empty_cache()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem) -> None:
        self.host = host_mem
        self.device = device_mem
    
    def __str__(self) -> str:
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    
    def __repr__(self):
        return self.__str__()

class TensorrtBase:
    def __init__(self, engine_file_path, input_names, output_names, *, gpu_id=0, dynamic_factor=2, max_batch_size=1, getTo = "cpu") -> None:
        self.input_names = input_names
        self.output_names = output_names
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.cuda_ctx = cuda.Device(gpu_id).make_context()
        self.max_batch_size = max_batch_size
        self.engine = self._load_engine(engine_file_path)
        self.binding_names = self.input_names + self.output_names
        self.context = self.engine.create_execution_context()
        self.buffers = self._allocate_buffer(dynamic_factor)
        self.getTo = getTo
        
    def _load_engine(self, engine_file_path):
        # Force init TensorRT plugins
        trt.init_libnvinfer_plugins(None, '')
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(self.trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
    
    def _allocate_buffer(self, dynamic_factor):
        """Allocate buffer with proper error checking"""
        inputs = []
        outputs = []
        bindings = [None] * len(self.binding_names)
        stream = cuda.Stream()
        
        for i, binding in enumerate(self.binding_names):
            binding_idx = self.engine.get_binding_index(binding)
            if binding_idx == -1:
                print(f"❌ Binding '{binding}' not found!")
                continue
            
            # Get binding shape - handle dynamic shapes properly
            binding_shape = self.engine.get_binding_shape(binding_idx)
            
            # Calculate size more safely
            if -1 in binding_shape:
                # Dynamic shape - use a reasonable default
                if self.engine.binding_is_input(binding):
                    size = 1 * 640 * 640 * 3  # Input size
                else:
                    # For outputs, allocate larger buffer
                    size = 1000000  # 1M elements as safe default
            else:
                size = int(np.prod(binding_shape)) * self.max_batch_size
            
            size *= dynamic_factor
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            # Allocate host and device buffers
            try:
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            except cuda.MemoryError as e:
                print(f"❌ Memory allocation failed for {binding}: {e}")
                raise
            
            # Append the device buffer to device bindings
            bindings[binding_idx] = int(device_mem)
            
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
    def infer(self, input_np):
        # Validate input
        if not input_np.flags['C_CONTIGUOUS']:
            input_np = np.ascontiguousarray(input_np)
            print("Made input contiguous")
        
        # Set dynamic input shape
        input_binding_idx = self.engine.get_binding_index(self.input_names[0])
        self.context.set_binding_shape(input_binding_idx, input_np.shape)
        
        # Check if context is valid after setting shape
        if not self.context.all_binding_shapes_specified:
            print("❌ Not all binding shapes are specified")
            return None
        
        if not self.context.all_shape_inputs_specified:
            print("❌ Not all shape inputs are specified")
            return None
        
        # Get current buffers
        inputs, outputs, bindings, stream = self.buffers
        
        # Get actual output shapes after setting input shape
        output_shapes = []
        for i, output_name in enumerate(self.output_names):
            output_idx = self.engine.get_binding_index(output_name)
            if output_idx != -1:
                shape = self.context.get_binding_shape(output_idx)
                output_shapes.append(shape)
            else:
                print(f"❌ Output binding {output_name} not found")
                return None
        
        # Reallocate output buffers if needed
        for i, (out, expected_shape) in enumerate(zip(outputs, output_shapes)):
            expected_size = int(np.prod(expected_shape))
            if out.host.size < expected_size:
                dtype = out.host.dtype
                
                # Free old memory
                out.device.free()
                
                # Allocate new memory with some padding
                new_size = int(expected_size * 1.2)  # 20% padding
                new_host = cuda.pagelocked_empty(new_size, dtype)
                new_device = cuda.mem_alloc(new_host.nbytes)
                outputs[i] = HostDeviceMem(new_host, new_device)
                
                # Update binding
                output_idx = self.engine.get_binding_index(self.output_names[i])
                bindings[output_idx] = int(new_device)
        
        # Validate input size
        input_size = int(np.prod(input_np.shape))
        if input_size > inputs[0].host.size:
            print(f"❌ Input too large: {input_size} > {inputs[0].host.size}")
            return None
        if self.getTo == "cuda":
            try:
                # Flatten the input NumPy array to 1D.
                # ravel() returns a flattened view when possible (no data copy unless necessary).
                input_flat = input_np.ravel()

                # Copy the flattened input data into the host buffer (CPU memory)
                # allocated earlier for TensorRT. Only copy as many elements as needed.
                np.copyto(inputs[0].host[:input_flat.size], input_flat)

                # Asynchronously copy data from host (CPU RAM) to device (GPU VRAM)
                # using the specified CUDA stream to enable parallel execution.
                cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)

                
                # Run inference
                success = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                if not success:
                    print("❌ Inference execution failed")
                    return None
                
                # Synchronize stream để đảm bảo inference hoàn thành
                stream.synchronize()
                
                # Trả về GPU device pointers thay vì copy về CPU
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
                print(f"❌ CUDA error during inference: {e}")
                return None
            except Exception as e:
                print(f"❌ Unexpected error during inference: {e}")
                return None
            
        if self.getTo == "cpu":
            try:
                # Flatten the input NumPy array to 1D.
                # ravel() returns a flattened view when possible (no data copy unless necessary).
                input_flat = input_np.ravel()

                # Copy the flattened input data into the host buffer (CPU memory)
                # allocated earlier for TensorRT. Only copy as many elements as needed.
                np.copyto(inputs[0].host[:input_flat.size], input_flat)

                # Asynchronously copy data from host (CPU RAM) to device (GPU VRAM)
                # using the specified CUDA stream to enable parallel execution.
                cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)

                
                # Run inference
                success = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                if not success:
                    print("❌ Inference execution failed")
                    return None
                
                # Copy outputs back to host
                for out in outputs:
                    cuda.memcpy_dtoh_async(out.host, out.device, stream)
                
                # Synchronize stream
                stream.synchronize()
                
                # Return reshaped results
                results = []
                for i, (out, shape) in enumerate(zip(outputs, output_shapes)):
                    result_size = int(np.prod(shape))
                    result = out.host[:result_size].reshape(shape)
                    results.append(result)
                
                return results
            
            except cuda.Error as e:
                print(f"❌ CPU error during inference: {e}")
                return None
            except Exception as e:
                print(f"❌ Unexpected error during inference: {e}")
                return None
    
    def __del__(self):
        try:
            if hasattr(self, 'buffers'):
                inputs, outputs, bindings, stream = self.buffers
                # Free device memory
                for inp in inputs:
                    if hasattr(inp, 'device'):
                        inp.device.free()
                for out in outputs:
                    if hasattr(out, 'device'):
                        out.device.free()
            
            if hasattr(self, 'cuda_ctx'):
                self.cuda_ctx.pop()
                del self.cuda_ctx
        except:
            pass  # Ignore cleanup errors 

class TensorrtBase_M2:
    def __init__(self, engine_file_path, input_names, output_names, *, gpu_id=0, dynamic_factor=1, max_batch_size=32, lenEmb=256) -> None:
        self.input_names = input_names
        self.output_names = output_names
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.cuda_ctx = cuda.Device(gpu_id).make_context()
        atexit.register(self.cleanup_context)
        self.max_batch_size = max_batch_size
        self.lenEmb = lenEmb  # Gán lenEmb trước khi gọi _allocate_buffer
        self.engine = self._load_engine(engine_file_path)
        self.binding_names = self.input_names + self.output_names
        self.context = self.engine.create_execution_context()
        self.buffers = self._allocate_buffer(dynamic_factor)
    
    def _load_engine(self, engine_file_path):
        trt.init_libnvinfer_plugins(None, '')
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(self.trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
        
    def _allocate_buffer(self, dynamic_factor):
        inputs = []
        outputs = []
        bindings = [None] * len(self.binding_names)
        stream = cuda.Stream()
        
        for binding in self.binding_names:
            binding_idx = self.engine[binding]
            if binding_idx == -1:
                print(f"❌ Binding '{binding}' not found!")
                continue
            
            binding_shape = self.engine.get_binding_shape(binding)
            if -1 in binding_shape:
                size = abs(trt.volume([self.max_batch_size, 3, 224, 224])) if self.engine.binding_is_input(binding) else self.max_batch_size * self.lenEmb
            else:
                size = abs(trt.volume(binding_shape)) * self.max_batch_size * dynamic_factor
            
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings[binding_idx] = int(device_mem)
            
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem_M2(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem_M2(host_mem, device_mem))
        return inputs, outputs, bindings, stream
    
    def infer(self, tensor: torch.Tensor):
        assert tensor.is_cuda, "Tensor must be on CUDA"
        assert tensor.dtype == torch.float16, "Tensor must be float16"
        
        self.cuda_ctx.push()

        inputs, outputs, bindings, stream = self.buffers
        shape = tuple(tensor.shape)
        
        # Clear buffers để đảm bảo tính nhất quán
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
        
        cuda.memcpy_dtod_async(
            dest=output_tensor.data_ptr(),
            src=outputs[0].device,
            size=output_tensor.numel() * output_tensor.element_size(),
            stream=stream
        )
        
        stream.synchronize()
        self.cuda_ctx.pop()
        return output_tensor
            

    def __del__(self):
        try:
            if hasattr(self, 'cuda_ctx'):
                self.cuda_ctx.pop()
                del self.cuda_ctx
        except:
            pass
        
    def cleanup_context(self):
        try:
            self.cuda_ctx.pop()
            del self.cuda_ctx
        except:
            pass

class HostDeviceMem_M2:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    
    def __str__(self) -> str:
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    
    def __repr__(self):
        return self.__str__()