
import logging
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically manages CUDA context
import numpy as np

# Variables
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
logger = logging.getLogger(__name__)

class Wrapper:
    def __init__(self, model_path: str, model_type: str):
        if not os.path.exists(model_path):
            raise Exception(f"Model is not found at {model_path}")
        
        # Specify model parameters
        self.model_type = model_type
        self.model_path = model_path

        # Start inference session
        try:
            self.engine = self.get_engine()
            self.context = self.engine.create_execution_context()
            self.buffers = self.allocate_buffers(self.engine)

            logger.info(f"Initialized TensorRT Model: {self.model_path}")
        except Exception as e:
            logger.error(e)
            raise Exception(f"Failed to initialize TensorRT model session")

    def get_engine(self):
        with open(self.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine_data = f.read()
        
        return runtime.deserialize_cuda_engine(engine_data)
    
    def allocate_buffers(self, engine):
            input_tensor_name = None
            output_tensor_name = None

            stream = cuda.Stream()
            tensors = [None] * engine.num_io_tensors

            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
 
                if shape[0] == -1:
                    logger.info(f"Dynamic shape detected for tensor {name}. Changing first dim to maximum size.")

                    try:
                        profile_shapes = engine.get_tensor_profile_shape(name, 0)
                        max_shape = tuple(profile_shapes[2])

                        # Append first dimension to shape
                        shape[0] = max_shape[0] if max_shape[0] > 0 else 1
                    except:
                        shape[0] = 1  # Default to 1 if dynamic shape cannot be determined

                dtype = trt.nptype(engine.get_tensor_dtype(name))
                size = trt.volume(shape)

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                tensors[i] = int(device_mem)

                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    input_tensor_name = name
                    input_host, input_device = host_mem, device_mem
                else:
                    output_tensor_name = name
                    output_host, output_device = host_mem, device_mem

            return {
                "input_name": input_tensor_name,
                "output_name": output_tensor_name,
                "input_host": input_host,
                "input_device": input_device,
                "output_host": output_host,
                "output_device": output_device,
                "tensors": tensors,
                "stream": stream
            }

    def predict(self, model_input: np.ndarray) -> np.ndarray:
        # Ensure input is float32, flattened, and contiguous
        input_array = np.ascontiguousarray(model_input.astype(np.float32))

        # Optional: set dynamic input shape
        self.context.set_input_shape(self.buffers["input_name"], input_array.shape)

        # Copy input data
        self.buffers["input_host"][:input_array.size] = input_array.ravel()
        cuda.memcpy_htod_async(self.buffers["input_device"], self.buffers["input_host"], self.buffers["stream"])

        # Inference
        # Set tensor device addresses before execution
        for i, device_mem_ptr in enumerate(self.buffers["tensors"]):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, device_mem_ptr)

        self.context.execute_async_v3(self.buffers["stream"].handle)

        # Copy output data
        cuda.memcpy_dtoh_async(self.buffers["output_host"], self.buffers["output_device"], self.buffers["stream"])
        self.buffers["stream"].synchronize()

        return self.buffers["output_host"]
