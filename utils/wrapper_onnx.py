
import logging
import os
import onnxruntime as ort

# Custom modules
import utils.config as config

# Variables
logger = logging.getLogger(__name__)

class Wrapper:
    def __init__(self, model_path: str, model_type: str, input_name: str = 'images', output_name: str = 'output'):
        if not os.path.exists(model_path):
            raise Exception(f"Model is not found at {model_path}")
        
        # Specify model parameters
        self.model_type = model_type
        self.model_path = model_path
        self.input_name = input_name
        self.output_name = output_name

        try:
            self.session = self.get_ort_session()

            logger.info(f"Initialized ONNX Model: {self.model_path}")
        except Exception as e:
            logger.error(e)
            raise Exception(f"Failed to initialize ONNX model session")

    def get_ort_session(self):
        providers = [config.PROVIDER_CPU]
        # Check if CUDA is available
        try:
            available_providers = ort.get_available_providers()
            if config.PROVIDER_CUDA in available_providers:
                providers = [config.PROVIDER_CUDA]
        except Exception as e:
            logger.warning(f"CUDA not available: {e}")

        logger.info(f"Using providers: {providers}")

        # Load model using ONNX
        ort_session = ort.InferenceSession(
            self.model_path, 
            providers=providers
        )

        logger.info(f"Loaded model {self.model_path} of type {self.model_type if self.model_type is not None else 'UNKNOWN'}")
        logger.info(f"Running on {'CUDA' if config.PROVIDER_CUDA in providers else 'CPU'}")

        return ort_session

    def predict(self, model_input):
        """
            Runs inferrence on a model with a given input(based on pre-defined input shape)
            returns model output
        """
        # Get model outputs
        model_output = self.session.run(None, {self.input_name: model_input})

        return model_output[0]