import onnxruntime as ort
import logging
import os

# Custom modules
from utils.detection import Detection
import utils.config as config

# Variables
logger = logging.getLogger(__name__)

class Wrapper:
    def __init__(self, model_path: str, model_task: str = None, model_type: str = config.MODEL_DEFAULT):
        if not os.path.exists(model_path):
            raise Exception(f"Model is not found at {model_path}")
        if model_task is None:
            raise Exception('Model task is required')
        if model_type is None:
            raise Exception('Model type is required')
        
        # Specify model parameters
        self.model_type = model_type
        self.model_path = model_path
        self.model_task = model_task
        
        # Create ONNX session for prediction
        self.ort_session = self.get_ort_session()

        # Define model evaluator
        if model_task == config.TASK_DETECTION:
            self.evaluator = Detection(self)

    def get_ort_session(self):
        # Determine ONNX runtime providers
        providers = self.get_providers()

        # Load model using ONNX
        ort_session = ort.InferenceSession(
            self.model_path, 
            providers=providers
        )

        logger.info(f"Loaded model {self.model_path} of type {self.model_type if self.model_type is not None else 'UNKNOWN'}")
        logger.info(f"Running on {'CUDA' if config.PROVIDER_CUDA in providers else 'CPU'}")

        return ort_session

    def get_providers(self):
        # Check if CUDA is available
        try:
            available_providers = ort.get_available_providers()
            if config.PROVIDER_CUDA in available_providers:
                return [config.PROVIDER_CUDA]
        except Exception as e:
            logger.warning(f"CUDA not available: {e}")
            
        return [config.PROVIDER_CPU]

    def predict(self, model_input):
        """
            Runs inferrence on a model with a given input(based on pre-defined input shape)
            returns model output
        """
        # Get model outputs
        model_output = self.ort_session.run(None, {'input': model_input})

        return model_output