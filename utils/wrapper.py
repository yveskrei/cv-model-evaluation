import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import torch
import torchvision

# Model Types
MODEL_YOLO = 'YOLO'

# Device type
PROVIDER_CPU = 'CPUExecutionProvider'
PROVIDER_CUDA = 'CUDAExecutionProvider'

class Wrapper:
    def __init__(self, model_path: str, model_type: str=None):
        if not os.path.exists(model_path):
            raise Exception(f"Model is not found at {model_path}")
        
        # Specify model parameters
        self.model_type = model_type
        self.model_path = model_path
        self.model_providers = self.get_providers()
        self.ort_session = self.load_model()

    def load_model(self):
        if self.model_type is None:
            print(f"Model is not directly supported. Errors may occur")

        # Load model using ONNX
        ort_session = ort.InferenceSession(
            self.model_path, 
            providers=self.model_providers
        )

        print(f"Loaded model {self.model_path} of type {self.model_type if self.model_type is not None else 'UNKNOWN'}")
        print(f"Running on {'CUDA' if PROVIDER_CUDA in self.model_providers else 'CPU'}")

        return ort_session

    def get_providers(self):
        # Check if CUDA is available
        try:
            available_providers = ort.get_available_providers()
            if PROVIDER_CUDA in available_providers:
                return [PROVIDER_CUDA]
        except Exception as e:
            print(f"CUDA not available: {e}")
            
        return [PROVIDER_CPU]
    
    def process_input(self, image: Image):
        """
            Preprocess input image based on model type
            takes PIL image as input
        """
        # Transform image for YOLO
        if self.model_type == MODEL_YOLO:
            image = image.resize((640, 640))

        # Convert the image to a numpy array and normalize it to [0, 1]
        image_array = np.array(image).astype(np.float32)
        image_array = image_array / 255.0

        # Ensure the image has 3 channels
        if image_array.ndim == 2:  # If grayscale, convert to 3 channels
            image_array = np.stack([image_array] * 3, axis=-1)

        # Transpose to match the expected input shape - from (H, W, C) to (C, H, W)
        image_array = np.transpose(image_array, (2, 0, 1))

        # Add a batch dimension (B, C, H, W) - from (C, H, W) to (1, C, H, W)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    def predict(self, image: Image):
        """
            Returns predictions for a single image, as a list.
            bbboxes are converted to original dimensions with format [x_min, y_min, x_max, y_max]
        """
        results = []

        # Preprocess image based on model type
        image_array = self.process_input(image)

        # Get model outputs
        model_output = self.ort_session.run(None, {'input': image_array})

        if self.model_type == MODEL_YOLO:
            output = torch.from_numpy(model_output[0])    

            for batch in output:
                predictions = batch.T  # (8400, 84)
        
                # Split into boxes and class scores
                boxes = predictions[:, :4]
                scores = predictions[:, 4:]
                
                # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
                boxes_xyxy = torch.zeros_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2  # x1
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2  # y1
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2  # x2
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2  # y2
                
                # Get max confidence score and class id for each prediction
                max_scores, class_ids = torch.max(scores, dim=1)
                
                # Get final predictions
                for bbox, score, class_id in zip(boxes_xyxy, max_scores, class_ids):
                    x_min, y_min, x_max, y_max = bbox.tolist()

                    # Convert bbox to original dimensions
                    image_width, image_height = image.size
                    x_min = (x_min / 640) * image_width
                    y_min = (y_min / 640) * image_height
                    x_max = (x_max / 640) * image_width
                    y_max = (y_max / 640) * image_height

                    results.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'score': score.item(),
                        'class': int(class_id.item())
                    })

        return results