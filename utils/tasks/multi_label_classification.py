from torchmetrics.classification import MultilabelAccuracy
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import logging
import os

# Custom modules
from utils.evaluator import Evaluator
import utils.config as config

# Variables
logger = logging.getLogger(__name__)

class MultiLabelClassification(Evaluator):
    def __init__(self, model):
        self.model = model
        
        if self.model.model_type not in (config.MODEL_YOLO):
            raise Exception(f"Model type {self.model.model_type} is not supported for object detection")
    
    def evaluate(self, dataset_annotation: str, dataset_name: str):
        """
            Evaluates the model on a given dataset, for each confidence threshold(0.01 steps)
            returns a dictionary with the results
        """
        # Load dataset from path
        coco_dataset = self.load_dataset(dataset_annotation)

        # Get model predictions for each image from the dataset
        model_predictions = self.load_predictions(
            coco_dataset=coco_dataset,
            dataset_folder=os.path.dirname(dataset_annotation)
        )

        # Load annotations in torchmetrics format
        annotations = self.process_annotations(coco_dataset)
        results = []

        # Initialize metrics
        metric_accuracy = MultilabelAccuracy(num_labels=len(config.SUPPORTED_COCO_CLASSES))

        for conf_threshold in tqdm(np.arange(0.01, 1.01, 0.01), desc="Processing confidence thresholds"):
            # Format confidence threshold
            conf_threshold = round(conf_threshold, 2)

            # Load predictions in torchmetrics format
            filtered_predictions = self.process_predictions(model_predictions, conf_threshold)

            # Update the metric_map for the current threshold
            accuracy = metric_accuracy(filtered_predictions, annotations)

            results.append({
                'conf_threshold': conf_threshold,
                'accuracy': accuracy.item()
            })

            logger.info(f"Confidence threshold: {conf_threshold}, Accuracy: {accuracy.item()}")
        
        return {
            'dataset': dataset_name,
            'model': os.path.basename(self.model.model_path),
            'task': self.model.model_task,
            'results': results
        }
    
    def get_model_predictions(self, image: Image):
        """
            Returns predictions for a single image, as a list.
            returns a list of scores for each class, 0 if class is not present
        """
        max_classes_scores = {}

        if self.model.model_type == config.MODEL_YOLO:
            # Resize to model's input size + Convert to tensor (C, H, W) & scale to [0,1]
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor()
            ])
            image_array = transform(image).unsqueeze(0).numpy()

            # Get model output
            model_output = self.model.predict(image_array)
            output = torch.from_numpy(model_output[0]) 

            for batch in output:
                predictions = batch.T  # (8400, 84)
        
                # Get prediction scores
                scores = predictions[:, 4:]
                
                # Get max confidence score and class id for each prediction
                max_scores, class_ids = torch.max(scores, dim=1)
                
                # Extract the max score for each class
                for score, class_id in zip(max_scores, class_ids):
                    # Yolo returns output in COCO classes
                    bbox_class = int(class_id.item()) + 1

                    if bbox_class in max_classes_scores:
                        max_classes_scores[bbox_class] = max(max_classes_scores[bbox_class], score.item())
                    else:
                        max_classes_scores[bbox_class] = score.item()
                
        # Create predictions for each image
        results = [max_classes_scores.get(label, 0) for label in config.SUPPORTED_COCO_CLASSES]
        
        return results
    
    def process_predictions(self, predictions: list, conf_threshold: float):
        """
            Converts raw model predictions to torchmetrics format
            Filters predictions and applies NMS and confidence thresholding
        """
        processed_predictions = []

        for image_predictions in tqdm(predictions, desc=f"Processing predictions for threshold {conf_threshold}"):     
            # Apply confidence threshold
            image_predictions = [0 if score < conf_threshold else score for score in image_predictions]

            # Append predictions
            processed_predictions.append(image_predictions)
        
        # Convert predictions to torch tensor
        processed_predictions = torch.tensor(processed_predictions, dtype=torch.float32)

        return processed_predictions
    
    def process_annotations(self, coco_dataset: COCO):
        """
            Converts COCO annotations to torchmetrics format
        """
        annotations = []

        for image_id in coco_dataset.getImgIds():
            ann_ids = coco_dataset.getAnnIds(imgIds=image_id)
            anns = coco_dataset.loadAnns(ann_ids)

            # Extract labels for each image
            existing_classes = list(set([ann['category_id'] for ann in anns]))
            results = [1 if label in existing_classes else 0 for label in config.SUPPORTED_COCO_CLASSES]
                
            # Append target for this image
            annotations.append(results)
        
        # Convert annotations to torch tensor
        annotations = torch.tensor(annotations, dtype=torch.float32)
    
        return annotations

