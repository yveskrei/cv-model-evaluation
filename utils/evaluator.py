from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import Precision, Recall

# Custom modules
import utils.wrapper as wrapper

# Object class mappings
SUPPORTED_COCO_CLASSES = [1, 3, 4] # Person, Car, Motorcycle
YOLO_TO_COCO = {
    0: 1, # Person
    2: 3, # Car
    3: 4, # Motorcycle
}   

class Evaluator:
    def __init__(self, model: wrapper.Wrapper):
        self.model = model
    
    def evaluate(self, annotation_path: str):
        """
            Evaluates the model on a given dataset, returning results for each confidence
            threshold between 0 and 1
        """
        # Load dataset from path
        coco_dataset = self.load_dataset(annotation_path)

        # Get model predictions for each image from the dataset
        model_predictions = self.load_predictions(annotation_path, coco_dataset)

        # Load annotations in torchmetrics format
        annotations = self.process_annotations(coco_dataset)
        results = []

        # Initialize metrics
        map_metric = MeanAveragePrecision()

        for conf_threshold in tqdm(np.arange(0.50, 1.00, 0.05), desc="Processing confidence thresholds"):
            # Load predictions in torchmetrics format
            predictions = self.process_predictions(model_predictions, conf_threshold)

            # Update the map_metric for the current threshold
            map_metric.reset()
            map_metric.update(predictions, annotations)
            map_results = map_metric.compute()

            print(f"Confidence Threshold: {conf_threshold:.2f}, mAP: {map_results['map'].item()}")

        return results
        
    def load_dataset(self, annotation_path: str):
        """
            Dataset should be in the following format:
            A folder, including images, with it an annotations.json file, in COCO format.
            returns the base annotations together with the predictions of the model, in COCO format.
        """

        # Validate existance of annotations file
        if not os.path.exists(annotation_path):
            raise Exception(f"Annotation file not found at {annotation_path}")
        
        # Parse COCO Annotations
        coco_dataset = COCO(annotation_path)
        print(f"Loaded dataset with {len(coco_dataset.getImgIds())} images, {len(coco_dataset.getAnnIds())} annotations")

        # Filter dataset annotations to allowed classes
        filtered_annotations = [
            ann for ann in coco_dataset.dataset['annotations']
            if ann['category_id'] in SUPPORTED_COCO_CLASSES
        ]
        filtered_categories = [
            cat for cat in coco_dataset.dataset['categories']
            if cat['id'] in SUPPORTED_COCO_CLASSES
        ]

        if len(filtered_annotations) == 0:
            raise Exception("No annotations found for supported classes")
        elif len(filtered_annotations) != len(coco_dataset.getAnnIds()):
            print(f"Filtered dataset to {len(filtered_annotations)} annotations due to unsupported classes")

            # Update dataset with filtered annotations
            coco_dataset.dataset['annotations'] = filtered_annotations
            coco_dataset.dataset['categories'] = filtered_categories
            coco_dataset.createIndex()

        return coco_dataset
    
    def load_predictions(self, annotation_path: str, coco_dataset: COCO):
        """
            Loads model predictions for each image in the dataset
            returns all predictions in tensormetrics format
        """

        # Load all images in folder
        predictions = []
        for image_id in tqdm(coco_dataset.getImgIds(), desc="Processing images predictions"):
            try:
                image_info = coco_dataset.loadImgs(image_id)[0]
                image_path = os.path.join(os.path.dirname(annotation_path), image_info["file_name"])
                image = Image.open(image_path).convert("RGB")
                
                # Get image predictions
                image_predictions = self.model.predict(image)
                predictions.append(image_predictions)

            except Exception as e:
                print(f"Image {image_id} Error: {e}")

        return predictions
    
    def process_annotations(self, coco_dataset: COCO):
        """
            Converts COCO annotations to torchmetrics format
        """
        targets = []

        for image_id in tqdm(coco_dataset.getImgIds(), desc="Processing images annotations"):
            ann_ids = coco_dataset.getAnnIds(imgIds=image_id)
            anns = coco_dataset.loadAnns(ann_ids)

            # Extract boxes and labels for this image
            boxes = []
            labels = []
            for ann in anns:
                # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
                x_min, y_min, width, height = ann['bbox']

                # Calculate x_max and y_max
                x_max = x_min + width
                y_max = y_min + height

                # Append values
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])

            # Append target for this image
            targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            })
    
        return targets

    def process_predictions(self, predictions: list, conf_threshold: float, nms_iou_threshold: float = 0.4):
        """
            Converts model predictions to torchmetrics format
        """
        processed_predictions = []

        for image_predictions in tqdm(predictions, desc=f"Processing predictions for threshold {conf_threshold}"):
            # Map all classes to coco classes
            if self.model.model_type == wrapper.MODEL_YOLO:
                image_predictions = [{**x, 'class': YOLO_TO_COCO.get(x['class'])} for x in image_predictions]
            
            # Filter out unsupported classes
            image_predictions = [x for x in image_predictions if x['class'] in SUPPORTED_COCO_CLASSES]
            
            # Process predictions
            boxes_final = torch.tensor([])
            scores_final = torch.tensor([])
            labels_final = torch.tensor([])

            if len(image_predictions):
                # Convert predictions to tensor
                image_predictions = [[*x['bbox'], x['score'], x['class']] for x in image_predictions]
                image_predictions = torch.tensor(image_predictions)

                # Set values
                boxes = image_predictions[:, :4]
                scores = image_predictions[:, 4]
                labels = image_predictions[:, 5]

                # Filter by confidence threshold
                conf_mask = scores >= conf_threshold
                boxes_filtered = boxes[conf_mask]
                scores_filtered = scores[conf_mask]
                labels_filtered = labels[conf_mask]

                # Apply NMS
                nms_mask = torchvision.ops.nms(
                    boxes_filtered,
                    scores_filtered,
                    nms_iou_threshold
                )

                # Filte final predictions
                boxes_final = boxes_filtered[nms_mask]
                scores_final = scores_filtered[nms_mask]
                labels_final = labels_filtered[nms_mask]

            processed_predictions.append({
                'boxes': boxes_final.to(torch.float32),
                'scores': scores_final.to(torch.float32),
                'labels': labels_final.to(torch.int64),
            })


        return processed_predictions