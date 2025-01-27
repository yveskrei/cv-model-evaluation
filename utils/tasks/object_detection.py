from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import logging
import os

# Custom modules
from utils.evaluator import Evaluator
import utils.config as config

# Variables
logger = logging.getLogger(__name__)
IOU_THRESHOLD = 0.5

class ObjectDetection(Evaluator):
    def __init__(self, model):
        self.model = model
        
        if self.model.model_type not in (config.MODEL_YOLO, config.MODEL_DETR):
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
        metric_map = MeanAveragePrecision()
        metric_map.warn_on_many_detections = False

        for conf_threshold in tqdm(np.arange(0.01, 1.01, 0.01), desc="Processing confidence thresholds"):
            # Format confidence threshold
            conf_threshold = round(conf_threshold, 2)

            # Load predictions in torchmetrics format
            filtered_predictions = self.process_predictions(model_predictions, conf_threshold)

            # Update the metric_map for the current threshold
            metric_map.reset()
            metric_map.update(filtered_predictions, annotations)
            map_results = metric_map.compute()

            # Compute confusion matrix for IOU threshold of 0.5
            tp, fp, fn = self.get_confusion_matrix(filtered_predictions, annotations, IOU_THRESHOLD)

            # Compute precision, recall, and F1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Compute amount of bboxes in annotations and predictions
            total_annotations = sum([len(x['labels']) for x in annotations])
            total_predictions = sum([len(x['labels']) for x in filtered_predictions])

            results.append({
                'conf_threshold': conf_threshold,
                'annotations': total_annotations,
                'predictions': total_predictions,
                'map': map_results['map'].item(),
                'map50': map_results['map_50'].item(),
                'map75': map_results['map_75'].item(),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'tp': tp,
                'fp': fp,
                'fn': fn
            })

            logger.info(results[-1])

        return {
            'dataset': dataset_name,
            'model': os.path.basename(self.model.model_path),
            'task': self.model.model_task,
            'results': results
        }
    
    def get_model_predictions(self, image: Image):
        """
            Returns predictions for a single image, as a list.
            removes unsupported classes, bboxes are converted to original dimensions with format [x_min, y_min, x_max, y_max]
        """
        results = []

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
        
                # Split into boxes and class scores
                boxes = predictions[:, :4]
                scores = predictions[:, 4:]
                
                # Convert bboxes to original dimensions
                boxes_xyxy = torch.zeros_like(boxes)
                image_width, image_height = image.size
                scale_x = image_width / 640
                scale_y = image_height / 640

                # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2  # x1
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2  # y1
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2  # x2
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2  # y2

                # Convert bboxes to original size
                boxes_xyxy[:, [0, 2]] *= scale_x
                boxes_xyxy[:, [1, 3]] *= scale_y
                
                # Get max confidence score and class id for each prediction
                max_scores, class_ids = torch.max(scores, dim=1)
                
                # Get final predictions
                for bbox, score, class_id in zip(boxes_xyxy, max_scores, class_ids):
                    x_min, y_min, x_max, y_max = bbox.tolist()

                    # Yolo returns output in COCO classes
                    bbox_class = int(class_id.item()) + 1

                    if bbox_class in config.SUPPORTED_COCO_CLASSES:
                        results.append({
                            'bbox': [x_min, y_min, x_max, y_max],
                            'score': score.item(),
                            'class': bbox_class
                        })

        elif self.model.model_type == config.MODEL_DETR:
            # Resize to model's input size + Convert to tensor (C, H, W) & scale to [0,1]
            transform = transforms.Compose([
                transforms.Resize((800, 1066)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            ])
            image_array = transform(image).unsqueeze(0).numpy()

            # Get model output
            model_output = self.model.predict(image_array)
            output_scores = torch.from_numpy(model_output[0])
            output_boxes = torch.from_numpy(model_output[1])

            for batch in range(output_boxes.shape[0]):
                # Get values
                boxes = output_boxes[batch]
                scores = output_scores[batch]

                # Convert boxes to original dimensions
                boxes_xyxy = torch.zeros_like(boxes)
                resized_height, resized_width = image_array.shape[2], image_array.shape[3]
                image_width, image_height = image.size
                scale_x = image_width / resized_width
                scale_y = image_height / resized_height

                # Denormalize boxes
                boxes_xyxy[:, 0] = boxes[:, 0] * resized_width
                boxes_xyxy[:, 1] = boxes[:, 1] * resized_height
                boxes_xyxy[:, 2] = boxes[:, 2] * resized_width
                boxes_xyxy[:, 3] = boxes[:, 3] * resized_height

                # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
                boxes_xyxy[:, 0] = boxes_xyxy[:, 0] - boxes_xyxy[:, 2]/2  # x1
                boxes_xyxy[:, 1] = boxes_xyxy[:, 1] - boxes_xyxy[:, 3]/2  # y1
                boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]/2  # x2
                boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]/2  # y2

                # Convert bboxes to original size
                boxes_xyxy[:, [0, 2]] *= scale_x
                boxes_xyxy[:, [1, 3]] *= scale_y
                
                # Get max confidence score and class id for each prediction
                max_scores, class_ids = torch.max(scores, dim=1)

                # Get final predictions
                for bbox, score, class_id in zip(boxes_xyxy, max_scores, class_ids):
                    x_min, y_min, x_max, y_max = bbox.tolist()

                    # Yolo returns output in COCO classes
                    bbox_class = int(class_id.item()) + 1

                    if bbox_class in config.SUPPORTED_COCO_CLASSES:
                        results.append({
                            'bbox': [x_min, y_min, x_max, y_max],
                            'score': score.item(),
                            'class': bbox_class
                        })
        return results

    def process_predictions(self, predictions: list, conf_threshold: float):
        """
            Converts raw model predictions to torchmetrics format
            Filters predictions and applies NMS and confidence thresholding
        """
        processed_predictions = []

        for image_predictions in tqdm(predictions, desc=f"Processing predictions for threshold {conf_threshold}"):
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
                boxes_final = boxes[conf_mask]
                scores_final = scores[conf_mask]
                labels_final = labels[conf_mask]

                # Apply NMS for YOLO models
                if self.model.model_type == config.MODEL_YOLO:
                    nms_mask = torchvision.ops.nms(
                        boxes_final,
                        scores_final,
                        IOU_THRESHOLD
                    )

                    # Filter final predictions
                    boxes_final = boxes_final[nms_mask]
                    scores_final = scores_final[nms_mask]
                    labels_final = labels_final[nms_mask]

            processed_predictions.append({
                'boxes': boxes_final.to(torch.float32),
                'scores': scores_final.to(torch.float32),
                'labels': labels_final.to(torch.int64),
            })

        return processed_predictions

    def process_annotations(self, coco_dataset: COCO):
        """
            Converts COCO annotations to torchmetrics format
        """
        annotations = []

        for image_id in coco_dataset.getImgIds():
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
            annotations.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            })
    
        return annotations

    def get_confusion_matrix(self, predictions: list, annotations: list, iou_thresh: float):
        """
            Computes the confusion matrix for a given set of predictions and annotations.
        """
        tp, fp, fn = 0, 0, 0

        for pred, annot in zip(predictions, annotations):
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            annot_boxes = annot["boxes"]
            annot_labels = annot["labels"]

            # Count amount of matches to annotations
            matched = torch.zeros(len(annot_boxes), dtype=torch.bool)

            # Compute IOUs for each prediction
            pred_ious = torchvision.ops.box_iou(pred_boxes, annot_boxes)
            
            for ious, label in zip(pred_ious, pred_labels):
                max_iou, max_idx = ious.max(dim=0)

                if max_iou.item() >= iou_thresh and label == annot_labels[max_idx].item() and not matched[max_idx].item():
                    tp += 1
                    matched[max_idx] = True
                else:
                    fp += 1

            fn += len(annot_boxes) - matched.sum().item()

        return tp, fp, fn