from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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

        for conf_threshold in tqdm(np.arange(0.05, 1.00, 0.01), desc="Processing confidence thresholds"):
            # Format confidence threshold
            conf_threshold = round(conf_threshold, 2)

            # Load predictions in torchmetrics format
            predictions = self.process_predictions(model_predictions, conf_threshold)

            # Update the map_metric for the current threshold
            map_metric.reset()
            map_metric.update(predictions, annotations)
            map_results = map_metric.compute()

            # Compute confusion matrix for IOU threshold of 0.5
            tp, fp, fn = self.get_confusion_matrix(predictions, annotations, 0.5, conf_threshold)

            # Compute precision, recall, and F1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Compute amount of bboxes in annotations and predictions
            total_annotations = sum([len(x['labels']) for x in annotations])
            total_predictions = sum([len(x['labels']) for x in predictions])

            results.append({
                'conf_threshold': conf_threshold,
                'map': map_results['map'].item(),
                'map50': map_results['map_50'].item(),
                'map75': map_results['map_75'].item(),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'annotations': total_annotations,
                'predictions': total_predictions,
                'tp': tp,
                'fp': fp,
                'fn': fn
            })

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

    def get_confusion_matrix(self, predictions: list, annotations: list, iou_thresh: float, conf_thresh: float):
        tp, fp, fn = 0, 0, 0

        for pred, annot in zip(predictions, annotations):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]
            annot_boxes = annot["boxes"]
            annot_labels = annot["labels"]

            matched = torch.zeros(len(annot_boxes), dtype=torch.bool)  # Track matched targets

            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if score < conf_thresh:
                    continue  # Skip low-confidence predictions

                ious = self.calculate_iou(box, annot_boxes)
                max_iou, max_idx = ious.max(0)

                if max_iou.item() >= iou_thresh and label == annot_labels[max_idx].item() and not matched[max_idx].item():
                    tp += 1
                    matched[max_idx] = True
                else:
                    fp += 1

            fn += len(annot_boxes) - matched.sum().item()

        return tp, fp, fn

    # IoU calculation function
    @staticmethod
    def calculate_iou(bbox: torch.Tensor, target_bboxes: torch.Tensor):
        """
            Calculate the Intersection over Union (IoU) between a single bounding box and target bounding boxes.
        """
        # Ensure the input bbox is of the correct shape
        bbox = bbox.view(-1, 4)
        
        # Compute the intersection coordinates
        x1_inter = torch.max(bbox[:, 0], target_bboxes[:, 0])
        y1_inter = torch.max(bbox[:, 1], target_bboxes[:, 1])
        x2_inter = torch.min(bbox[:, 2], target_bboxes[:, 2])
        y2_inter = torch.min(bbox[:, 3], target_bboxes[:, 3])
        
        # Compute intersection area
        intersection = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
        
        # Compute areas of each box
        bbox_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        target_areas = (target_bboxes[:, 2] - target_bboxes[:, 0]) * (target_bboxes[:, 3] - target_bboxes[:, 1])
        
        # Compute union area
        union = bbox_area + target_areas - intersection
        
        # Compute IoU
        iou = intersection / torch.clamp(union, min=1e-6)
        
        return iou