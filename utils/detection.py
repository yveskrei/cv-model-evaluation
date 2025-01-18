from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import logging
import os

# Custom modules
import utils.config as config

# Variables
logger = logging.getLogger(__name__)
IOU_THRESHOLD = 0.5

class Detection:
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, dataset_annotation: str, dataset_name: str):
        """
            Evaluates the model on a given dataset, returning results for each confidence
            threshold between 0 and 1
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
        map_metric = MeanAveragePrecision()
        map_metric.warn_on_many_detections = False

        for conf_threshold in tqdm(np.arange(0.01, 1.01, 0.01), desc="Processing confidence thresholds"):
            # Format confidence threshold
            conf_threshold = round(conf_threshold, 2)

            # Load predictions in torchmetrics format
            filtered_predictions = self.process_predictions(model_predictions, conf_threshold)

            # Update the map_metric for the current threshold
            map_metric.reset()
            map_metric.update(filtered_predictions, annotations)
            map_results = map_metric.compute()

            # Compute confusion matrix for IOU threshold of 0.5
            tp, fp, fn = self.get_confusion_matrix(filtered_predictions, annotations, IOU_THRESHOLD, conf_threshold)

            # Compute precision, recall, and F1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Compute amount of bboxes in annotations and predictions
            total_annotations = sum([len(x['labels']) for x in annotations])
            total_predictions = sum([len(x['labels']) for x in filtered_predictions])

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

        return {
            'dataset': dataset_name,
            'model': os.path.basename(self.model.model_path),
            'results': results
        }
        
    def load_dataset(self, dataset_annotation: str):
        """
            Dataset should be in the following format:
            A folder, including images, with it an annotations.json file, in COCO format.
            returns the base annotations together with the predictions of the model, in COCO format.
        """

        # Validate existance of annotations file
        if not os.path.exists(dataset_annotation):
            raise Exception(f"Annotation file not found at {dataset_annotation}")
        
        # Parse COCO Annotations
        coco_dataset = COCO(dataset_annotation)
        logger.info(f"Loaded dataset with {len(coco_dataset.getImgIds())} images, {len(coco_dataset.getAnnIds())} annotations")

        # Filter dataset annotations to allowed classes
        filtered_annotations = [
            ann for ann in coco_dataset.dataset['annotations']
            if ann['category_id'] in config.SUPPORTED_COCO_CLASSES
        ]
        filtered_categories = [
            cat for cat in coco_dataset.dataset['categories']
            if cat['id'] in config.SUPPORTED_COCO_CLASSES
        ]

        if len(filtered_annotations) == 0:
            raise Exception("No annotations found for supported classes")
        elif len(filtered_annotations) != len(coco_dataset.getAnnIds()):
            logger.warning(f"Filtered dataset to {len(filtered_annotations)} annotations due to unsupported classes")

            # Update dataset with filtered annotations
            coco_dataset.dataset['annotations'] = filtered_annotations
            coco_dataset.dataset['categories'] = filtered_categories
            coco_dataset.createIndex()

        return coco_dataset
    
    def load_predictions(self, coco_dataset: COCO, dataset_folder: str):
        """
            Loads model predictions for each image in the dataset
            returns all predictions in tensormetrics format
        """

        # Load all images in folder
        predictions = []
        for image_id in tqdm(coco_dataset.getImgIds(), desc="Processing images predictions"):
            try:
                image_info = coco_dataset.loadImgs(image_id)[0]
                image_path = os.path.join(dataset_folder, image_info['file_name'])
                image = Image.open(image_path).convert('RGB')
            
                # Process model output   
                image_predictions = self.get_predictions(image)

                # Append predictions
                if len(image_predictions):
                    predictions.append(image_predictions)
                else:
                    logger.warning(f"Image {image_id} - No predictions")

            except Exception as e:
                logger.error(f"Image {image_id} - {e}")

        return predictions
    
    def get_predictions(self, image: Image):
        """
            Returns predictions for a single image, as a list.
            bboxes are converted to original dimensions with format [x_min, y_min, x_max, y_max]
        """
        results = []

        if self.model.model_type in (config.MODEL_YOLO, config.MODEL_DEFAULT):
            # Preprocess image
            image_input = image.resize((640, 640))
            image_array = np.array(image_input).astype(np.float32)
            image_array = image_array / 255.0

            # Ensure the image has 3 channels - If grayscale, convert to 3 channels
            if image_array.ndim == 2:
                image_array = np.stack([image_array] * 3, axis=-1)

            # Transpose to match the expected input shape - from (H, W, C) to (C, H, W)
            image_array = np.transpose(image_array, (2, 0, 1))

            # Add a batch dimension (B, C, H, W) - from (C, H, W) to (1, C, H, W)
            image_array = np.expand_dims(image_array, axis=0)

            # Get model output
            model_output = self.model.predict(image_array)
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

                    # Yolo returns output in COCO classes
                    bbox_class = int(class_id.item()) + 1

                    # Convert bbox to original dimensions
                    image_width, image_height = image.size
                    x_min = (x_min / 640) * image_width
                    y_min = (y_min / 640) * image_height
                    x_max = (x_max / 640) * image_width
                    y_max = (y_max / 640) * image_height

                    results.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'score': score.item(),
                        'class': bbox_class
                    })
        
        return results

    def process_predictions(self, predictions: list, conf_threshold: float):
        """
            Converts model predictions to torchmetrics format
        """
        processed_predictions = []

        for image_predictions in tqdm(predictions, desc=f"Processing predictions for threshold {conf_threshold}"):     
            # Filter out unsupported classes
            image_predictions = list(filter(lambda x: x['class'] in config.SUPPORTED_COCO_CLASSES, image_predictions))
            
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
                    IOU_THRESHOLD
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

    def get_confusion_matrix(self, predictions: list, annotations: list, iou_thresh: float, conf_thresh: float):
        tp, fp, fn = 0, 0, 0

        for pred, annot in zip(predictions, annotations):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]
            annot_boxes = annot["boxes"]
            annot_labels = annot["labels"]

            matched = torch.zeros(len(annot_boxes), dtype=torch.bool)

            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                # Skip low-confidence predictions
                if score < conf_thresh:
                    continue

                ious = self.calculate_iou(box, annot_boxes)
                max_iou, max_idx = ious.max(dim=0)

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