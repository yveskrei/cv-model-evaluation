from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
import torchvision
import logging
import torch
import json
import os

# Custom modules
import utils.config as config
import utils.object_detection.inference as inference
import utils.object_detection.results as results
from utils.wrapper import Wrapper

# Variables
logger = logging.getLogger(__name__)

def evaluate_model_inference(model: Wrapper, coco_annotation: str, export_results: bool = False):
    """
        Given a model and a COCO dataset, will return results after
        running inference on its own
        returns results in torchmetrics format
    """
    # Load dataset from path
    coco_dataset = inference.load_coco_dataset(coco_annotation)

    # Load results for each image in the dataset
    inference_results = inference.get_torchmetrics_results(
        model=model,
        coco_dataset=coco_dataset,
        dataset_folder=os.path.dirname(coco_annotation)
    )

    # Save results to file
    if export_results:
        with open('inference_results.json', 'w', encoding='utf-8') as file:
            json.dump(results.get_results_json(inference_results), file)
    
    # Run evaluation on results
    return evaluate(inference_results)

def evaluate_results_file(results_path: str):
    """
        Given results file, will return results without running
        any inference
    """
    # Get results from a file
    inference_results = results.get_torchmetrics_results(results_path)

    # Run evaluation on results
    return evaluate(inference_results)

def evaluate(inference_results: list[dict]):
    """
        Evaluates inference results after necessary preprocessing
        all results are in torchmetrics format
    """

    # Split results into target + predictions for easier aggregations
    targets = [result['target'] for result in inference_results]
    predictions = [result['pred'] for result in inference_results]

    # Get Per-Confidence threshold statistics
    per_confidence = get_per_confidence_statistics(targets, predictions)

    return {
        'task': config.TASK_DETECTION,
        'per_confidence': per_confidence
    }

def get_per_confidence_statistics(targets: list[dict], predictions: list[dict]) -> list:
    """
        Itterates over differnet confidence intervals(steps of 0.01)
        returns several object-deterction related metrics
    """
    confidence_results = []

    # Initialize metrics
    metric_map = MeanAveragePrecision()
    metric_map.warn_on_many_detections = False

    for conf_threshold in np.arange(0.01, 1.01, 0.01):
        # Format confidence threshold
        conf_threshold = round(conf_threshold, 2)
        logger.info(f"Processing predictions for confidence threshold {conf_threshold}")

        # Load predictions in torchmetrics format
        filtered_predictions = get_filtered_predictions(predictions, conf_threshold)

        # Update the metric_map for the current threshold
        metric_map.reset()
        metric_map.update(filtered_predictions, targets)
        map_results = metric_map.compute()

        # Compute confusion matrix for IOU threshold of 0.5
        tp, fp, fn = get_confusion_matrix(targets, filtered_predictions)

        # Compute precision, recall, and F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compute amount of bboxes in targets and predictions
        total_targets = sum([len(x['labels']) for x in targets])
        total_predictions = sum([len(x['labels']) for x in filtered_predictions])

        confidence_results.append({
            'conf_threshold': conf_threshold,
            'targets': total_targets,
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
    
    return confidence_results

def get_filtered_predictions(predictions: list[dict], confidence_threshold: float):
    """
        Process all predicitions and filter by confidence interval
        Returns predictions equal or higher than confidence interval
    """
    filtered_predictions = []

    for image_results in predictions:
        boxes = image_results['boxes']
        scores = image_results['scores']
        labels = image_results['labels']

        # Filter on confidence interval
        conf_mask = scores >= confidence_threshold
        boxes_final = boxes[conf_mask]
        scores_final = scores[conf_mask]
        labels_final = labels[conf_mask]

        filtered_predictions.append({
            **image_results,
            'boxes': boxes_final,
            'scores': scores_final,
            'labels': labels_final
        })

    return filtered_predictions

def get_confusion_matrix(targets: list[dict], predictions: list[dict], iou_threshold: float = 0.5):
    """
        Computes the confusion matrix for a given set of predictions and targets.
        both predictions and targets are in torchmetrics format

        returns the amount of true positives, false positives, and false negatives
    """
    tp, fp, fn = 0, 0, 0

    for target, pred in zip(targets, predictions):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        target_boxes = target["boxes"]
        target_labels = target["labels"]

        if target_boxes.size(0) > 0 and pred_boxes.size(0) > 0:
            # Count amount of matches to targets
            matched = torch.zeros(len(target_boxes), dtype=torch.bool)

            # Compute IOUs for each prediction
            pred_ious = torchvision.ops.box_iou(pred_boxes, target_boxes)
            
            for ious, label in zip(pred_ious, pred_labels):
                max_iou, max_idx = ious.max(dim=0)

                if max_iou.item() >= iou_threshold and label == target_labels[max_idx].item() and not matched[max_idx].item():
                    matched[max_idx] = True
                    tp += 1
                else:
                    fp += 1

            fn += target_boxes.size(0) - matched.sum().item()
        elif target_boxes.size(0) > 0 and pred_boxes.size(0) == 0:
            # No predictions for targets
            fn += target_boxes.size(0)
        elif target_boxes.size(0) == 0 and pred_boxes.size(0) > 0:
            # No targets for predictions
            fp += pred_boxes.size(0)

    return tp, fp, fn
