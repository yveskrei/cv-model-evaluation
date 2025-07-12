from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
import torchvision
import logging
import torch
import json
import os

# Custom modules
import utils.object_detection.inference as inference
import utils.object_detection.results as results

# Variables
logger = logging.getLogger(__name__)

def evaluate_model_inference(model, coco_annotation: str):
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

    # Run evaluation on results
    return results.get_results_json(inference_results), evaluate(inference_results)

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

    # Get all possible classes
    all_classes = set()
    target_classes = [target['labels'] for target in targets]
    prediction_classes = [pred['labels'] for pred in predictions]
    for labels in target_classes + prediction_classes:
        all_classes.update(labels.cpu().numpy().tolist())

    map_stats = get_map_statistics(targets, predictions)

    # Get Per-Confidence threshold statistics
    per_confidence = get_per_confidence_statistics(targets, predictions, all_classes)

    return {
        'overall': {
            'total_targets': sum([len(x['labels']) for x in targets]),
            'total_predictions': sum([len(x['labels']) for x in predictions]),
            'total_images': len(targets)
        },
        'per_confidence': per_confidence,
        'map': map_stats,
    }

def get_map_statistics(targets: list[dict], predictions: list[dict]) -> dict:
    """
        Calculates mean average precision for different IOU thresholds
        Calculated for whole dataset, and per class
    """
    map_metrics = {}

    # Calcualte map for different IOU thresholds
    for iou_threshold in [0.5, 0.75, None]:
        # Initialize the metric
        metric_map = MeanAveragePrecision(
            iou_thresholds=[iou_threshold] if iou_threshold is not None else None, 
            class_metrics=True
        )
        metric_map.warn_on_many_detections = False
        metric_map.update(predictions, targets)
        map_results = metric_map.compute()

        # Print information
        iou_threshold_name = str(int(iou_threshold * 100)) if iou_threshold is not None else '50:95'
        iou_threshold_key = f"map_{iou_threshold_name}"
        logger.info(f"Calculating mAP for IOU threshold {iou_threshold_name}")

        # Initiate threshold metrics
        map_metrics[iou_threshold_key] = {}

        # Add overall mAP results
        map_metrics[iou_threshold_key]['overall'] = map_results['map'].item()

        # Add per-class mAP results
        map_per_class = map_results['map_per_class'].tolist()
        map_classes = map_results['classes'].tolist()
        for idx, cls in enumerate(map_classes):
            map_metrics[iou_threshold_key][cls] = map_per_class[idx]

    return map_metrics

def get_per_confidence_statistics(targets: list[dict], predictions: list[dict], classes: list) -> list:
    """
        Itterates over differnet confidence intervals(steps of 0.01)
        returns several object-deterction related metrics
    """
    confidence_results = []

    for conf_threshold in np.arange(0.00, 1.01, 0.01):
        # Format confidence threshold
        conf_threshold = round(conf_threshold, 2)
        logger.info(f"Processing predictions for confidence threshold {conf_threshold:.2f}")

        # Compute amount of bboxes in predictions
        threshold_stats = {}
        total_predictions = 0

        # Calculate per class metrics
        for cls in classes:
            # Load predictions in torchmetrics format
            filtered_predictions = get_filtered_predictions(predictions, conf_threshold, cls)
            filtered_targets = get_filtered_targets(targets, cls)
            total_predictions += sum([len(x['labels']) for x in filtered_predictions])

            # Compute confusion matrix for IOU threshold of 0.5
            tp, fp, fn = get_confusion_matrix(filtered_targets, filtered_predictions)

            # Compute precision, recall, and F1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            threshold_stats[cls] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        # Calculate overall metrics
        for cls in classes:
            threshold_stats['macro'] = {
                'precision': np.mean([threshold_stats[cls]['precision'] for cls in classes]),
                'recall': np.mean([threshold_stats[cls]['recall'] for cls in classes]),
                'f1_score': np.mean([threshold_stats[cls]['f1_score'] for cls in classes]),
            }

        confidence_results.append({
            'confidence_threshold': conf_threshold,
            'total_predictions': total_predictions,
            **threshold_stats
        })
    
    return confidence_results

def get_filtered_predictions(predictions: list[dict], confidence_threshold: float, cls: str):
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

        # Filter on class
        class_mask = labels_final == cls
        boxes_final = boxes_final[class_mask]
        scores_final = scores_final[class_mask]
        labels_final = labels_final[class_mask]

        filtered_predictions.append({
            **image_results,
            'boxes': boxes_final,
            'scores': scores_final,
            'labels': labels_final
        })

    return filtered_predictions

def get_filtered_targets(targets: list[dict], cls: str):
    """
        Process all targets and filter by class
    """
    filtered_targets = []

    for image_results in targets:
        boxes = image_results['boxes']
        labels = image_results['labels']

        # Filter on class
        class_mask = labels == cls
        boxes_final = boxes[class_mask]
        labels_final = labels[class_mask]

        filtered_targets.append({
            **image_results,
            'boxes': boxes_final,
            'labels': labels_final
        })

    return filtered_targets

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
