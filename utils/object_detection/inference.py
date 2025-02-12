from pycocotools.coco import COCO
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import logging
import traceback
import torchvision
import torch
import os

# Custom modules
import utils.config as config
from utils.wrapper import Wrapper

# Variables
logger = logging.getLogger(__name__)
NMS_IOU_THRESHOLD = 0.5
NMS_CONF_THRESHOLD = 0.05

def load_coco_dataset(coco_annotation: str) -> COCO:
    """
        Dataset should be in the following format:
        A folder, including images, with it an annotations.json file, in COCO format.
        returns the base annotations together with the predictions of the model, in COCO format.
    """

    # Validate existance of annotations file
    if not os.path.exists(coco_annotation):
        raise Exception(f"Annotation file not found at {coco_annotation}")
    
    # Parse COCO Annotations
    coco_dataset = COCO(coco_annotation)
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

def get_torchmetrics_predictions(model: Wrapper, coco_dataset: COCO, dataset_folder: str) -> list[dict]:
    """
        Loads model predictions for each image in the dataset
        returns all predictions in torchmetrics format
        predictions are in raw format and no confidence threshold is applied
    """

    # Load all images in folder
    predictions = []
    for image_id in tqdm(coco_dataset.getImgIds(), desc="Processing images predictions"):
        try:
            image_info = coco_dataset.loadImgs(image_id)[0]
            image_path = os.path.join(dataset_folder, image_info['file_name'])
            image = Image.open(image_path).convert('RGB')
        
            # Process model output   
            image_predictions = get_image_predictions(model, image)
            predictions.append(image_predictions)

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Image {image_id} - {e}")

    return predictions

def get_image_predictions(model: Wrapper, image: Image) -> dict:
    """
        Required by the Evaluator class
        Parses raw output from the model for a single image
        Returns a dict(in torchmetrics format), including all 
    """
    # Process predictions
    boxes_final = torch.tensor([])
    scores_final = torch.tensor([])
    labels_final = torch.tensor([])
    
    if model.model_type == config.MODEL_YOLO:
        # Resize to model's input size + Convert to tensor (C, H, W) & scale to [0,1]
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        image_array = transform(image).unsqueeze(0).numpy()

        # Get model output
        model_output = model.predict(image_array)
        output = torch.from_numpy(model_output[0]) 

        # Process predictions - Only one image
        predictions = output[0].T  # (8400, 84)

        if predictions.shape[0] > 0:
            # Split into boxes and class scores
            boxes = predictions[:, :4]
            scores = predictions[:, 4:]

            # Get max confidence score and class id for each prediction
            scores_max, scores_labels = torch.max(scores, dim=1)

            # Map label to coco class
            scores_labels += 1

            # Filter unsupported coco classes
            supported_mask = torch.isin(scores_labels, torch.tensor(config.SUPPORTED_COCO_CLASSES))
            boxes_filtered = boxes[supported_mask]
            scores_filtered = scores_max[supported_mask]
            labels_filtered = scores_labels[supported_mask]

            # Filter bboxes for confidence aas low as possible to apply NMS
            conf_mask = scores_filtered >= NMS_CONF_THRESHOLD
            boxes_filtered = boxes_filtered[conf_mask]
            scores_filtered = scores_filtered[conf_mask]
            labels_filtered = labels_filtered[conf_mask]

            # Apply NMS
            nms_mask = torchvision.ops.nms(
                boxes_filtered,
                scores_filtered,
                NMS_IOU_THRESHOLD
            )
            boxes_filtered = boxes_filtered[nms_mask]
            scores_filtered = scores_filtered[nms_mask]
            labels_filtered = labels_filtered[nms_mask]
            
            # Convert bboxes to original dimensions
            boxes_final = torch.zeros_like(boxes_filtered)
            image_width, image_height = image.size
            scale_x = image_width / 640
            scale_y = image_height / 640

            # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
            boxes_final[:, 0] = boxes_filtered[:, 0] - boxes_filtered[:, 2]/2  # x1
            boxes_final[:, 1] = boxes_filtered[:, 1] - boxes_filtered[:, 3]/2  # y1
            boxes_final[:, 2] = boxes_filtered[:, 0] + boxes_filtered[:, 2]/2  # x2
            boxes_final[:, 3] = boxes_filtered[:, 1] + boxes_filtered[:, 3]/2  # y2

            # Convert bboxes to original size
            boxes_final[:, [0, 2]] *= scale_x
            boxes_final[:, [1, 3]] *= scale_y

            # Finalize values
            scores_final = scores_filtered
            labels_final = labels_filtered
        
    # Return predictions
    return {
        'boxes': boxes_final.to(torch.float32),
        'scores': scores_final.to(torch.float32),
        'labels': labels_final.to(torch.int64)
    }
