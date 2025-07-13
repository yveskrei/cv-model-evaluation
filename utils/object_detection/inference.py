import torchvision.transforms as transforms
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image
import torchvision
import logging
import torch
import os
import traceback

# Custom modules
import utils.config as config

# Variables
logger = logging.getLogger(__name__)
NMS_IOU_THRESHOLD = 0.70
NMS_CONF_THRESHOLD = 0.001
MAX_IMAGE_DETECTIONS = 300

def load_coco_dataset(coco_annotation: str) -> COCO:
    """
        Dataset should be in the following format:
        A folder, including images, with it an annotations.json file, in COCO format.
        returns the base annotations together with the predictions of the model, in COCO format.
        We filter all categories to only include the first 80 categories(actual COCO categories).
    """

    # Validate existance of annotations file
    if not os.path.exists(coco_annotation):
        raise Exception(f"Annotation file not found at {coco_annotation}")
    
    try:
        # Parse COCO Annotations
        coco_dataset = COCO(coco_annotation)

        # Step 1: Filter categories
        filtered_cat_ids = [cat_id for cat_id in coco_dataset.getCatIds() if cat_id <= 80]

        # Step 2: Filter annotations
        filtered_ann_ids = coco_dataset.getAnnIds(catIds=filtered_cat_ids)
        filtered_anns = coco_dataset.loadAnns(filtered_ann_ids)

        # Step 3: Find image ids referenced by filtered anns
        filtered_img_ids = list(set(ann['image_id'] for ann in filtered_anns))

        # Step 4: Filter images
        filtered_imgs = [coco_dataset.imgs[img_id] for img_id in filtered_img_ids]

        # Step 5: Filter categories
        filtered_cats = [coco_dataset.cats[cat_id] for cat_id in filtered_cat_ids]

        # Now rebuild the dataset dict
        coco_dataset.dataset['annotations'] = filtered_anns
        coco_dataset.dataset['images'] = filtered_imgs
        coco_dataset.dataset['categories'] = filtered_cats

        # Rebuild index dicts
        coco_dataset.createIndex()

        logger.info(f"Loaded dataset with {len(coco_dataset.getImgIds())} images, {len(coco_dataset.getAnnIds())} annotations")
            
        return coco_dataset
    except Exception as e:
        logger.error(e)
        raise Exception('Could not load COCO dataset')

def get_torchmetrics_results(model, coco_dataset: COCO, dataset_folder: str) -> list[dict]:
    """
        Loads model annotations & predictions for each image in the dataset
        returns all results in torchmetrics format
        predictions are in raw format, before applying confidence threshold filter
    """

    # Load all images in folder
    predictions = []
    for image_id in tqdm(coco_dataset.getImgIds(), desc="Processing results images in dataset"):
        try:
            image_info = coco_dataset.loadImgs(image_id)[0]
            image_path = os.path.join(dataset_folder, image_info['file_name'])
            image = Image.open(image_path).convert('RGB')

            # Get image annotations for the image(Ground-truth)
            image_annotations = get_image_annotations(coco_dataset, image_id)
        
            # Run inference on image and get model predictions
            image_predictions = get_image_predictions(model, image)
            
            # Append results
            predictions.append({
                'target': image_annotations,
                'pred': image_predictions,
                'image_file': image_info['file_name'],
                'image_id': image_id
            })

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Image {image_id} - {e}")

    return predictions

def get_image_annotations(coco_dataset: COCO, image_id: str):
    """
        Returns all annotations for a given image
        returns in a torchmetrics format
    """
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
    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64)
    }

def get_image_predictions(model, image: Image) -> dict:
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
            transforms.Resize((640, 640), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        image_array = transform(image).unsqueeze(0).numpy()

        # Get model output
        model_output = model.predict(image_array)
        output = torch.from_numpy(model_output) 

        # Get first element of output if it's a list or tuple
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        if len(output.shape) == 1:
            output = output.reshape(84, 8400)
        
        # Convert to float32 if needed
        if output.dtype == torch.float16:
            output = output.to(torch.float32)

        # Process predictions - Only one image
        predictions = output.T  # (8400, 84)

        if predictions.shape[0] > 0:
            # Split into boxes and class scores
            boxes = predictions[:, :4]
            scores = predictions[:, 4:]

            # Get max confidence score and class id for each prediction
            scores_max, scores_labels = torch.max(scores, dim=1)

            # Map label to coco class
            scores_labels += 1

            # Filter bboxes for confidence as low as possible to apply NMS
            conf_mask = scores_max >= NMS_CONF_THRESHOLD
            boxes_filtered = boxes[conf_mask]
            scores_filtered = scores_max[conf_mask]
            labels_filtered = scores_labels[conf_mask]

            # Convert bboxes to original dimensions
            boxes_xyxy = torch.zeros_like(boxes_filtered)
            image_width, image_height = image.size
            scale_x = image_width / 640
            scale_y = image_height / 640

            # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
            boxes_xyxy[:, 0] = boxes_filtered[:, 0] - boxes_filtered[:, 2]/2  # x1
            boxes_xyxy[:, 1] = boxes_filtered[:, 1] - boxes_filtered[:, 3]/2  # y1
            boxes_xyxy[:, 2] = boxes_filtered[:, 0] + boxes_filtered[:, 2]/2  # x2
            boxes_xyxy[:, 3] = boxes_filtered[:, 1] + boxes_filtered[:, 3]/2  # y2

            # Convert bboxes to original size
            boxes_xyxy[:, [0, 2]] *= scale_x
            boxes_xyxy[:, [1, 3]] *= scale_y

            # Apply NMS
            nms_mask = torchvision.ops.nms(
                boxes_xyxy,
                scores_filtered,
                NMS_IOU_THRESHOLD
            )
            boxes_xyxy = boxes_xyxy[nms_mask]
            scores_filtered = scores_filtered[nms_mask]
            labels_filtered = labels_filtered[nms_mask]

            # Get top-k predictions based on scores
            sorted_indices = torch.argsort(scores_filtered, descending=True)
            top_indices = sorted_indices[:MAX_IMAGE_DETECTIONS]

            # Step 3: Index into all final tensors
            boxes_final = boxes_xyxy[top_indices]
            scores_final = scores_filtered[top_indices]
            labels_final = labels_filtered[top_indices]

        
    # Return predictions
    return {
        'boxes': boxes_final.to(torch.float32),
        'scores': scores_final.to(torch.float32),
        'labels': labels_final.to(torch.int64)
    }
