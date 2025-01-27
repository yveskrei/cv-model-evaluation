from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import logging
import os
import traceback

# Custom modules
import utils.config as config

# Variables
logger = logging.getLogger(__name__)

class Evaluator:
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
            returns all predictions in raw format(before processing NMS, confidence thresholding, etc)
        """

        # Load all images in folder
        predictions = []
        for image_id in tqdm(coco_dataset.getImgIds(), desc="Processing images predictions"):
            try:
                image_info = coco_dataset.loadImgs(image_id)[0]
                image_path = os.path.join(dataset_folder, image_info['file_name'])
                image = Image.open(image_path).convert('RGB')
            
                # Process model output   
                image_predictions = self.get_model_predictions(image)

                # Append predictions
                if len(image_predictions):
                    predictions.append(image_predictions)
                else:
                    logger.warning(f"Image {image_id} - No predictions")

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Image {image_id} - {e}")

        return predictions