from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm
import numpy as np
import os

# Custom modules
import utils.wrapper as wrapper

# Object class mappings
SUPPORTED_COCO_CLASSES = [1, 3, 4] # Person, Car, Motorcycle
YOLO_TO_COCO = {
    0: 1, # Person
    2: 3, # Car
    3: 4, # Motorcycle
}   

class ModelEvaluator:
    def __init__(self, model_wrapper: wrapper.ModelWrapper):
        self.model_wrapper = model_wrapper
    
    def evaluate(self, annotation_path: str):
        """
            Evaluates the model on a given dataset, returning results for each confidence
            threshold between 0 and 1
        """
        # Load dataset from path
        coco_dataset = self.load_dataset(annotation_path)
        results = {
            'model_name': os.path.basename(self.model_wrapper.model_path),
            'dataset_name': os.path.basename(os.path.dirname(annotation_path)),
        }

        for conf_threshold in tqdm(np.arange(0.50, 1.05, 0.05), desc="Processing confidence thresholds"):
            # Load predictions
            predictions = self.load_predictions(annotation_path, coco_dataset, conf_threshold)

            # Define metrics
            map = 0
            map50 = 0
            map75 = 0
            precision = 0
            recall = 0

            # Load predictions results
            if len(predictions):
                coco_predictions = coco_dataset.loadRes(predictions)
                
                # Evaluate predictions
                coco_eval = COCOeval(coco_dataset, coco_predictions, iouType="bbox")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                # Update metrics
                map = coco_eval.stats[0]
                map50 = coco_eval.stats[1]
                map75 = coco_eval.stats[2]
                precision = coco_eval.stats[5]
                recall = coco_eval.stats[6]

            # Append results
            results.append({
                "confidence_threshold": conf_threshold,
                "mAP": map,
                "mAP50": map50,
                "mAP75": map75,
                "precision": precision,
                "recall": recall,
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
    
    def load_predictions(self, annotation_path: str, coco_dataset: COCO, conf_threshold: float):
        """
            Loads model predictions for each image in the dataset
            returns the dataset with the model predictions
        """

        # Load all images in folder
        predictions = []
        for image_id in tqdm(coco_dataset.getImgIds(), desc="Processing images"):
            try:
                image_info = coco_dataset.loadImgs(image_id)[0]
                image_path = os.path.join(os.path.dirname(annotation_path), image_info["file_name"])
                image = Image.open(image_path).convert("RGB")
                
                # Get image predictions
                image_predictions = self.model_wrapper.predict(
                    image=image,
                    conf_threshold=conf_threshold
                )

                # Format prediction in COCO format
                image_predictions = list(map(
                    lambda pred: self.process_prediction(pred, image_id), 
                    image_predictions
                ))
                image_predictions = list(filter(
                    lambda x: x is not None, 
                    image_predictions
                ))
                
                # Append new predictions
                if len(image_predictions):
                    predictions.extend(image_predictions)
            except Exception as e:
                print(f"Image {image_id} Error: {e}")

        return predictions

    def process_prediction(self, prediction: dict, image_id: str):
        """
            Standardizes model predictions to COCO format, mapping classes from model classes
            to coco formatted classes, and converting bounding boxes to COCO format.
        """
        coco_class = None
        
        # Map model class to coco class
        if self.model_wrapper.model_type == wrapper.MODEL_YOLO:
            coco_class = YOLO_TO_COCO.get(prediction['class'], None)
        
        if coco_class is not None:
            return {
                "image_id": image_id,
                "category_id": coco_class,
                "bbox": prediction['bbox'],
                "score": prediction['score']
            }
        else:
            return None