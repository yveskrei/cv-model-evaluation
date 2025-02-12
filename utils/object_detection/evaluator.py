from pycocotools.coco import COCO
import logging
import os

# Custom modules
import utils.config as config
import utils.object_detection.inference as inference
from utils.wrapper import Wrapper

# Variables
logger = logging.getLogger(__name__)

def evaluate_model(model: Wrapper, coco_annotation: str):
    """
        Given a model and a COCO dataset, will return results after
        doing inference on its own
    """
    # Load dataset from path
    coco_dataset = inference.load_coco_dataset(coco_annotation)

    # Load raw model predictions for each image from the dataset
    model_predictions = inference.get_torchmetrics_predictions(
        model=model,
        coco_dataset=coco_dataset,
        dataset_folder=os.path.dirname(coco_annotation)
    )

    logger.info(model_predictions[:5])

    return {}


