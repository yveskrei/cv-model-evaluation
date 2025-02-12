import json

# Custom modules
from utils.wrapper import Wrapper
import utils.object_detection.evaluator as object_detection
import utils.config as config

model = Wrapper(
    model_path='./models/yolov9-m.onnx',
    model_type=config.MODEL_YOLO
)

# Evaluate model
results = object_detection.evaluate_model(
    model=model,
    coco_annotation='./datasets/val2017/annotation.json'
)

# Save results to JSON
with open('results.json', 'w') as file:
    json.dump(results, file, indent=4)