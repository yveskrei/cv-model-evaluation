import json

# Custom modules
from utils.wrapper import Wrapper
import utils.config as config

model = Wrapper(
    model_path='./models/yolov9-m.onnx',
    model_task=config.TASK_DETECTION,
    model_type=config.MODEL_YOLO
)

# Evaluate model
results = model.evaluator.evaluate(
    dataset_annotation='./datasets/val2017/annotation.json',
    dataset_name='val2017',
)

# Save results to JSON
with open('results.json', 'w') as file:
    json.dump(results, file, indent=4)