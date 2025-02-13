import json

# Custom modules
from utils.wrapper import Wrapper
import utils.object_detection.evaluator as object_detection
import utils.config as config

# Model inference
model = Wrapper(
    model_path='./models/yolov9-m.onnx',
    model_type=config.MODEL_YOLO
)
results = object_detection.evaluate_model_inference(
    model=model,
    coco_annotation='./datasets/val2017/annotation.json',
    export_results=True
)

# From results
# results = object_detection.evaluate_results_file(
#     results_path='./inference_results.json'
# )

# Save results to JSON
with open('results.json', 'w') as file:
    json.dump(results, file, indent=4)