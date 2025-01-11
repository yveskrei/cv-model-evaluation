from utils.wrapper import ModelWrapper, MODEL_YOLO
from utils.evaluator import ModelEvaluator 
import json

# Evaluate ONNX model
model = ModelWrapper(
    model_path='models/yolo11m.onnx',
    model_type=MODEL_YOLO
)
model_evaluator = ModelEvaluator(model)

# Evaluate model
results = model_evaluator.evaluate(
    annotation_path='datasets/val2017/annotation.json'
)

# Save results to JSON
with open('results.json', 'w') as file:
    json.dump(results, file, indent=4)