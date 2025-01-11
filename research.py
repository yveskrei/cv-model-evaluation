from utils.wrapper import Wrapper, MODEL_YOLO
from utils.evaluator import Evaluator 
import json

# Evaluate ONNX model
evaluator = Evaluator(
    model = Wrapper(
        model_path='models/yolo11m.onnx',
        model_type=MODEL_YOLO
    )
)

# Evaluate model
results = evaluator.evaluate(
    annotation_path='datasets/val2017/annotation.json'
)

# # Save results to JSON
# with open('results.json', 'w') as file:
#     json.dump(results, file, indent=4)