# CV Model evaluation
Acts as an evaluation tool for variaty of models, outputting various graphs and metrics to help get a better
understanding of the behaviour of the model.<br>
The script is designed to both run inference with a given dataset(COCO format), and ONNX model, and create metrics given a file of model predictions(torchmetrics format).<br>
Script is currently supporting YOLO models only for inference

## Setting things up
Make sure you install all the required python modules to run the tool properly
```bash
pip install -r requirements.txt
```
Also, make sure you install **torch** and **torchvision** python modules, where CUDA versions are recommanded.<br>
To download the CUDA versions for these packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage
You can either choose to do inference on a model, or provide a file to calculate metrics on.
```python
import json

# Custom modules
from utils.wrapper import Wrapper
import utils.object_detection.evaluator as object_detection
import utils.config as config

# Get metrics from model inference
model = Wrapper(
    model_path='./models/yolov9-m.onnx',
    model_type=config.MODEL_YOLO
)
metrics_results = object_detection.evaluate_model_inference(
    model=model,
    coco_annotation='./datasets/val2017/annotation.json',
    export_results=True
)

# Get metrics from results
metrics_results = object_detection.evaluate_results_file(
    results_path='./inference_results.json'
)

# Save results to JSON
with open('results.json', 'w') as file:
    json.dump(metrics_results, file, indent=4)
```

## Examples
Examples for the following can be found in folder **Examples**:
- COCO annotation(For running inference)
- Model predictions file(For getting metrics from a file)
- Metrics results

## Prerequisites - Inference
The tool is designed to handle ONNX models only, meaning you must export your model to ONNX format first.<br>
When doing so, please make sure you follow the format below:
```python
INPUT_MODEL = 'yolov9-m.pt'
OUTPUT_MODEL = 'yolov9-m.onnx'

# Load model(Can vary between models)
model_base = torch.load(
    INPUT_MODEL, 
    map_location='cuda', 
    weights_only=False
)
model = model_base['model'].float()

# Create a dummpy input for ONNX to understand what input it would be expecting at inference
# Change to the input your model is expecting(this is YOLO Architecture specific)
dummy_input = torch.randn(1, 3, 640, 640).to('cuda')

# Export model to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    OUTPUT_MODEL, 
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}}  # Allow dynamic shapes
)
```
Make sure you have **input_names**, and **output_names** equivalent to those in the example above.