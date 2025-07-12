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
You can either choose to do inference on a model, or provide a file to calculate metrics on.<br>
Evaluation using a model & COCO dataset:
```bash
python3 -m evaluate --model_path <path_to_model> --model_type <model_type> --coco_annotation <path_to_coco_annotation>
```

Evaluation using a file with model predictions:
```bash
python3 -m evaluate --results_path <path_to_results_file>   
```

Inference results will be saved in the current working directory, in a file named **inference_results.json**.<br>
Evaluation results will be saved in the current working directory, in a file named **eval_results.json**.<br>

## Examples
Examples for the following can be found in folder **Examples**:
- COCO annotation(For running inference)
- Model predictions file(For getting metrics from a file)
- Metrics results

## Prerequisites - Inference
The tool is designed to handle ONNX/TensorRt models, and currently supports YOLO models only.<br>
A tool is provided to convert YOLO models to ONNX format, and can be found in **assets/convert_onnx_yolo.py**:<br>
```bash
python3 -m convert_onnx_yolo --model-path <path_to_model> --input-name <input_name> --output-name <output_name>
```

To convert a model to TensorRT, run the following command:
```bash
trtexec \
  --onnx=MODEL.onnx \
  --saveEngine=CONVERTED.engine \
  --optShapes=images:4x3x640x640 \
  --minShapes=images:1x3x640x640 \
  --maxShapes=images:8x3x640x640 \
  --shapes=images:4x3x640x640 \
  --fp16(Or omit for FP32)
```