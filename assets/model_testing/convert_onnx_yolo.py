import torch
import torch.onnx
import onnxruntime as ort

# Load model
model_base = torch.load('./yolov9-m.pt', map_location='cuda', weights_only=False)
model = model_base['model'].float()

# Export model to ONNX
dummy_input = torch.randn(1, 3, 640, 640).to('cuda')
torch.onnx.export(
    model, 
    dummy_input, 
    'yolov9-m.onnx', 
    input_names=['input'],
    output_names=['output']
)

dummy_input = torch.randn(1, 3, 640, 640).to('cuda')
ort_sesson = ort.InferenceSession('yolov9-m.onnx')
outputs = ort_sesson.run(None, {'input': dummy_input.cpu().numpy()})
print(outputs)