from transformers import DetrForObjectDetection
import torch
from PIL import Image
import requests
from torchvision import transforms

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# you can specify the revision tag if you don't want the timm dependency
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

transform = transforms.Compose([
    transforms.Resize(800),  # Resize image so that the shorter side is 800 pixels
    transforms.ToTensor(),  # Convert image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Apply the transformations
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

torch.onnx.export(
    model,  # model to export
    image_tensor,  # dummy input to trace the model
    'detr_resnet_50.onnx',  # file path where the ONNX model will be saved
    input_names=['input'],  # name of the input
    output_names=['output'],  # name of the output
    dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}}  # Allow dynamic shapes
)