from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch
import timm
import json

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = "./datasets/val2017/000000577182.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)

# Download ImageNet-1k class labels
with open('./sources/imagenet_class_index.json', 'r') as file:
    class_idx = json.load(file)
labels = [class_idx[str(i)][1] for i in range(len(class_idx))]

# Load the original PyTorch model
model = timm.create_model("hf_hub:timm/mvitv2_large.fb_in1k", pretrained=True)
model = model.eval()

# Run inference on the PyTorch model
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1).numpy()[0]

# Print the top-5 predictions
top5_indices = np.argsort(probabilities)[-5:][::-1]
for i in top5_indices:
    print(f"Class: {labels[i]}, Probability: {probabilities[i]:.4f}")