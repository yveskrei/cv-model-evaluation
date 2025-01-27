import timm
import torch

# Load model
model = timm.create_model("hf_hub:timm/mvitv2_small.fb_in1k", pretrained=True)
model = model.eval()

dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the shape if needed

# Export the model
torch.onnx.export(
    model,                  # Model to export
    dummy_input,            # Dummy input tensor
    "mvitv2_small.onnx",       # Output ONNX file path
    export_params=True,     # Store the trained parameter weights in the model file
    input_names=["input"],  # Input tensor name
    output_names=["output"],  # Output tensor name
    dynamic_axes={          # Define dynamic axes for variable batch size
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)