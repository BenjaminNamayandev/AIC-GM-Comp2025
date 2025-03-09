import torch
from ultralytics import YOLO

# Load your model (replace with your model file)
model = YOLO("models/best-11n150_fp16.pt").model
model.eval()  # Set to evaluation mode

# Create a dummy input matching your input dimensions (batch, channels, height, width)
dummy_input = torch.randn(1, 3, 512, 640)

# Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12,  # Use a compatible opset version
    do_constant_folding=True,
)
