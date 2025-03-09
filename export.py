import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

model = YOLO("models/best-11n150.pt").model

for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        # Prune 20% of the channels along the output dimension (dim=0)
        prune.ln_structured(module, name="weight", amount=0.2, n=1, dim=0)


for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

# Save the pruned model (optional)
torch.save(model.state_dict(), "best-11n150_pruned.pt")


### THIS IS WHERE THE MODEL GETS TURNED INTO AN ONNX
### AND WHERE FP16 IS APPLIED ###


# # Assuming 'model' is already your pruned model loaded from above
# model.eval().half()  # Set to evaluation mode and convert to half precision
# 
# # Create a dummy input matching your thermal image dimensions (batch, channels, height, width)
# dummy_input = torch.randn(1, 3, 512, 640).half()
# 
# # Export the model to ONNX
# torch.onnx.export(
#     model,
#     dummy_input,
#     "yolov11_pruned_fp16.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     opset_version=12,  # Adjust if needed
#     do_constant_folding=True,
# )


## Run these commands to change it into NCNN for the rasberrypi:
## ./onnx2ncnn yolov11_pruned_fp16.onnx yolov11_pruned_fp16.param yolov11_pruned_fp16.bin
## ./ncnnoptimize yolov11_pruned_fp16.param yolov11_pruned_fp16.bin yolov11_pruned_fp16_opt.param yolov11_pruned_fp16_opt.bin 65536 0