from ultralytics import YOLO
import torch

target_model = "models/best-11s150.pt"

model = torch.load(target_model)
print(model)
# model.export(format="ncnn", imgsz=640, batch=1)