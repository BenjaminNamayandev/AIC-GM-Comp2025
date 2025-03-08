from ultralytics import YOLO

# Load a pretrained model (e.g., YOLOv11 medium)
model = YOLO("yolo11m.pt")  # or "yolov11.pt", whichever you have

# Train the model
model.train(
    data="config.yaml", 
    imgsz=640,
    batch=8,
    epochs=100,
    workers=1,
    cache=False,
    device="cpu"  # or device=0 if you have a GPU
)
