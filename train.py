from ultralytics import YOLO

# Load a pretrained model (for example, YOLOv11 medium)
model = YOLO("yolo11m.pt")

# Train the model
model.train(data="config.yaml", 
            imgsz=640, 
            batch=8, 
            epochs=100, 
            workers=1,
            cache=False,
            device="cpu")
