from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained model (e.g., YOLOv11 medium)
    model = YOLO("yolo11n.pt")  # or "yolov11.pt", whichever you have

    # Train the model using CUDA (GPU)
    model.train(
        data="config.yaml",
        imgsz=(640, 512),
        batch=64,
        epochs=5,
        workers=1,
        cache=False,
        device="cuda:0")

