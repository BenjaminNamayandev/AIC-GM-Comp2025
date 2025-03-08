from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data="config.yaml", 
            imgsz = 640, batch = 8, 
            epochs = 100, workers = 1, 
            cache = False,
            device = "cpu")

