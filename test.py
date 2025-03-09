from ultralytics import YOLO

model = YOLO("models/best-11n.pt")

results = model("ethan/data/test-thermal-data/test_images_8_bit/image_1.jpeg")

for result in results:
    boxes_np = result.boxes.xyxy.cpu().numpy()
    for box, box_np in zip(result.boxes, boxes_np):
        print(f"Class: {int(box.cls)}, Confidence: {float(box.conf)}, Bounding Box: {box_np}")