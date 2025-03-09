from ultralytics import YOLO
import os

output_file = "detection_results/results.txt"

model = YOLO("models/best-11n.pt")  # path to the model

image_path = "ethan/data/test-thermal-data/test_images_8_bit/image_1.jpeg" # Process the image
results = model(image_path)

image_name = os.path.basename(image_path)

# Open the output file in write mode
with open(output_file, "w") as f:
    for result in results:
        boxes_np = result.boxes.xyxy.cpu().numpy()
        for box, box_np in zip(result.boxes, boxes_np):
            # Format: <image_name> <class_id> <confidence_score> <x_min> <y_min> <x_max> <y_max>
            line = f"{image_name} {int(box.cls) + 1} {float(box.conf):.4f} {box_np[0]:.2f} {box_np[1]:.2f} {box_np[2]:.2f} {box_np[3]:.2f}\n"
            f.write(line)
