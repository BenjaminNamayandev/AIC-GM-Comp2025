import os
from ultralytics import YOLO

# Path to your model
model = YOLO("models/best-200n_ncnn_model")

# Directory containing your test images
image_dir = "ethan/data/test-thermal-data/test_images_8_bit"

# Output file path
output_file = "detection_results/results.txt"

# all images in directory
image_files = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpeg", ".jpg", ".png"))
]

# Open the output file in write mode
with open(output_file, "w") as f:
    # Loop through each image in the directory
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)

        # Run inference on the image
        results = model(image_path)

        # Write each detection to results.txt
        for result in results:
            boxes_np = result.boxes.xyxy.cpu().numpy()
            for box, box_np in zip(result.boxes, boxes_np):
                # Format: <image_name> <class_id> <confidence_score> <x_min> <y_min> <x_max> <y_max>
                line = (
                    f"{image_file} "        # image name
                    f"{int(box.cls)+1} "    # class ID (+1 if your labels start at 1 instead of 0)
                    f"{float(box.conf):.4f} "
                    f"{box_np[0]:.2f} "
                    f"{box_np[1]:.2f} "
                    f"{box_np[2]:.2f} "
                    f"{box_np[3]:.2f}\n"
                )
                f.write(line)
