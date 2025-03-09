import os
import torch
from ultralytics import YOLO

def quantize_model(input_model_path, output_model_path):
    """
    Load the original checkpoint, convert the model to FP16,
    update the checkpoint with the FP16 model, and save the checkpoint.
    """
    # Load the original checkpoint dictionary
    checkpoint = torch.load(input_model_path, map_location=torch.device('cpu'))
    
    # Load the original model using YOLO
    yolo = YOLO(input_model_path)
    model = yolo.model

    # Set the model to evaluation mode and convert it to half precision (FP16)
    model.eval()
    model = model.half()
    
    # Update the checkpoint with the FP16 model and its state_dict
    checkpoint["model"] = model
    checkpoint["state_dict"] = model.state_dict()

    # Save the updated checkpoint
    torch.save(checkpoint, output_model_path)
    print(f"FP16 quantized model saved to {output_model_path}")

def run_inference(model_path, image_dir, output_file):
    """
    Load the FP16 quantized model using YOLO, run inference on all images in image_dir,
    and write detection results to output_file.
    """
    # Load the FP16 model checkpoint with YOLO
    yolo = YOLO(model_path)
    
    # List all image files in the directory with valid extensions
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    ]
    
    # Open the output file for writing detection results
    with open(output_file, "w") as f:
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            results = yolo(image_path)
            for result in results:
                boxes_np = result.boxes.xyxy.cpu().numpy()
                for box, box_np in zip(result.boxes, boxes_np):
                    # Format: <image_name> <class_id> <confidence_score> <x_min> <y_min> <x_max> <y_max>
                    line = (
                        f"{image_file} "          # image name
                        f"{int(box.cls)+1} "       # class ID (+1 if labels start at 1)
                        f"{float(box.conf):.4f} "   # confidence score
                        f"{box_np[0]:.2f} "         # x_min
                        f"{box_np[1]:.2f} "         # y_min
                        f"{box_np[2]:.2f} "         # x_max
                        f"{box_np[3]:.2f}\n"        # y_max
                    )
                    f.write(line)
    print(f"Detections written to {output_file}")

if __name__ == "__main__":
    # Define paths for the original model checkpoint,
    # the FP16 quantized model checkpoint,
    # the image directory, and the output file.
    input_model_path = "models/best-11n150.pt"        # Original model checkpoint
    quantized_model_path = "models/best-11n150_fp16.pt"  # Path for the FP16 quantized checkpoint
    image_dir = "ethan/data/test-thermal-data/test_images_8_bit"
    output_file = "detection_results/results.txt"
    
    # Convert the model to FP16 and save the updated checkpoint
    quantize_model(input_model_path, quantized_model_path)
    
    # Run inference using the FP16 quantized model
    run_inference(quantized_model_path, image_dir, output_file)
