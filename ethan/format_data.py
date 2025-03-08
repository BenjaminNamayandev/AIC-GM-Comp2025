import os

# Image dimensions
FILE_WIDTH = 640
FILE_HEIGHT = 512

def convert_coordinates(x_min, y_min, x_max, y_max):
    """
    Convert bounding box corners into YOLO format:
      x_center, y_center, width, height
    All normalized by the image width/height.
    """
    x_center = (x_min + x_max) / (2.0 * FILE_WIDTH)
    y_center = (y_min + y_max) / (2.0 * FILE_HEIGHT)
    width = (x_max - x_min) / float(FILE_WIDTH)
    height = (y_max - y_min) / float(FILE_HEIGHT)
    return x_center, y_center, width, height

def process_annotations(input_file):
    """
    Read each line from input_file and convert bounding boxes to YOLO format.
    Returns a dict of {image_name: [annotation_line, ...]}.
    """
    with open(input_file, "r") as file:
        lines = file.readlines()

    annotations = {}
    for line in lines:
        parts = line.strip().split(" ")
        # Example line format:  image_name.jpeg  class  x_min  y_min  x_max  y_max
        file_name = parts[0].replace(".jpeg", "")  # remove .jpeg if present
        class_id = int(parts[1]) - 1              # YOLO classes start at 0
        x_min = int(parts[2])
        y_min = int(parts[3])
        x_max = int(parts[4])
        y_max = int(parts[5])

        x_center, y_center, width, height = convert_coordinates(
            x_min, y_min, x_max, y_max
        )

        if file_name not in annotations:
            annotations[file_name] = []
        annotations[file_name].append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return annotations

def save_annotations(annotations, output_dir):
    """
    Given a dict of annotations, save each image's annotations
    to a separate text file in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_name, data in annotations.items():
        output_file = os.path.join(output_dir, f"{file_name}.txt")
        with open(output_file, "w") as f:
            for annotation in data:
                f.write(annotation + "\n")

if __name__ == "__main__":
    # Build paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")  # one level up, then data/

    # Input text file (training annotations) and output directory
    input_file = os.path.join(data_dir, "val_labels_8_bit.txt")
    output_dir = os.path.join(data_dir, "val/labels")

    annotations = process_annotations(input_file)
    save_annotations(annotations, output_dir)

    print(f"Processed annotations from {input_file}")
    print(f"Saved YOLO-format txt files to {output_dir}")
