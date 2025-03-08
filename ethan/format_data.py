import os

FILE_WIDTH = 640
FILE_HEIGHT = 512


def convert_coordinates(x_min, y_min, x_max, y_max):
    x_center = (x_min + x_max) / (2 * FILE_WIDTH)
    y_center = (y_min + y_max) / (2 * FILE_HEIGHT)
    width = x_max - x_min / FILE_WIDTH
    height = y_max - y_min / FILE_HEIGHT
    return x_center, y_center, width, height


def process_annotations(input_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

        annotations = {}
        for line in lines:
            parts = line.strip().split(" ")
            file_name = parts[0].replace(".jpeg", "")  # Remove .jpeg extension
            class_id = int(parts[1]) - 1
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name, data in annotations.items():
        output_file = os.path.join(output_dir, f"{file_name}.txt")
        with open(output_file, "w") as file:
            for annotation in data:
                file.write(f"{annotation}\n")


input_file = "train_labels_8_bit.txt"
output_dir = "labels/train"
annotations = process_annotations(input_file)
save_annotations(annotations, output_dir)
