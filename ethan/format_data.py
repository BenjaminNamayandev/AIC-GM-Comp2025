import os

# constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512

# file path
file_path = "data/test-thermal-data/labels_test_8_bit.txt"

# Create a dictionary to adjust class ids (YOLO zero-indexed)
class_map = {1: 0, 2: 1, 3: 2}

# read annotations.txt file
annotations = {}
with open(file_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 6:
            raise ValueError("Invalid line: %s" % line)

        file_name, class_id, x_min, x_max, y_min, y_max = parts
        class_id = class_map[int(class_id)]
        x_min, x_max, y_min, y_max = map(float, [x_min, x_max, y_min, y_max])

        # Compute center coordinates and box size
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Normalize the coordinates
        x_center_norm = x_center / IMAGE_WIDTH
        y_center_norm = y_center / IMAGE_HEIGHT
        width_norm = box_width / IMAGE_WIDTH
        height_norm = box_height / IMAGE_HEIGHT

        # Create or append to the list for the file
        if file_name not in annotations:
            annotations[file_name] = []
        annotations[file_name].append(
            f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
        )

# Write out the YOLO annotation files (one per image)
for file_name, lines in annotations.items():
    base, _ = os.path.splitext(file_name)
    txt_file = base + ".txt"
    with open("yolodata/labels/" + txt_file, "w") as f:
        for line in lines:
            f.write(line + "\n")
