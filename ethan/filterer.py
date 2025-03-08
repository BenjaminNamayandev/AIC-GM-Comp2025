# imports
import os
import json
from typing import List

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512


# read coco json file
def read_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


# remove all coco entries with category_id > 3
def filter_json(data):
    annotations_data = data["annotations"]
    filtered_annotations = []
    for entry in annotations_data:
        if entry["category_id"] <= 3:
            filtered_annotations.append(entry)
    return filtered_annotations


# format to class-1 x-center y-center width height
def format_data(annotations_data, file_dict):
    formatted_data: dict[str, List[List[str]]] = {}
    for entry in annotations_data:
        file_name = file_dict[entry["image_id"]]
        # centerX = int((entry["bbox"][0] + entry["bbox"][2]) / 2)
        # if centerX & 1:
        #     centerX = centerX + 1
        #
        # centerY = int((entry["bbox"][1] + entry["bbox"][3]) / 2)
        # if centerY & 1:
        #     centerY = centerY + 1
        # row = [
        #     entry["category_id"],
        #     centerX,
        #     centerY,
        #     entry["bbox"][2],
        #     entry["bbox"][3],
        # ]

        row = [
            (entry["category_id"] - 1),
            round((entry["bbox"][0] + entry["bbox"][2] / 2) / IMAGE_WIDTH, 6),
            round((entry["bbox"][1] + entry["bbox"][3] / 2) / IMAGE_HEIGHT, 6),
            round(entry["bbox"][2] / IMAGE_WIDTH, 6),
            round(entry["bbox"][3] / IMAGE_HEIGHT, 6),
        ]

        row = [str(x) for x in row]

        if file_name in formatted_data:
            formatted_data[file_name].append(row)
        else:
            formatted_data[file_name] = [row]

    return formatted_data


# get image dictionary
def get_image_dict(data):
    images_data = data["images"]
    image_dict = {}

    for image in images_data:
        # remove leading data/
        image_dict[image["id"]] = image["file_name"][5:]

    return image_dict


# write filtered data to new json file
def write_data(data, file):
    with open(file, "w") as f:
        for entry in data:
            f.write(" ".join(entry) + "\n")


# main function
def main():
    # read coco json file
    data = read_json("coco.json")
    # filter data
    filtered_annotations = filter_json(data)
    # get image dictionary
    file_dict = get_image_dict(data)
    # format data
    final_data = format_data(filtered_annotations, file_dict)
    # make labels directory if it doesn't exist
    if not os.path.exists("labels"):
        os.makedirs("labels")

    # write data
    for key in final_data:
        # remove .jpg from key

        write_data(final_data[key], "labels/" + key[:-4] + ".txt")


if __name__ == "__main__":
    main()
