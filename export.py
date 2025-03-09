# export_to_ncnn.py
from ultralytics import YOLO

# Load your YOLO model from best-200n.pt
model = YOLO("best-200n.pt")

# Export directly to NCNN with half precision
model.export(format="ncnn", half=True)

print("Export complete. Check the output folder for .param and .bin files.")
