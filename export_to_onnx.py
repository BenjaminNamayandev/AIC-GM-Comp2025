import torch

def export_fp16_onnx(
        model_path="models/best-200n.pt",      # Path to your already trained model
        onnx_path="200n_fp16.onnx",  # Desired output ONNX file
        input_size=(1, 1, 640, 640)   # Adjust to match your model's expected (N,C,H,W)
    ):
    """
    Loads a PyTorch model, converts it to FP16, and exports it to ONNX.
    """

    # 1. Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load your model (assuming it was saved with torch.save(model, 'best.pt'))
    #    If you have a state_dict instead, you must first define your model architecture
    #    and then do something like model.load_state_dict(torch.load(...)).
    model = torch.load(model_path, map_location=device)

    # 3. Set model to inference mode
    model.eval()

    # 4. Move model to GPU (if available) and cast to FP16
    model = model.to(device).half()

    # 5. Create a dummy input in FP16 with the same shape you normally feed your model
    #    e.g., batch size=1, 3 channels, 640x640
    dummy_input = torch.randn(*input_size, dtype=torch.float16, device=device)

    # 6. Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,         # pick an appropriate opset (>=11 generally recommended)
        do_constant_folding=True,
        input_names=["images"],   # name your input layer(s)
        output_names=["output"]   # name your output layer(s)
    )
    print(f"Model exported to: {onnx_path}")


if __name__ == "__main__":
    # Example usage:
    export_fp16_onnx(
        model_path="models/best-200n.pt",      # Path to your already trained model
        onnx_path="200n_fp16.onnx",  # Desired output ONNX file
        input_size=(1, 1, 640, 640)  # typical YOLO input shape
    )
