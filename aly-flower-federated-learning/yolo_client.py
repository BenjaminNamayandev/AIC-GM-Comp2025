import flwr as fl
from ultralytics import YOLO
import torch
import numpy as np

class YOLOClient(fl.client.NumPyClient):
    def __init__(self, model_path, new_data_loader):
        self.model = YOLO(model_path)
        self.new_data_loader = new_data_loader

    def get_parameters(self, config=None):
        return [torch.flatten(param).detach().cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        params = [torch.tensor(param) for param in parameters]
        for param, new_param in zip(self.model.parameters(), params):
            param.data.copy_(new_param.view(param.size()))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        original_parameters = self.get_parameters()
        
        # Train the model on the new data
        for images, targets in self.new_data_loader:
            self.model.train()
            # The forward call is assumed to perform training.
            self.model(images, targets)
            
        updated_parameters = self.get_parameters()
        # Compute weight delta: updated - original
        delta = [upd - orig for orig, upd in zip(original_parameters, updated_parameters)]
        
        # Log weight change magnitude (L2 norm for each parameter, then average)
        delta_norms = [np.linalg.norm(d) for d in delta]
        avg_delta_norm = np.mean(delta_norms)
        print(f"Client: Average weight change (L2 norm): {avg_delta_norm:.6f}")
        
        return delta, len(self.new_data_loader)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Evaluation logic could be added here.
        return 0.0, len(self.new_data_loader)

if __name__ == "__main__":
    from utils.training import create_data_loader
    new_data_loader = create_data_loader("../ethan/data/test-thermal-data/test_images_8_bit")
    
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=YOLOClient("models/best-200n.pt", new_data_loader)
    )