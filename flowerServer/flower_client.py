import flwr as fl
from ultralytics import YOLO
import torch
# make sure to do pip install flwr ultralytics pytorch, i didnt add to requirements just yet


# Load the trained YOLO nano or small model, I used whatever model was currently pushed to the repo
model = YOLO("flowerServer/best-11s.pt")

class YOLOClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        # Extract parameters from the PyTorch model as a list of numpy arrays.
        return [val.cpu().numpy() for val in model.model.state_dict().values()]

    def set_parameters(self, parameters):
        # Set the PyTorch model parameters from a list of numpy arrays.
        state_dict = model.model.state_dict()
        new_state_dict = { 
            key: torch.tensor(param)
            for key, param in zip(state_dict.keys(), parameters)
        }
        model.model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Run one (local) epoch of training. ( i dont wanna kill my laptop so i just did 1 epoch)
        # Adjust dataset and training parameters as u see fit.
        model.train(data="config.yaml", epochs=1, imgsz=(640, 512), batch=8, workers=1)
        
        # Return updated parameters, number of examples used, and an empty dict for metrics.
        new_parameters = self.get_parameters()
        num_examples = 10  # update this with wutever the actual number of training examples are, i was using 10
        return new_parameters, num_examples, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
    #I did not do it, but u must not forget to add the proper config.yaml to point to the validation dataset, the current val: is the old one
        results = model.val(data="config.yaml", imgsz=(640, 512), batch=8, workers=1)
        
        # made it for a mAP with an IoU of 0.5 like they want
        mAP50 = results.metrics.get("mAP50") if hasattr(results, "metrics") else 0.0

        num_examples = 10  # update with the actual count
        return float(mAP50), num_examples, {"mAP@0.5": mAP50}

if __name__ == "__main__":
    # Connect to the Flower server (adjust the server address if needed).
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=YOLOClient())

