import flwr as fl
import torch
from ultralytics import YOLO
import numpy as np

def load_initial_parameters(model_path):
    model = YOLO(model_path)
    return [torch.flatten(param).detach().cpu().numpy() for param in model.parameters()]

class DeltaFedAvgStrategy(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters, **kwargs):
        super().__init__(**kwargs)
        self.current_parameters = initial_parameters
        
    def aggregate_fit(self, rnd, results, failures):
        if failures:
            return None
        
        # Each result contains (delta, num_examples)
        deltas, num_examples_list = zip(*[
            (fit_res.parameters, fit_res.num_examples) for _, fit_res in results
        ])
        total_examples = sum(num_examples_list)
        # Weighted average of the deltas (each delta is a flat numpy vector)
        agg_delta = [
            sum(delta[i] * num_examples for delta, num_examples in zip(deltas, num_examples_list)) / total_examples
            for i in range(len(deltas[0]))
        ]
        # Log aggregated delta norm
        agg_delta_norm = np.linalg.norm(np.array(agg_delta))
        print(f"Server: Aggregated weight update (L2 norm): {agg_delta_norm:.6f}")
        
        # Update global parameters by adding aggregated delta
        self.current_parameters = [current + delta for current, delta in zip(self.current_parameters, agg_delta)]
        return self.current_parameters

if __name__ == "__main__":
    initial_parameters = load_initial_parameters("models/best-200n.pt")
    strategy = DeltaFedAvgStrategy(
        initial_parameters=initial_parameters,
        fraction_fit=0.5,
        min_fit_clients=1
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )