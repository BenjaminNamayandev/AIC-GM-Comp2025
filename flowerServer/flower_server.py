import flwr as fl
# make sure to do pip install flwr ultralytics pytorch
if __name__ == "__main__":
    # Start the Flower server on localhost port 8080. Here, we run 3 rounds.
    fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=3))