import os
import torch
from torchvision import datasets, transforms

def load_data(data_dir, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Helper function to be used by clients.
def create_data_loader(data_dir, batch_size=4, shuffle=True):
    return load_data(data_dir, batch_size, shuffle)