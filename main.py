import torch
from src.model import PINN
from src.data import initial_points, collocation_points
from src.train import train
from src.visualize import visualize_solution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = PINN(layers=[2, 100, 100, 100, 100, 2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

data = {
    "initial": initial_points(50, device),
    "collocation": collocation_points(10000, device)
}

train(model, optimizer, data)

visualize_solution(model, device)

