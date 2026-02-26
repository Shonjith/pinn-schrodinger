import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_solution(model, device):
    model.eval()                       ## Set the model to evaluation mode

    x = np.linspace(-5, 5, 256)
    t = np.linspace(0, np.pi/2, 200)

    T, X = np.meshgrid(t, x)

    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    with torch.no_grad():
        uv = model(torch.cat([t_flat, x_flat], dim=1))
        u = uv[:, 0].cpu().numpy()
        v = uv[:, 1].cpu().numpy()

    h_abs = np.sqrt(u**2 + v**2)
    h_abs = h_abs.reshape(X.shape)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T, X, h_abs, shading="auto", cmap="viridis")
    plt.colorbar(label=r"$|h(t,x)|$")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("PINN Solution of Nonlinear Schrödinger Equation")
    plt.tight_layout()
    plt.show()
