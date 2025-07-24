

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import trange

def main(csv_path: str):
    # 1) Load & sort full dataset
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path, usecols=["z","y"]).sort_values("z").reset_index(drop=True)

    # Extract raw arrays
    z_full = df["z"].values.astype(np.float32)    # depths
    y_full = df["y"].values.astype(np.float32)    # deflections (first is negative)

    # 2) Top‐boundary measurement
    z0, z1 = z_full[0], z_full[1]
    y0, y1 = y_full[0], y_full[1]
    slope0 = (y1 - y0) / (z1 - z0)                 # slope at the head

    # 3) Non‐dimensionalize entire range for PINN input
    L_char = z_full.max()
    y_char = max(abs(y_full.min()), abs(y_full.max()))
    z_nd   = (z_full / L_char).reshape(-1,1)
    y_nd   = (y_full / y_char).reshape(-1,1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_data = torch.tensor(z_nd, dtype=torch.float32, device=device).requires_grad_(True)

    # 4) Collocation points on [0, 1]
    M = 500
    z_colloc = torch.linspace(0,1,M,device=device).unsqueeze(1).requires_grad_(True)

    # 5) Prepare the two BCs: head (z=0) and toe (z=1)
    z0_nd      = float(z_nd[0])   # should be 0
    y0_nd      = float(y_nd[0])   # this is negative
    slope0_nd  = slope0 / y_char

    # toe displacement BC at z=1
    y1_nd      = float(y_nd[-1])

    # Torch tensors for BC enforcement
    z_head = torch.tensor([[z0_nd]], dtype=torch.float32, device=device, requires_grad=True)
    y_head = torch.tensor([[y0_nd]], dtype=torch.float32, device=device)
    s_head = torch.tensor([slope0_nd],  dtype=torch.float32, device=device)

    z_toe  = torch.tensor([[1.0]],    dtype=torch.float32, device=device, requires_grad=True)
    y_toe  = torch.tensor([[y1_nd]],  dtype=torch.float32, device=device)

    # 6) Define the PINN
    class PINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1,128), nn.Tanh(),
                nn.Linear(128,128), nn.Tanh(),
                nn.Linear(128,128), nn.Tanh(),
                nn.Linear(128,1)
            )
        def forward(self,x):
            return self.net(x)

    model    = PINN().to(device)
    EI_tilde = nn.Parameter(torch.tensor(5.0, device=device))
    params   = list(model.parameters()) + [EI_tilde]

    # 7) Physics residual for y'''' + β^4 y = 0
    def physics_residual(z):
        y   = model(z)
        d1  = torch.autograd.grad(y, z, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        d2  = torch.autograd.grad(d1,z, grad_outputs=torch.ones_like(d1),create_graph=True)[0]
        d3  = torch.autograd.grad(d2,z, grad_outputs=torch.ones_like(d2),create_graph=True)[0]
        d4  = torch.autograd.grad(d3,z, grad_outputs=torch.ones_like(d3),create_graph=True)[0]
        β   = (1.0/EI_tilde).pow(0.25)
        return d4 + β**4 * y

    # 8) Combined loss: PDE + head BC disp+slope + toe BC disp
    def loss_fn():
        res      = physics_residual(z_colloc)
        loss_pde = res.pow(2).mean()

        # head displacement
        y_head_pred = model(z_head)
        loss_hd     = (y_head_pred - y_head).pow(2).mean()
        # head slope
        dy_head     = torch.autograd.grad(y_head_pred, z_head,
                          grad_outputs=torch.ones_like(y_head_pred),
                          create_graph=True)[0]
        loss_hs     = (dy_head - s_head).pow(2).mean()

        # toe displacement
        y_toe_pred  = model(z_toe)
        loss_td     = (y_toe_pred - y_toe).pow(2).mean()

        # weight BCs heavily so they’re exactly satisfied
        return loss_pde + 100*(loss_hd + loss_hs + loss_td)

    # 9) Optimize: Adam → L‐BFGS
    opt = torch.optim.Adam(params, lr=1e-4)
    for _ in trange(3000, desc="Adam"):
        opt.zero_grad(); L=loss_fn(); L.backward(); opt.step()

    lbfgs = torch.optim.LBFGS(params, max_iter=2000, tolerance_grad=1e-9,
                              tolerance_change=1e-9, history_size=50,
                              line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad(); L=loss_fn(); L.backward(); return L
    lbfgs.step(closure)

    # 10) Evaluate on the full raw z
    model.eval()
    z_test_nd = torch.tensor(z_nd, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pred_nd = model(z_test_nd).cpu().numpy().flatten()
    y_pred = y_pred_nd * y_char  # back to physical
    
    
    # 11) Plot raw vs PINN
    plt.figure(figsize=(6,8))
    plt.plot(y_full, z_full, 'k--', lw=2, label="Raw data")
    plt.plot(y_pred, z_full, 'g-',  lw=2, label="PINN pred")
    plt.ylim(z_full.min(), z_full.max())
    plt.xlabel("Deflection y [m]")
    plt.ylabel("Depth z [m]")
    plt.title("Raw vs PINN (head BC enforced at y0≈{:.3f})".format(y0))
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # 12) Metrics
    rmse = np.sqrt(mean_squared_error(y_full, y_pred))
    r2   = r2_score(y_full, y_pred)
    print(f"EI_tilde   = {EI_tilde.item():.4f}")
    print(f"RMSE (raw) = {rmse:.4e} m")
    print(f"R² (raw)   = {r2:.4f}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="case1(Reese).csv", help="Path to CSV")
    args = p.parse_args()
    main(args.csv)
