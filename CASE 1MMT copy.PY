import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 1) Settings
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2) Load raw data
csv_path = "case1(Reese).csv"
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"{csv_path} not found")
df = pd.read_csv(csv_path)
z_raw = df['z'].values.reshape(-1,1).astype(np.float32)
M_raw = df['M'].values.reshape(-1,1).astype(np.float32)
print(f"Loaded {len(z_raw)} points for z and M")

# 3) Non-dimensionalization scales
k0 = 1.0
z_star = k0**(-0.25)
M_star = np.max(np.abs(M_raw))

# Dimensionless variables
z_bar = z_raw / z_star
M_bar = M_raw / M_star

# 4) Convert to tensors
z_data = torch.tensor(z_bar, dtype=torch.float32, device=device, requires_grad=True)
M_data = torch.tensor(M_bar, dtype=torch.float32, device=device)
z_colloc = z_data.clone()

# 5) Boundary and special point conditions (dimensionless)
# Dirichlet BC at top & bottom
z_bc = torch.tensor(np.vstack([z_bar[0], z_bar[-1]]), dtype=torch.float32, device=device, requires_grad=True)
M_bc = torch.tensor(np.vstack([M_bar[0], M_bar[-1]]), dtype=torch.float32, device=device)

# Slopes for Neumann BC in dimensionless space
dz_top = float(z_raw[1,0] - z_raw[0,0])
slope_top_val = ((M_raw[1,0] - M_raw[0,0]) / dz_top) / M_star * z_star
slope_top = torch.tensor(slope_top_val, dtype=torch.float32, device=device)

dz_bot = float(z_raw[-1,0] - z_raw[-2,0])
slope_bot_val = ((M_raw[-1,0] - M_raw[-2,0]) / dz_bot) / M_star * z_star
slope_bot = torch.tensor(slope_bot_val, dtype=torch.float32, device=device)

# Define the midpoint condition explicitly
mid_index = len(z_raw) // 2
z_midpoint = torch.tensor([[z_bar[mid_index,0]]], dtype=torch.float32, device=device, requires_grad=True)
M_midpoint = torch.tensor([[M_bar[mid_index,0]]], dtype=torch.float32, device=device)

# 6) Define PINN (dimensionless)
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,1)
        )

    def forward(self, x):
        return self.net(x)

model = PINN().to(device)
params = list(model.parameters())

# 7) Physics residual in nondimensional form: d4M/dz_bar4 + M = 0
def moment_residual(z):
    M_pred = model(z)
    d1 = torch.autograd.grad(M_pred, z, grad_outputs=torch.ones_like(M_pred), create_graph=True)[0]
    d2 = torch.autograd.grad(d1, z, grad_outputs=torch.ones_like(d1), create_graph=True)[0]
    d3 = torch.autograd.grad(d2, z, grad_outputs=torch.ones_like(d2), create_graph=True)[0]
    d4 = torch.autograd.grad(d3, z, grad_outputs=torch.ones_like(d3), create_graph=True)[0]
    return d4 + M_pred

# 8) Loss function with midpoint condition
def loss_fn(zc):
    # PDE residual
    loss_pde = torch.mean(moment_residual(zc)**2)
    # Dirichlet BC values
    Mbc_pred = model(z_bc)
    loss_val = torch.mean((Mbc_pred - M_bc)**2)
    # Neumann BC slopes
    dMdz = torch.autograd.grad(Mbc_pred, z_bc, grad_outputs=torch.ones_like(Mbc_pred), create_graph=True)[0]
    loss_slope = (dMdz[0] - slope_top)**2 + (dMdz[1] - slope_bot)**2
    # Midpoint condition
    loss_midpoint = torch.mean((model(z_midpoint) - M_midpoint)**2)
    return loss_pde + 100*(loss_val + loss_slope + loss_midpoint)

# 9) Training
optimizer = torch.optim.Adam(params, lr=1e-3)
for epoch in range(1, 10001):
    optimizer.zero_grad()
    L = loss_fn(z_colloc)
    L.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Adam Epoch {epoch} | Loss={L.item():.3e}")

# L-BFGS fine-tuning
lbfgs = torch.optim.LBFGS(params, max_iter=5000, tolerance_grad=1e-9,
                          tolerance_change=1e-9, history_size=20,
                          line_search_fn='strong_wolfe')

def closure():
    lbfgs.zero_grad()
    loss_val = loss_fn(z_colloc)
    loss_val.backward()
    return loss_val

lbfgs.step(closure)

# 10) Prediction & Plot (back to physical units)
model.eval()
with torch.no_grad():
    M_pred_bar = model(z_data).cpu().numpy()
    M_pred = M_pred_bar * M_star

plt.figure(figsize=(6,8))
plt.plot(M_raw, z_raw, 'k--', lw=2, label='True Data')
plt.plot(M_pred, z_raw, 'g--', lw=2, label='PINN M (nondim)')
plt.gca().invert_yaxis()
plt.xlabel('Moment M')
plt.ylabel('Depth z')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 11) Metrics
rmse = np.sqrt(mean_squared_error(M_raw, M_pred))
r2   = r2_score(M_raw, M_pred)
print(f"RMSE={rmse:.3e}, R2={r2:.4f}")
