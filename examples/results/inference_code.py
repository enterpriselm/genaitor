
import numpy as np
import torch
from torch import nn

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model = PINN()
model.load_state_dict(torch.load('trained_model.pt'))
model.eval()

# Prepare input data for inference
t = torch.linspace(0, 1, 100).reshape(-1, 1)
x = torch.linspace(0, 1, 100).reshape(-1, 1)
y = torch.linspace(0, 1, 100).reshape(-1, 1)
z = torch.linspace(0, 1, 100).reshape(-1, 1)
input_data = torch.cat([t, x, y, z], dim=1)

# Run inference
with torch.no_grad():
    u, v, w, p, T = model(input_data)

# Print or return the output
print(u, v, w, p, T)

# Generate data for comparison
# Boundary conditions
u_bc = torch.zeros(u.shape)  # Zero Dirichlet boundary condition for u
v_bc = torch.zeros(v.shape)  # Zero Dirichlet boundary condition for v
w_bc = torch.zeros(w.shape)  # Zero Dirichlet boundary condition for w
p_bc = torch.zeros(p.shape)  # Zero Dirichlet boundary condition for p
T_bc = torch.zeros(T.shape)  # Zero Dirichlet boundary condition for T

# Governing equations
def governing_equations(u, v, w, p, T, t, x, y, z):
    # Momentum equations
    du_dt = - (u * u_x + v * u_y + w * u_z) + (p_x - (mu/rho) * (u_xx + u_yy + u_zz))
    dv_dt = - (u * v_x + v * v_y + w * v_z) + (p_y - (mu/rho) * (v_xx + v_yy + v_zz))
    dw_dt = - (u * w_x + v * w_y + w * w_z) + (p_z - (mu/rho) * (w_xx + w_yy + w_zz))

    # Continuity equation
    div_u = u_x + v_y + w_z

    # Energy equation
    dT_dt = - (u * T_x + v * T_y + w * T_z) + (k/rho*cp) * (T_xx + T_yy + T_zz)

    return du_dt, dv_dt, dw_dt, div_u, dT_dt

# Calculate the residuals
du_dt, dv_dt, dw_dt, div_u, dT_dt = governing_equations(u, v, w, p, T, t, x, y, z)
residuals = torch.stack([du_dt, dv_dt, dw_dt, div_u, dT_dt])

# Calculate the boundary conditions
u_bc = torch.zeros(u.shape)  # Zero Dirichlet boundary condition for u
v_bc = torch.zeros(v.shape)  # Zero Dirichlet boundary condition for v
w_bc = torch.zeros(w.shape)  # Zero Dirichlet boundary condition for w
p_bc = torch.zeros(p.shape)  # Zero Dirichlet boundary condition for p
T_bc = torch.zeros(T.shape)  # Zero Dirichlet boundary condition for T

# Calculate the loss
loss = torch.mean(residuals**2) + torch.mean(u_bc**2) + torch.mean(v_bc**2) + torch.mean(w_bc**2) + torch.mean(p_bc**2) + torch.mean(T_bc**2)

# Print the loss
print(loss)
