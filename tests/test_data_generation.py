# %%
from experiments.data_generation import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Initialize the perturbation fields
n_attractors = 2
min_val_sim = 2
n_grid = 40
u_total = np.zeros((n_grid, n_grid))
v_total = np.zeros((n_grid, n_grid))
fixed_pts = np.random.uniform(-min_val_sim, min_val_sim, 2 * n_attractors).reshape(
    -1, 2
)

for i in range(n_attractors):
    fixed_pt = fixed_pts[i]
    print(f"Fixed point: {fixed_pt}")
    U_pert, V_pert, perturb_u, perturb_v = get_random_vector_field_from_ringattractor(
        fixed_pt=fixed_pt,
        add_limit_cycle=True,
        n_grid=n_grid,
        min_val_sim=min_val_sim,
    )
    u_total += U_pert
    v_total += V_pert

# Generate a grid for plotting
x = np.linspace(-min_val_sim, min_val_sim, u_total.shape[1])
y = np.linspace(-min_val_sim, min_val_sim, u_total.shape[0])
X, Y = np.meshgrid(x, y)

# Create the streamline plot
plt.figure(figsize=(10, 6))
plt.streamplot(X, Y, u_total, v_total, density=3, linewidth=0.1, color="b")
plt.title("Streamline Plot of Perturbed Vector Field with Multiple Ring Attractors")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# %%
