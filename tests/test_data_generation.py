# %%
from experiments.data_generation import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Initialize the perturbation fields
min_val_sim = 2
n_grid = 40

U_pert, V_pert, perturb_u, perturb_v = get_random_vector_field_from_ringattractor(
    add_limit_cycle=False,
    n_grid=n_grid,
    min_val_sim=min_val_sim,
)
# Generate a grid for plotting
x = np.linspace(-min_val_sim, min_val_sim, n_grid)
y = np.linspace(-min_val_sim, min_val_sim, n_grid)
X, Y = np.meshgrid(x, y)

# Create the streamline plot
plt.figure(figsize=(10, 6))
plt.streamplot(X, Y, perturb_u, perturb_v, density=5, linewidth=0.1, color="b")
plt.title("Streamline Plot of Perturbed Vector Field with Multiple Ring Attractors")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# %%
