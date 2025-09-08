# %%

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from vjf.model import VJF

# %%

# Setup precision and random seeds
torch.set_default_dtype(torch.double)
np.random.seed(0)
torch.manual_seed(0)

# Generate data
T = 100.
dt = 1e-2 * math.pi
xdim = 2
ydim = 20
udim = 0

C = torch.randn(xdim, ydim)
d = torch.randn(ydim)

t = torch.arange(0, T, step=dt)
x = torch.column_stack((torch.sin(t), torch.cos(t))) + torch.randn(len(t), 2) * 0.1
y = x @ C + d + torch.randn(len(t), ydim) * 0.1

# %%

# Setup VJF
n_rbf = 50
hidden_sizes = [20]
likelihood = 'gaussian'

model = VJF.make_model(ydim, xdim, udim=udim,
                       n_rbf=n_rbf,
                       hidden_sizes=hidden_sizes,
                       likelihood=likelihood)

# %%

# Online training loop
window = 20   # process data in chunks of this size
posterior_means = []

for start in range(0, len(y), window):
    y_batch = y[start:start+window]

    # fit incrementally (parameters update inside model)
    m, logvar, _ = model.fit(y_batch, max_iter=30)

    posterior_means.append(m.detach().numpy().squeeze())

posterior_means = np.vstack(posterior_means)

# %%

# Plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)

# true state
plt.plot(x[:, 0].numpy(), alpha=0.6, label="True state dim 1")
plt.plot(x[:, 1].numpy(), alpha=0.6, label="True state dim 2")

# posterior mean
plt.plot(posterior_means[:, 0], label="Posterior mean dim 1")
plt.plot(posterior_means[:, 1], label="Posterior mean dim 2")

plt.legend()
plt.title("Online inference (per dimension)")
plt.xlabel("Time")
plt.ylabel("Latent value")

# Forecast from last estimated state
plt.subplot(1,2,2)
x_forecast, y_forecast = model.forecast(
    x0=posterior_means[-1], n_step=25, noise=False
)
x_forecast = x_forecast.detach().numpy().squeeze()
plt.plot(x_forecast[:, 0], label="Forecast dim 1")
plt.plot(x_forecast[:, 1], label="Forecast dim 2")
plt.title("Forecast from online learned model")
plt.legend()
plt.tight_layout()
plt.show()
