# %%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from actdyn.utils.torch_helper import to_np
from torch.nn.functional import softplus

# Small constant to prevent numerical instability
eps = 1e-6


plt.rcParams["text.usetex"] = False
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.size"] = 16
plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts

if __name__ == "__main__":
    """Use Amortized RNN with linear layout to learn mean and variance"""
    mean = 0.0
    noise_var = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # RNN model
    rnn = nn.GRU(input_size=1, hidden_size=5, num_layers=1, batch_first=True, device=device)
    linear = nn.Linear(5, 1, device=device)
    fc_logvar = nn.Sequential(linear).to(device)
    linear = nn.Linear(5, 1, device=device)
    fc_mu = nn.Sequential(linear).to(device)

    # Generate data
    T = 1000
    y = torch.randn(1000, T, 1, device=device) * np.sqrt(noise_var) + mean

    # Train model
    parameters = list(fc_mu.parameters()) + list(fc_logvar.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    for epoch in range(1000):
        optimizer.zero_grad()
        rnn_out, _ = rnn(y)
        mu = fc_mu(rnn_out)
        log_var = fc_logvar(rnn_out)
        var = softplus(log_var) + eps
        kl_d = 0.5 * (torch.log(var) + ((mu - mean) ** 2) / var + (var / noise_var) - 1)
        loss = kl_d.mean()
        loss.backward()
        optimizer.step()
