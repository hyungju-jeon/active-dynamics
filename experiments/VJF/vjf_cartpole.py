#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from vjf.model import VJF
from actdyn.environment.cartpole import ContinuousCartPoleEnv

#%% configuration / seeds
torch.set_default_dtype(torch.double)
np.random.seed(0)
torch.manual_seed(0)

RENDER = False

#%% environment / data collection params
env = ContinuousCartPoleEnv(dt=0.001)
N_steps = 1000          # total timesteps to collect
window = 20             # chunk size for online updating

# collect observations (y) and true states (x_true)
ys = []
x_trues = []

# reset env
obs, _ = env.reset()
env.phi = np.random.uniform(0, 2*np.pi) # random angle between 0 and 2pi
env.theta = 0.0
env.phi_dot, env.theta_dot = 0.0, 0.0
for t in range(N_steps):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    ys.append(obs.astype(np.float32))
    # access internal true state if available
    if env.state is None:
        x_trues.append(np.zeros(4, dtype=np.float32))
    else:
        phi, phi_dot, theta, theta_dot = env.state
        x_trues.append(np.array([phi, phi_dot, theta, theta_dot], dtype=np.float32))

    if done:
        obs, _ = env.reset()

ys = np.vstack(ys)            # shape (N_steps, ydim)
x_trues = np.vstack(x_trues)  # shape (N_steps, xdim)

env.close()

#%% plot observation dimensions through time

plt.figure(figsize=(10,6))
for i in range(ys.shape[1]):
    plt.subplot(3, 2, i+1)
    plt.plot(ys[:, i], label=f"obs dim {i+1}")
    plt.xlabel("Timestep")
    plt.ylabel(f"Obs dim {i+1}")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.suptitle("CartPole observations over time")
plt.subplots_adjust(top=0.92)
plt.savefig('cartpole_observations.png')


#%% VJF model setup
ydim = ys.shape[1]       # 6 dim
# xdim = x_trues.shape[1]  # 4 (phi, phi_dot, theta, theta_dot)
xdim = 6
udim = 0

n_rbf = 50
hidden_sizes = [20]
likelihood = 'gaussian'

model = VJF.make_model(ydim, xdim, udim=udim,
                       n_rbf=n_rbf,
                       hidden_sizes=hidden_sizes,
                       likelihood=likelihood)

#%% online training loop
posterior_means = []

# convert observations to torch dtype double up front for convenience
ys_torch = torch.tensor(ys, dtype=torch.double)

for start in range(0, len(ys), window):
    y_batch = ys_torch[start:start+window] # shape (window, ydim)

    # fit incrementally
    m, logvar, _ = model.fit(y_batch, max_iter=30)

    # m_np = m.detach().cpu().numpy().squeeze()
    # posterior_means.append(m_np)

    m_np = m.mean(dim=0).detach().cpu().numpy()
    posterior_means.append(m_np)

# %%

posterior_means = np.vstack(posterior_means)
time_batches = np.arange(len(posterior_means))
true_at_batches = x_trues

#%% forecast from the last estimated state
x0 = torch.tensor(posterior_means[-1], dtype=torch.double)
n_forecast = 100
x_forecast, y_forecast = model.forecast(x0=x0, n_step=n_forecast, noise=False)
x_forecast = x_forecast.detach().cpu().numpy().squeeze()  # (n_forecast, xdim)

# %% plotting
n_batches = posterior_means.shape[0]
# The below line correctly calculates the start times for each batch
batch_starts = np.arange(0, len(ys), window)[:n_batches]
time_batches = batch_starts
true_at_batches = x_trues[batch_starts]
full_time = np.arange(len(ys))

plt.figure(figsize=(12, 6))

# Plot each latent dimension (true sampled at batches vs posterior mean)
for dim in range(4):
    plt.subplot(2, 2, dim+1)
    plt.plot(time_batches, true_at_batches[:, dim], label=f"True dim {dim+1}", alpha=0.7)
    plt.plot(time_batches, posterior_means[:, dim], label=f"Posterior mean dim {dim+1}", linestyle='--')
    plt.xlabel("Timestep")
    plt.ylabel(f"Latent dim {dim+1}")
    plt.legend()
    plt.grid(True)

plt.suptitle("Online inference per latent dimension")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('latent_dimensions_plot.png')



# # forecast subplot: first two dims as example
# plt.figure(figsize=(8,4))
# plt.plot(np.arange(n_forecast+1), x_forecast[:, 0], label="Forecast latent dim 1")
# plt.plot(np.arange(n_forecast+1), x_forecast[:, 2], label="Forecast latent dim 3 (theta)")
# plt.title("Forecast from the last estimated latent state")
# plt.xlabel("Forecast step")
# plt.legend()
# plt.tight_layout()
# plt.show()
