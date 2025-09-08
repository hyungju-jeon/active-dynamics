# %%
import os
import numpy as np
import torch
from actdyn.config import ExperimentConfig
from actdyn.utils import setup_experiment
from actdyn.utils import save_load
from actdyn.utils import rollout
from actdyn.utils.rollout import Rollout, RolloutBuffer

import matplotlib.pyplot as plt
from actdyn.utils.torch_helper import to_np

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "conf/config.yaml")
    exp_config = ExperimentConfig.from_yaml(config_path)
    results_dir = os.path.dirname(__file__)
    exp_config.results_dir = os.path.join(results_dir, "../../results", "cartpole")

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    if torch.cuda.is_available() and exp_config.device == "cuda":
        torch.set_default_device(exp_config.device)

    # Set up experiment
    experiment, agent, env, model_env = setup_experiment(exp_config)
    model = model_env.model

    # Run the experiment using the Experiment class's run method
    experiment.run()
    # experiment.run(reset=False)
    print(f"Experiment setup completed. Results directory: {results_dir}")

    # print("Performing offline learning...")
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    offline_experiment, agent, _, _ = setup_experiment(exp_config)
    offline_experiment.offline_learning()

    # Memory cleanup after experiment following the project pattern
    if "cuda" in str(exp_config.device):
        torch.cuda.empty_cache()


# %%
# generate traj
ro = Rollout()
state, _ = env.reset()
# env.env.state = (0.0, 0.0, 0.5, 0.0)
state = [env.env._state_to_observation(env.env.state)]
for i in range(10000):
    ro.add(obs=state)
    new_state = env.step(env.action_space.sample() * 0)[0][0]
    ro.add(next_obs=new_state)
    state = new_state
ro.finalize()

# %%
_, _, _, test_env = setup_experiment(exp_config)
optim_cfg = exp_config.training.get_optim_cfg()
optim_cfg.update({"verbose": True, "n_epochs": 10000, "lr": 1e-3, "optimizer": "Adam"})
test_env.train_model(offline_experiment.rollout, batch_size=4, chunk_size=1000, **optim_cfg)


# %%
T = 2500
offline_experiment.rollout.finalize()
y = offline_experiment.rollout["obs"]
# y = ro["obs"]
x_on = model_env.model.encoder(y)[0]
y_on = model_env.model.decoder(x_on)

offline_model = offline_experiment.agent.model_env.model
x_off = offline_model.encoder(y)[0]
y_off = offline_model.decoder(x_off)

fig, axs = plt.subplots(2, 3, figsize=(20, 10))
axs = axs.flatten()
y_labels = [
    r"$\cos(\phi)$",
    r"$\sin(\phi)$",
    r"$\dot{\phi}$",
    r"$\cos(\theta)$",
    r"$\sin(\theta)$",
    r"$\dot{\theta}$",
]
for i in range(6):
    axs[i].plot(to_np(y[0, :T, i]), alpha=0.7, label="y")
    axs[i].plot(to_np(y_on[0, :T, i]), alpha=0.7, label="y_on")
    axs[i].plot(to_np(y_off[0, :T, i]), alpha=0.7, label="y_off")
    axs[i].legend()
    axs[i].set_title(y_labels[i])


# %%
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs = axs.flatten()

theta_y = torch.atan2(y[0, :, 4], y[0, :, 3])
theta_y_on = torch.atan2(y_on[0, :, 4], y_on[0, :, 3])
theta_y_off = torch.atan2(y_off[0, :, 4], y_off[0, :, 3])

phi_y = torch.atan2(y[0, :, 1], y[0, :, 0])
phi_y_on = torch.atan2(y_on[0, :, 1], y_on[0, :, 0])
phi_y_off = torch.atan2(y_off[0, :, 1], y_off[0, :, 0])

T = 10000
axs[0].plot(np.unwrap(to_np(phi_y[:T])), to_np(y[0, :T, 2]), alpha=0.7, label="y")
# axs[1].plot(np.unwrap(to_np(phi_y_on[:T])), to_np(y_on[0, :T, 2]), alpha=0.7, label="y_on")
axs[0].plot(np.unwrap(to_np(phi_y_off[:T])), to_np(y_off[0, :T, 2]), alpha=0.7, label="y_off")
axs[0].legend()
axs[0].set_title(r"$\phi$")

axs[1].plot(np.unwrap(to_np(theta_y[:T])), to_np(y[0, :T, 5]), alpha=0.7, label="y")
# axs[0].plot(np.unwrap(to_np(theta_y_on[:T])), to_np(y_on[0, :T, 5]), alpha=0.7, label="y_on")
axs[1].plot(np.unwrap(to_np(theta_y_off[:T])), to_np(y_off[0, :T, 5]), alpha=0.7, label="y_off")
axs[1].legend()
axs[1].set_title(r"$\theta$")

# %% K-step
k = 10
T = 2500

y_kstep = offline_model.decoder(offline_model.dynamics.sample_forward(x_off, None, k)[0])
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
axs = axs.flatten()
y_labels = [
    r"$\cos(\phi)$",
    r"$\sin(\phi)$",
    r"$\dot{\phi}$",
    r"$\cos(\theta)$",
    r"$\sin(\theta)$",
    r"$\dot{\theta}$",
]
for i in range(6):
    axs[i].plot(to_np(y[0, k : T + k, i]), alpha=0.7, label="y")
    axs[i].plot(to_np(y_off[0, k : T + k, i]), alpha=0.7, label="y_off")
    axs[i].plot(to_np(y_kstep[0, :T, i]), alpha=0.7, label="y_kstep")
    axs[i].legend()
    axs[i].set_title(y_labels[i])
