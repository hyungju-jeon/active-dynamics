import numpy as np
import torch
import colorednoise
from gym import spaces
from warnings import warn

from actdyn.policy.base import BaseMPC
from actdyn.utils.rollout import RolloutBuffer
from actdyn.utils.logger import Logger
from actdyn.config import PolicyConfig


class MpcICem(BaseMPC):
    mean: np.ndarray
    std: np.ndarray
    model_evals_per_timestep: int
    elite_samples: RolloutBuffer

    def __init__(self, *, cost_fn, model, icem_params: PolicyConfig, **kwargs):
        # Extract required arguments from kwargs
        mpc_params = kwargs.pop("mpc_params")

        # Convert PolicyConfig to dictionary for parent class
        mpc_params_dict = {
            "horizon": icem_params.horizon,
            "num_samples": icem_params.num_samples,
            "num_iterations": icem_params.num_iterations,
            "num_elite": icem_params.num_elite,
            "alpha": icem_params.alpha,
            "device": icem_params.device,
        }

        # Call parent constructor with correct arguments
        super().__init__(
            cost_fn=cost_fn, model=model, mpc_params=mpc_params_dict, **kwargs
        )

        # Set ICEM-specific parameters
        self.alpha = getattr(icem_params, "alpha", 0.1)
        self.num_elites = getattr(icem_params, "num_elites", 10)
        self.opt_iter = getattr(icem_params, "opt_iterations", 40)
        self.init_std = getattr(icem_params, "init_std", 0.5)
        self.noise_beta = getattr(icem_params, "noise_beta", 1.0)

        self.frac_prev_elites = getattr(icem_params, "frac_prev_elites", 0.2)
        self.factor_decrease_num = getattr(icem_params, "factor_decrease_num", 1.25)
        self.frac_elites_reused = getattr(icem_params, "frac_elites_reused", 0.3)
        self.use_mean_actions = getattr(icem_params, "use_mean_actions", True)
        self.shift_elites = getattr(icem_params, "shift_elites", True)
        self.keep_elites = getattr(icem_params, "keep_elites", True)

        self.logger = Logger()
        self.was_reset = False

    def beginning_of_rollout(self, state: torch.Tensor):
        super().beginning_of_rollout(state=state)
        self.mean = self.get_init_mean()
        self.std = self.get_init_std()
        self.elite_samples = RolloutBuffer()
        self.was_reset = True

        self.model_evals_per_timestep = (
            sum(
                [
                    max(
                        self.num_elites * 2,
                        int(self.num_samples / (self.factor_decrease_num**i)),
                    )
                    for i in range(0, self.opt_iter)
                ]
            )
            * self.horizon
        )

        print(
            f"iCEM using {self.model_evals_per_timestep} evaluations per step "
            f"and {self.model_evals_per_timestep / self.horizon} trajectories per step"
        )

    def end_of_rollout(self, total_time, total_return, mode):
        super().end_of_rollout(total_time, total_return, mode)

    def get_init_mean(self):
        if self.action_bounds is not None:
            mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
            for dim in range(self.action_dim):
                mean[:, dim] = torch.tensor(
                    (self.action_bounds[dim][1] + self.action_bounds[dim][0]) / 2.0,
                    device=self.device,
                )
            return mean
        return torch.zeros(self.horizon, self.action_dim, device=self.device)

    def get_init_std(self):
        if self.action_bounds is not None:
            std = torch.ones(self.horizon, self.action_dim, device=self.device)
            for dim in range(self.action_dim):
                std[:, dim] = torch.tensor(
                    (self.action_bounds[dim][1] - self.action_bounds[dim][0])
                    / 2.0
                    * self.init_std,
                    device=self.device,
                )
            return std
        return torch.ones(self.horizon, self.action_dim, device=self.device)

    def sample_action_sequences(self, num_samples):
        # Generate action sequences with colored noise
        if self.noise_beta > 0:
            assert self.mean.ndim == 2
            samples = torch.tensor(
                colorednoise.powerlaw_psd_gaussian(
                    self.noise_beta, size=(num_samples, self.action_dim, self.horizon)
                ),
                device=self.device,
                dtype=torch.float32,
            ).transpose(1, 2)
        else:
            samples = torch.randn(
                num_samples, self.horizon, self.action_dim, device=self.device
            )
        actions = samples * self.std + self.mean

        if self.action_bounds is not None:
            # Clip each dimension separately
            for dim in range(self.action_dim):
                min_val = float(self.action_bounds[dim][0])
                max_val = float(self.action_bounds[dim][1])
                actions[..., dim] = torch.clamp(actions[..., dim], min_val, max_val)
        return actions

    def simulate(self, initial_state: torch.Tensor, actions: torch.Tensor):
        simulated_paths = torch.zeros(
            actions.shape[0],
            self.horizon + 1,
            initial_state.shape[-1],
            device=self.device,
        )
        simulated_paths[:, 0] = initial_state.repeat(actions.shape[0], 1)

        for step in range(self.horizon):
            simulated_paths[:, step + 1] = self.model.dynamics(
                simulated_paths[:, step]
            ) + self.model.action_encoder(actions[:, step])
        rollout = RolloutBuffer()
        rollout.from_dict(
            {
                "action": actions,
                "model_state": simulated_paths[:, :-1],
                "next_model_state": simulated_paths[:, 1:],
            }
        )
        return rollout

    def get_action(self, state):
        if not self.was_reset:
            raise AttributeError("beginning_of_rollout() needs to be called before")

        best_cost = float("inf")
        best_first_action = None
        costs = [float("inf")]

        current_num_samples = self.num_samples
        for iter in range(self.opt_iter):
            # Decay of sample size
            if iter > 0:
                current_num_samples = max(
                    self.num_elites * 2,
                    int(current_num_samples / self.factor_decrease_num),
                )

            # Sample actions from distribution
            actions = self.sample_action_sequences(current_num_samples)

            # Adding mean actions as candidate at the last iteration
            if self.use_mean_actions and iter == self.opt_iter - 1:
                actions[0] = self.mean

            # Shifting elites over time
            if iter == 0 and self.shift_elites and self.elite_samples:
                elites_actions = self.elite_samples.as_array("action")
                reused_actions = elites_actions[:, 1:]
                num_elites = int(reused_actions.shape[0] * self.frac_elites_reused)
                reused_actions = reused_actions[:num_elites]
                last_actions = self.sample_action_sequences(num_elites)[:, -1:]
                elites_actions = torch.cat([reused_actions, last_actions], dim=1)
                actions = torch.cat([actions, elites_actions], dim=0)

            # Simulate and Compute Cost
            rollout = self.simulate(state, actions)
            cost_traj = self.cost_fn(rollout)
            costs = cost_traj.sum(dim=1)

            # Keep elites from previous iteration
            if iter > 0 and self.keep_elites:
                num_elites_to_keep = int(
                    len(self.elite_samples) * self.frac_elites_reused
                )
                if num_elites_to_keep > 0:
                    prev_elites_actions = self.elite_samples.as_array("action")
                    prev_elite_costs = self.elite_samples.as_array("cost")
                    # Ensure prev_elites_actions has the same shape as actions except for the batch dimension
                    assert (
                        actions.shape[1:] == prev_elites_actions.shape[1:]
                    ), f"Shape mismatch: actions {actions.shape}, prev_elites_actions {prev_elites_actions.shape}"
                    actions = torch.cat(
                        [actions, prev_elites_actions[:num_elites_to_keep]], dim=0
                    )
                    # Ensure cost dimensions match except for batch dimension
                    if cost_traj.shape[1:] != prev_elite_costs.shape[1:]:
                        prev_elite_costs = prev_elite_costs.view(
                            prev_elite_costs.shape[0], *cost_traj.shape[1:]
                        )
                    cost_traj = torch.cat(
                        [cost_traj, prev_elite_costs[:num_elites_to_keep]], dim=0
                    )

            # Get elite samples
            elite_idxs = torch.topk(-costs, self.num_elites, dim=0)[1]
            elite_actions = actions[elite_idxs]
            elite_costs_traj = cost_traj[elite_idxs]
            self.elite_samples.from_dict(
                {
                    "action": elite_actions,
                    "cost": elite_costs_traj,
                }
            )

            # Update best first action if we found better solution
            min_cost_idx = elite_idxs[0]
            if costs[min_cost_idx] < best_cost:
                best_cost = costs[min_cost_idx]
                best_first_action = actions[min_cost_idx, 0]

            # Update mean and std
            new_mean = elite_actions.mean(dim=0)
            new_std = elite_actions.std(dim=0)

            self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean
            self.std = (1 - self.alpha) * new_std + self.alpha * self.std

            # Print cost for debugging
            if self.verbose:
                best_actions = self.elite_samples.as_array("action")[0]
                print(
                    f"iter {iter}:{current_num_samples} "
                    f"--- best cost: {costs.min()} "
                    f"--- mean: {costs.mean()} "
                    f"--- worst: {costs.max()} "
                    f"--- best action: {best_actions[0:6]}..."
                )

        ### Shift initialization ###
        # Shift mean time-wise
        shifted_mean = self.mean.clone()
        shifted_mean[:-1] = self.mean[1:]
        self.mean = shifted_mean
        self.std = self.get_init_std()

        self.logger.log(key="Expected_trajectory_cost", value=best_cost)

        return best_first_action
