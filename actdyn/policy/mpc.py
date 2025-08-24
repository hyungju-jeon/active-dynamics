import numpy as np
from actdyn.environment import action
import torch
import colorednoise

from actdyn.metrics.base import BaseMetric
from actdyn.models.base import BaseModel

from .base import BaseMPC
from actdyn.utils.rollout import RolloutBuffer


class MpcICem(BaseMPC):
    mean: np.ndarray
    std: np.ndarray
    model_evals_per_timestep: int
    elite_samples: RolloutBuffer

    def __init__(
        self,
        metric: BaseMetric,
        model: BaseModel,
        horizon: int = 10,
        num_samples: int = 32,
        num_iterations: int = 10,
        num_elite: int = 100,
        alpha: float = 0.1,
        init_std: float = 0.5,
        noise_beta: float = 1.0,
        factor_decrease_num: float = 1.25,
        frac_prev_elites: float = 0.2,
        frac_elites_reused: float = 0.3,
        use_mean_actions: bool = True,
        shift_elites: bool = True,
        keep_elites: bool = True,
        verbose: bool = False,
        device: str = "cpu",
    ):

        # Call parent constructor with correct arguments
        super().__init__(
            metric=metric,
            model=model,
            horizon=horizon,
            num_samples=num_samples,
            verbose=verbose,
            device=device,
        )

        # Set ICEM-specific parameters
        self.alpha = alpha
        self.num_elites = num_elite
        self.num_iterations = num_iterations
        self.init_std = init_std
        self.noise_beta = noise_beta
        self.factor_decrease_num = factor_decrease_num
        self.frac_prev_elites = frac_prev_elites
        self.frac_elites_reused = frac_elites_reused
        self.use_mean_actions = use_mean_actions
        self.shift_elites = shift_elites
        self.keep_elites = keep_elites

        self.was_reset = False

    def beginning_of_rollout(self, state: torch.Tensor):
        super().beginning_of_rollout(state=state)
        self.mean = self.get_init_mean()
        self.std = self.get_init_std()
        self.elite_samples = RolloutBuffer(device=self.device)
        self.was_reset = True

        self.model_evals_per_timestep = (
            sum(
                [
                    max(
                        self.num_elites * 2,
                        int(self.num_samples / (self.factor_decrease_num**i)),
                    )
                    for i in range(0, self.num_iterations)
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
                    (self.action_bounds[1][dim] + self.action_bounds[0][dim]) / 2.0,
                    device=self.device,
                )
            return mean
        return torch.zeros(self.horizon, self.action_dim, device=self.device)

    def get_init_std(self):
        if self.action_bounds is not None:
            std = torch.ones(self.horizon, self.action_dim, device=self.device)
            for dim in range(self.action_dim):
                std[:, dim] = torch.tensor(
                    (self.action_bounds[1][dim] - self.action_bounds[0][dim]) / 2.0 * self.init_std,
                    device=self.device,
                )
            return std
        return self.init_std * torch.ones(self.horizon, self.action_dim, device=self.device)

    def sample_action_sequences(self, num_samples):
        # Generate action sequences with colored noise
        if self.noise_beta > 0:
            samples = torch.tensor(
                colorednoise.powerlaw_psd_gaussian(
                    self.noise_beta, size=(num_samples, self.action_dim, self.horizon)
                ),
                device=self.device,
                dtype=torch.float32,
            ).transpose(1, 2)
        else:
            samples = torch.randn(num_samples, self.horizon, self.action_dim, device=self.device)
        actions = samples * self.std + self.mean

        if self.action_bounds is not None:
            # Clip each dimension separately
            for dim in range(self.action_dim):
                min_val = float(self.action_bounds[0][dim])
                max_val = float(self.action_bounds[1][dim])
                actions[..., dim] = torch.clamp(actions[..., dim], min_val, max_val)
        return actions

    def simulate(self, initial_state: torch.Tensor, actions: torch.Tensor):
        simulated_paths = torch.zeros(
            actions.shape[0],
            self.horizon + 1,
            initial_state.shape[-1],
            device=self.device,
        )  # (num_samples, horizon + 1, state_dim)
        simulated_paths[:, 0, :] = initial_state[:, 0, :].repeat(actions.shape[0], 1)

        for step in range(self.horizon):
            simulated_paths[:, step + 1] = self.model.dynamics.sample_forward(
                simulated_paths[:, step], self.model.action_encoder(actions[:, step])
            )[0]
        rollout = RolloutBuffer(device=self.device)
        rollout.add_dict(
            {
                "action": actions,
                "model_state": simulated_paths[:, :-1],
                "next_model_state": simulated_paths[:, 1:],
            }
        )
        return rollout

    def get_action(self, state, debug=False):
        if not self.was_reset:
            self.beginning_of_rollout(state)

        best_cost = float("inf")
        best_first_action = None
        costs = [float("inf")]

        current_num_samples = self.num_samples
        for iter in range(self.num_iterations):
            # Decay of sample size
            if iter > 0:
                current_num_samples = max(
                    self.num_elites * 2,
                    int(current_num_samples / self.factor_decrease_num),
                )

            # Sample actions from distribution
            actions = self.sample_action_sequences(current_num_samples)

            # Adding mean actions as candidate at the last iteration
            if self.use_mean_actions and iter == self.num_iterations - 1:
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
            costs = self.metric(rollout).squeeze(-1)

            # Keep elites from previous iteration
            if iter > 0 and self.keep_elites:
                num_elites_to_keep = int(len(self.elite_samples) * self.frac_elites_reused)
                if num_elites_to_keep > 0:
                    prev_elites_actions = self.elite_samples.as_array("action")
                    prev_elite_costs = self.elite_samples.as_array("cost")
                    # Ensure prev_elites_actions has the same shape as actions except for the batch dimension
                    assert (
                        actions.shape[1:] == prev_elites_actions.shape[1:]
                    ), f"Shape mismatch: actions {actions.shape}, prev_elites_actions {prev_elites_actions.shape}"
                    actions = torch.cat([actions, prev_elites_actions[:num_elites_to_keep]], dim=0)
                    # Ensure cost dimensions match except for batch dimension
                    if costs.shape[1:] != prev_elite_costs.shape[1:]:
                        prev_elite_costs = prev_elite_costs.view(
                            prev_elite_costs.shape[0], *costs.shape[1:]
                        )
                    costs = torch.cat([costs, prev_elite_costs[:num_elites_to_keep]], dim=0)

            # Get elite samples
            elite_idxs = torch.topk(-costs, self.num_elites, dim=0)[1]
            elite_actions = actions[elite_idxs]
            elite_costs_traj = costs[elite_idxs]
            self.elite_samples.add_dict(
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
                print(
                    f"iter {iter}:{current_num_samples} "
                    f"--- best cost: {costs.min()} "
                    f"--- mean: {costs.mean()} "
                    f"--- worst: {costs.max()} "
                )

        ### Shift initialization ###
        # Shift mean time-wise
        shifted_mean = self.mean.clone()
        shifted_mean[:-1] = self.mean[1:]
        self.mean = shifted_mean
        self.std = self.get_init_std()

        return best_first_action.unsqueeze(0).unsqueeze(0)  # Return as (1, 1, action_dim)
