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

    def __init__(self, *, icem_params: PolicyConfig, **kwargs):
        super().__init__(icem_params, **kwargs)
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
        self.mean = self.get_init_mean(True)
        self.std = self.get_init_std(True)
        self.elite_rollouts = RolloutBuffer()
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
            return (
                torch.zeros(self.horizon, self.action_dim, device=self.device)
                + (self.action_bounds[1] + self.action_bounds[0]) / 2.0
            )
        return torch.zeros(self.horizon, self.action_dim, device=self.device)

    def get_init_std(self):
        if self.action_bounds is not None:
            return (
                torch.ones(self.horizon, self.action_dim, device=self.device)
                * (self.action_bounds[1] - self.action_bounds[0])
                / 2.0
                * self.init_std
            )
        return self.init_std * torch.ones(
            self.horizon, self.action_dim, device=self.device
        )

    def sample_action_sequences(self, num_samples):
        # Generate action sequences with colored noise
        if self.noise_beta > 0:
            assert self.mean.ndim == 2
            samples = torch.tensor(
                colorednoise.powerlaw_psd_gaussian(
                    self.noise_beta, size=(num_samples, self.horizon, self.action_dim)
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
            actions = torch.clamp(actions, self.action_bounds[0], self.action_bounds[1])
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
        return simulated_paths

    def get_action(self, state):
        if not self.was_reset:
            raise AttributeError("beginning_of_rollout() needs to be called before")

        best_traj_idx = None
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
                last_actions = self.sample_action_sequences(num_elites)[:, -1]
                elites_actions = torch.cat([reused_actions, last_actions], dim=1)
                actions = torch.cat([actions, elites_actions], dim=0)

            # Simulate and Compute Cost
            simulated_paths = self.simulate(state, actions)
            costs = self.cost_fn(simulated_paths)

            # Keep elites from previous iteration
            if iter > 0 and self.keep_elites:
                num_elites_to_keep = int(
                    len(self.elite_samples) * self.frac_elites_reused
                )
                if num_elites_to_keep > 0:
                    prev_elites_actions = self.elite_samples.as_array("action")
                    prev_elite_costs = self.elite_samples.as_array("cost")
                    actions = torch.cat(
                        [actions, prev_elites_actions[:num_elites_to_keep]], dim=0
                    )
                    costs = torch.cat(
                        [costs, prev_elite_costs[:num_elites_to_keep]], dim=0
                    )

            # Get elite samples
            elite_idxs = torch.topk(-costs, self.num_elites)[1]
            elite_actions = actions[elite_idxs]
            elite_costs = costs[elite_idxs]
            self.elite_samples.add(

            # Update mean and std
            new_mean = elite_actions.mean(dim=0)
            new_std = elite_actions.std(dim=0)

            self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean
            self.std = (1 - self.alpha) * new_std + self.alpha * self.std

            # Print cost for debugging
            if self.verbose:

                def display_cost(cost):
                    return (
                        cost / self.horizon
                        if self.cost_along_trajectory == "sum"
                        else cost
                    )

                best_actions = simulated_paths[best_traj_idx]["actions"][0]
                print(
                    "iter {}:{} --- best cost: {:.2f} --- mean: {:.2f} --- worst: {:.2f}  best action: {}...".format(
                        i,
                        num_sim_traj,
                        display_cost(np.amin(costs)),
                        display_cost(np.mean(costs)),
                        display_cost(np.amax(costs)),
                        best_actions[0:6],
                    )
                )
            self.update_distributions(simulated_paths, costs)

            # end of inner loop

        executed_action = simulated_paths[best_traj_idx]["actions"][0]

        ### Shift initialization ###
        # Shift mean time-wise
        self.mean[:-1] = self.mean[1:]

        # compute new action (default is to preserve the last one)
        last_predicted_ob = simulated_paths[best_traj_idx]["observations"][-1]
        self.mean[-1] = self.compute_new_mean(obs=last_predicted_ob)
        ############################

        ### initialization of std dev ###
        self.std = self.get_init_std(True)

        self.logger.log(min(costs), key="Expected_trajectory_cost")
        # self.logger.log(min(costs) / self.horizon, key="Expected average cost")

        if self.do_visualize_plan:
            viz_obs = simulated_paths[best_traj_idx]["observations"]
            acts = simulated_paths[best_traj_idx]["actions"]
            self.visualize_plan(obs=viz_obs, state=self.forward_model_state, acts=acts)

        # for stateful models, actually simulate step (forward model stores the state internally)
        if self.forward_model_state is not None:
            obs_, self.forward_model_state, rewards = self.forward_model.predict(
                observations=obs,
                states=self.forward_model_state,
                actions=executed_action,
            )
        return executed_action

    def compute_new_mean(self, obs):
        return self.mean[-1]
