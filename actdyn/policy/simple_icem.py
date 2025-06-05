import torch
import colorednoise as cn


class SimpleICem:
    """Implementation of the Information-Theoretic Cross-Entropy Method (ICEM) with improvements from the original paper."""

    def __init__(
        self,
        action_dim: int,
        horizon: int = 10,
        num_samples: int = 1000,
        num_elite: int = 100,
        alpha: float = 0.1,
        noise_beta: float = 2.0,  # Colored noise exponent
        factor_decrease_num: float = 1.5,  # Population size decay factor
        fraction_elites_reused: float = 0.3,  # Fraction of elites to reuse
        device: str = "cuda",
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elite = num_elite
        self.alpha = alpha
        self.noise_beta = noise_beta
        self.factor_decrease_num = factor_decrease_num
        self.fraction_elites_reused = fraction_elites_reused
        self.device = device

        # Initialize mean and covariance
        self.mean = torch.zeros(horizon, action_dim, device=device)
        self.cov = torch.eye(action_dim, device=device).repeat(horizon, 1, 1)

        # Store previous elites
        self.previous_elites = None
        self.previous_elite_returns = None

    def sample_actions(self) -> torch.Tensor:
        """Sample action sequences using colored noise."""
        # Generate colored noise
        noise = cn.powerlaw_psd_gaussian(
            self.noise_beta, (self.num_samples, self.horizon, self.action_dim)
        )
        noise = torch.tensor(noise, device=self.device)

        # Scale noise by covariance
        actions = torch.zeros(
            self.num_samples, self.horizon, self.action_dim, device=self.device
        )
        for t in range(self.horizon):
            actions[:, t] = self.mean[t] + torch.matmul(
                noise[:, t], torch.linalg.cholesky(self.cov[t])
            )

        # Add mean as a sample
        actions[0] = self.mean

        # Add previous elites if available
        if self.previous_elites is not None:
            num_reuse = int(self.num_elite * self.fraction_elites_reused)
            if num_reuse > 0:
                # Shift previous elites
                shifted_elites = torch.roll(self.previous_elites, -1, dims=1)
                shifted_elites[:, -1] = torch.randn(
                    num_reuse, self.action_dim, device=self.device
                )
                actions[1 : 1 + num_reuse] = shifted_elites

        return actions

    def update_distribution(self, actions: torch.Tensor, returns: torch.Tensor) -> None:
        """Update mean and covariance using elite samples."""
        # Get elite samples
        elite_idx = torch.argsort(returns, descending=True)[: self.num_elite]
        elite_actions = actions[elite_idx]

        # Store elites for next iteration
        self.previous_elites = elite_actions.clone()
        self.previous_elite_returns = returns[elite_idx].clone()

        # Update mean and covariance
        new_mean = elite_actions.mean(dim=0)
        new_cov = torch.zeros_like(self.cov)

        for t in range(self.horizon):
            diff = elite_actions[:, t] - new_mean[t]
            new_cov[t] = torch.matmul(diff.t(), diff) / self.num_elite

        # Smooth update
        self.mean = (1 - self.alpha) * self.mean + self.alpha * new_mean
        self.cov = (1 - self.alpha) * self.cov + self.alpha * new_cov

        # Decay population size
        self.num_samples = max(
            self.num_elite * 2, int(self.num_samples / self.factor_decrease_num)
        )

    def __call__(self, state: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Get action sequence using ICEM."""
        # Sample action sequences
        actions = self.sample_actions()

        # Evaluate returns
        returns = torch.zeros(self.num_samples, device=self.device)

        for i in range(self.num_samples):
            current_state = state.clone()
            total_reward = 0

            for t in range(self.horizon):
                # Get next state prediction
                with torch.no_grad():
                    next_state = model.predict_next_state(current_state, actions[i, t])

                # Compute reward (negative distance from origin)
                reward = -torch.norm(next_state)
                total_reward += reward

                current_state = next_state

            returns[i] = total_reward

        # Update distribution
        self.update_distribution(actions, returns)

        # Return first action of best sequence
        best_idx = torch.argmax(returns)
        return actions[best_idx, 0]
