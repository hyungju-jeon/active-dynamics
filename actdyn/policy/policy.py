"""Policy implementation for active learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .base import BasePolicy


class Policy(BasePolicy):
    """Policy network for active learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[list] = None,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(
            state_dim=state_dim, action_dim=action_dim, device=device, **kwargs
        )

        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Policy network
        policy_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            policy_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim

        policy_layers.append(nn.Linear(hidden_dims[-1], action_dim))
        self.policy_network = nn.Sequential(*policy_layers)

        # Value network
        value_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            value_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim

        value_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.value_network = nn.Sequential(*value_layers)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the policy network."""
        action = self.policy_network(state)
        value = self.value_network(state)

        return {"action": action, "value": value}

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get action for given state."""
        return self.policy_network(state)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate for given state."""
        return self.value_network(state)

    def loss_function(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute policy loss."""
        # Get action logits and values
        outputs = self.forward(states)
        action_logits = outputs["action"]
        values = outputs["value"]

        # Policy loss (negative log likelihood)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        policy_loss = -(action_log_probs * actions * advantages.unsqueeze(-1)).sum()

        # Value loss (MSE)
        value_loss = F.mse_loss(values, returns)

        # Total loss
        total_loss = policy_loss + value_loss

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        }
