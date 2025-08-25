"""
Helper functions for validating reconstruction results in Active Dynamics framework.
"""

from actdyn.environment.observation import LinearObservation, LogLinearObservation
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt

from actdyn.environment import action
from actdyn.models.model import SeqVae
from actdyn.utils.rollout import RolloutBuffer, Rollout


def compare_kstep_observation(
    model: SeqVae,
    rollout: Union[Rollout, RolloutBuffer, Dict],
    k_steps: int = 5,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Compare k-step observation predictions using SeqVae model components.
    """
    # Set model components to eval mode manually to avoid SeqVae.train() conflict
    model.encoder.eval()
    model.decoder.eval()
    model.dynamics.eval()
    if model.action_encoder is not None:
        model.action_encoder.eval()

    device = torch.device(device)
    model.to(device)
    observation = rollout["obs"].unsqueeze(0)
    actions = rollout["action"].unsqueeze(0)

    observation = observation.to(device)
    actions = actions.to(device)

    if actions is not None:
        actions = actions.to(device)

    with torch.no_grad():
        batch_size, seq_len, obs_dim = observation.shape

        # Ensure we have enough observations for k-step prediction
        if seq_len < k_steps + 1:
            raise ValueError(f"Need at least {k_steps + 1} observations, got {seq_len}")

        # Encode initial observations to latent space
        # Use first observation as initial state
        initial_obs = observation[:, 0:1, :]  # [batch_size, 1, obs_dim]

        if actions is not None:
            # Concatenate observation and action for encoder input
            initial_action = actions[:, 0:1, :]  # [batch_size, 1, action_dim]
            encoder_input = torch.cat([initial_obs, initial_action], dim=-1)
        else:
            encoder_input = initial_obs

        # Get initial latent state
        z_samples, mu_z, var_z, _ = model.encoder(encoder_input)
        current_z = z_samples[:, 0, :]  # [batch_size, latent_dim]

        # Store latent trajectory and predictions
        latent_states = [current_z]
        predicted_observations = []

        # Iteratively predict k steps
        for step in range(k_steps):
            # Get action for this step
            if actions is not None and step < actions.shape[1]:
                current_action = actions[:, step, :]  # [batch_size, action_dim]
                # Encode action if action encoder exists
                if model.action_encoder is not None:
                    current_action_encoded = model.action_encoder(current_action)
                else:
                    current_action_encoded = current_action
            else:
                current_action_encoded = None

            # Predict next latent state using dynamics
            # Note: SeqVae dynamics.sample_forward takes (state, action) separately
            if current_action_encoded is not None:
                next_z, _ = model.dynamics.sample_forward(current_z, current_action_encoded)
            else:
                next_z, _ = model.dynamics.sample_forward(current_z)

            # Decode latent state to observation
            pred_obs = model.decoder(next_z)

            # Store results
            latent_states.append(next_z)
            predicted_observations.append(pred_obs)

            # Update current latent state
            current_z = next_z

        # Stack predictions
        predicted_obs = torch.stack(predicted_observations, dim=1)  # [batch_size, k_steps, obs_dim]
        target_obs = observation[:, 1 : k_steps + 1, :]  # [batch_size, k_steps, obs_dim]

        # Compute reconstruction metrics
        mse_loss = torch.mean((predicted_obs - target_obs) ** 2)

        # Per-step losses (average across observation dimensions)
        mse_per_step = torch.mean((predicted_obs - target_obs) ** 2, dim=(0, 2))  # [k_steps]

        # Per-dimension losses (average across time steps)
        mse_per_dim = torch.mean((predicted_obs - target_obs) ** 2, dim=(0, 1))  # [obs_dim]

        metrics = {
            "mse_loss": mse_loss.item(),
            "mse_per_step": mse_per_step.cpu().numpy(),
            "mse_per_dim": mse_per_dim.cpu().numpy(),
        }

    return (metrics, predicted_obs, target_obs)


def rotate_latent(
    env,
    model,
    regularization: float = 1e-6,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Compute linear transformations that map between environment and model latent spaces.

    Returns both env->model and model->env transformations.

    Args:
        env: Environment with observation model
        model: SeqVae model with decoder
        regularization: Regularization parameter for pseudoinverse computation

    Returns:
        env_to_model: Dict with 'rotation' and 'translation' for env->model transform
        model_to_env: Dict with 'rotation' and 'translation' for model->env transform
    """
    if isinstance(env.obs_model, LinearObservation):
        env_weight = env.obs_model.network.weight.data
        env_bias = env.obs_model.network.bias.data
        model_weight = model.decoder.mapping.network.weight.data
        model_bias = model.decoder.mapping.network.bias.data
    elif isinstance(env.obs_model, LogLinearObservation):
        env_weight = env.obs_model.network[0].weight.data
        env_bias = env.obs_model.network[0].bias.data
        model_weight = model.decoder.mapping.network[0].weight.data
        model_bias = model.decoder.mapping.network[0].bias.data

    device = env_weight.device

    # Ensure all tensors are on the same device
    model_weight = model_weight.to(device)
    env_bias = env_bias.to(device)
    model_bias = model_bias.to(device)

    # Compute env->model transformation
    # R_e2m = W_model^+ @ W_env
    W_model_T = model_weight.T  # [latent_dim, obs_dim]
    W_model_T_W_model = W_model_T @ model_weight  # [latent_dim, latent_dim]

    # Add regularization to diagonal
    regularized_matrix_model = W_model_T_W_model + regularization * torch.eye(
        W_model_T_W_model.shape[0], device=device
    )

    # Compute pseudoinverse of model weights
    try:
        W_model_pinv = torch.linalg.solve(
            regularized_matrix_model, W_model_T
        )  # [latent_dim, obs_dim]
    except RuntimeError:
        # Fallback to SVD-based pseudoinverse if solve fails
        W_model_pinv = torch.linalg.pinv(model_weight)  # [latent_dim, obs_dim]

    # Env->Model transformation
    rotation_e2m = W_model_pinv @ env_weight  # [latent_dim, latent_dim]
    bias_diff_e2m = env_bias - model_bias  # [obs_dim]
    translation_e2m = W_model_pinv @ bias_diff_e2m  # [latent_dim]

    # Compute model->env transformation
    # R_m2e = W_env^+ @ W_model
    W_env_T = env_weight.T  # [latent_dim, obs_dim]
    W_env_T_W_env = W_env_T @ env_weight  # [latent_dim, latent_dim]

    # Add regularization to diagonal
    regularized_matrix_env = W_env_T_W_env + regularization * torch.eye(
        W_env_T_W_env.shape[0], device=device
    )

    # Compute pseudoinverse of env weights
    try:
        W_env_pinv = torch.linalg.solve(regularized_matrix_env, W_env_T)  # [latent_dim, obs_dim]
    except RuntimeError:
        # Fallback to SVD-based pseudoinverse if solve fails
        W_env_pinv = torch.linalg.pinv(env_weight)  # [latent_dim, obs_dim]

    # Model->Env transformation
    rotation_m2e = W_env_pinv @ model_weight  # [latent_dim, latent_dim]
    bias_diff_m2e = model_bias - env_bias  # [obs_dim]
    translation_m2e = W_env_pinv @ bias_diff_m2e  # [latent_dim]

    env_to_model = {"rotation": rotation_e2m, "translation": translation_e2m}

    model_to_env = {"rotation": rotation_m2e, "translation": translation_m2e}

    return env_to_model, model_to_env


def apply_latent_transform(
    latent_states: torch.Tensor,
    transform: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Apply computed transformation to latent states.

    Args:
        latent_states: Latent states tensor [batch_size, seq_len, latent_dim]
        transform: Dictionary with 'rotation' and 'translation' tensors

    Returns:
        transformed_states: Transformed latent states [batch_size, seq_len, latent_dim]
    """
    rotation_matrix = transform["rotation"]
    translation = transform["translation"]

    batch_size, seq_len, latent_dim = latent_states.shape

    # Reshape for matrix multiplication
    states_flat = latent_states.view(-1, latent_dim)  # [batch_size * seq_len, latent_dim]

    # Apply transformation: new_state = R @ state + t
    transformed_flat = (
        rotation_matrix @ states_flat.T
    ).T + translation  # [batch_size * seq_len, latent_dim]

    # Reshape back
    transformed_states = transformed_flat.view(batch_size, seq_len, latent_dim)

    return transformed_states


def visualize_reconstruction_comparison(
    predicted_obs: torch.Tensor,
    target_obs: torch.Tensor,
    metrics: dict,
    sample_idx: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize reconstruction comparison for a single sample.

    Args:
        predicted_obs: Predicted observations [batch_size, k_steps, obs_dim]
        target_obs: Target observations [batch_size, k_steps, obs_dim]
        metrics: Metrics dictionary from compare_kstep_observation
        sample_idx: Which sample from batch to visualize
        save_path: Optional path to save the plot
    """
    predicted = predicted_obs[sample_idx].detach().cpu().numpy()  # [k_steps, obs_dim]
    target = target_obs[sample_idx].detach().cpu().numpy()  # [k_steps, obs_dim]

    k_steps, obs_dim = predicted.shape

    # Create subplots for each observation dimension
    fig, axes = plt.subplots(obs_dim, 1, figsize=(10, 3 * obs_dim))
    if obs_dim == 1:
        axes = [axes]

    for dim in range(obs_dim):
        ax = axes[dim]
        steps = np.arange(k_steps)

        ax.plot(steps, target[:, dim], "b-", label="Target", linewidth=2)
        ax.plot(steps, predicted[:, dim], "r--", label="Predicted", linewidth=2)
        ax.fill_between(steps, target[:, dim], predicted[:, dim], alpha=0.3, color="gray")

        ax.set_xlabel("Prediction Step")
        ax.set_ylabel(f"Observation Dim {dim}")
        ax.set_title(f'Dimension {dim} - MSE: {metrics["mse_per_dim"][dim]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"K-Step Reconstruction Comparison (Sample {sample_idx})\n"
        f'Overall MSE: {metrics["mse_loss"]:.4f}, MAE: {metrics["mae_loss"]:.4f}'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compute_latent_alignment_metrics(
    latent1: torch.Tensor,
    latent2: torch.Tensor,
    transform: Dict[str, torch.Tensor],
) -> dict:
    """
    Compute metrics for how well latent spaces align after transformation.

    Args:
        latent1: Original latent states [batch_size, seq_len, latent_dim]
        latent2: Target latent states [batch_size, seq_len, latent_dim]
        transform: Dictionary with 'rotation' and 'translation' tensors

    Returns:
        metrics: Dictionary with alignment metrics
    """
    # Apply transformation to latent1
    transformed_latent1 = apply_latent_transform(latent1, transform)

    rotation_matrix = transform["rotation"]

    # Compute alignment metrics
    mse = torch.mean((transformed_latent1 - latent2) ** 2).item()
    mae = torch.mean(torch.abs(transformed_latent1 - latent2)).item()

    # Compute correlation per dimension
    batch_size, seq_len, latent_dim = latent1.shape
    transformed_flat = transformed_latent1.view(-1, latent_dim).cpu().numpy()
    target_flat = latent2.view(-1, latent_dim).cpu().numpy()

    correlations = []
    for dim in range(latent_dim):
        corr = np.corrcoef(transformed_flat[:, dim], target_flat[:, dim])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0.0)

    # Rotation matrix properties
    det_rotation = torch.det(rotation_matrix).item()
    rotation_norm = torch.norm(rotation_matrix).item()

    return {
        "mse": mse,
        "mae": mae,
        "correlations": correlations,
        "mean_correlation": np.mean(correlations),
        "rotation_determinant": det_rotation,
        "rotation_norm": rotation_norm,
        "is_orthogonal": abs(abs(det_rotation) - 1.0) < 1e-3,  # Check if rotation preserves volume
    }


if __name__ == "__main__":
    """
    Example usage of the helper functions.
    """
    print("Active Dynamics Reconstruction Validation Helper Functions")
    print("This script provides utilities for:")
    print("1. compare_kstep_observation() - Compare k-step predictions with SeqVae")
    print("2. rotate_latent() - Compute latent space transformations between models")
    print("3. Additional utilities for visualization and metrics")
    print("\nImport this module to use the functions in your experiments.")
