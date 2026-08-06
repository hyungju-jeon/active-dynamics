from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from actdyn.environment.action import PaddedIdentityActionEncoder
from actdyn.utils.experiment_runtime import (
    apply_loglinear_loading_asymmetry,
    as_bool,
    extract_remaining_plan_actions,
    extract_rollout_metrics,
    paired_diagonal_loglinear_loading,
    to_xy_pair,
    write_trace_csv,
)


class _Policy:
    def __init__(self):
        self.chunk = 3
        self.count = 2
        self.mean = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32)



def test_to_xy_pair_and_as_bool_handle_scalars_and_tensors():
    assert to_xy_pair(torch.tensor([1.5, -2.0])) == (1.5, -2.0)
    assert to_xy_pair(torch.tensor([3.0])) == (3.0, 0.0)
    assert as_bool(torch.tensor([1.0])) is True
    assert as_bool(torch.tensor([0.0])) is False
    assert as_bool('yes') is True
    assert as_bool(None) is False



def test_extract_remaining_plan_actions_uses_policy_chunk_offset():
    plan = extract_remaining_plan_actions(_Policy())
    assert plan is not None
    assert plan.shape == (1, 2, 2)
    assert torch.allclose(plan[0, 0], torch.tensor([3.0, 4.0]))
    assert torch.allclose(plan[0, 1], torch.tensor([5.0, 6.0]))



def test_apply_loglinear_loading_asymmetry_distributes_normal_loading():
    weight = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [3.0, 4.0]])
    env_preset = SimpleNamespace(asymmetric_loading=False)
    out = apply_loglinear_loading_asymmetry(weight, env_preset)
    angles = torch.remainder(torch.atan2(out[:, 1], out[:, 0]), 2.0 * torch.pi)
    sorted_angles = torch.sort(angles).values
    gaps = torch.diff(torch.cat((sorted_angles, sorted_angles[:1] + 2.0 * torch.pi)))
    assert torch.allclose(gaps, torch.full((4,), 0.5 * torch.pi), atol=1e-6)
    assert torch.allclose(torch.linalg.norm(out[:, :2], dim=1), torch.linalg.norm(weight, dim=1))


def test_apply_loglinear_loading_asymmetry_favors_vertical_axis():
    weight = torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=torch.float32)
    env_preset = SimpleNamespace(
        asymmetric_loading=True,
        observation_primary_scale=2.0,
        observation_secondary_scale=3.0,
        observation_row_skew=0.0,
    )
    out = apply_loglinear_loading_asymmetry(weight, env_preset)
    assert out.shape == weight.shape
    assert torch.allclose(out[:, 0], torch.tensor([0.1, -0.3]))
    assert torch.allclose(out[:, 1], torch.tensor([6.0, 12.0]))
    assert torch.all(out[:, 1] >= 0)
    assert out[:, 1].abs().mean() > 20.0 * out[:, 0].abs().mean()


def test_loading_asymmetry_scales_third_nuisance_coordinate():
    weight = torch.ones(2, 3, dtype=torch.float32)
    env_preset = SimpleNamespace(
        asymmetric_loading=True,
        observation_primary_scale=1.0,
        observation_secondary_scale=1.0,
        observation_nuisance_scale=0.02,
        observation_row_skew=0.0,
    )

    out = apply_loglinear_loading_asymmetry(weight, env_preset)

    assert torch.allclose(out[:, 2], torch.full((2,), 0.02))


def test_paired_diagonal_loglinear_loading_matches_configured_fisher():
    preset = SimpleNamespace(
        latent_dim=5,
        observation_dim=160,
        observation_loading_gains=(0.35, 0.35, 0.35, 0.35, 0.035),
        observation_loading_repeats_per_sign=16,
        mean_firing_rate_target=25.51020408163265,
        max_firing_rate_target=210.0,
        boundary_radius=6.0,
    )

    weight, bias = paired_diagonal_loglinear_loading(preset)
    count_at_origin = 0.01 * torch.exp(bias)
    fisher = weight.T @ torch.diag(count_at_origin) @ weight

    assert weight.shape == (160, 5)
    assert bias.shape == (160,)
    assert torch.allclose(
        fisher,
        torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, 0.01])),
        atol=1e-6,
    )
    peak_rate = torch.exp(bias + 6.0 * weight.abs().amax(dim=1))
    assert float(peak_rate.max()) < 210.0


def test_padded_identity_action_leaves_nuisance_uncontrolled():
    encoder = PaddedIdentityActionEncoder(
        d_action=2,
        d_latent=3,
        action_bounds=(-1.0, 1.0),
        device="cpu",
    )
    action = torch.tensor([[[0.25, -0.5]]], dtype=torch.float32)

    encoded = encoder(action)

    assert encoded.shape == (1, 1, 3)
    assert encoded.dtype == torch.float32
    assert torch.allclose(encoded, torch.tensor([[[0.25, -0.5, 0.0]]]))

def test_write_trace_csv_and_extract_rollout_metrics_missing_rollouts(tmp_path: Path):
    path = tmp_path / 'trace.csv'
    write_trace_csv(path, [{'step': 0, 'value': 1.0}], ['step', 'value'])
    assert path.exists()
    metrics = extract_rollout_metrics(tmp_path)
    assert metrics['rollout_steps'] == 0
    assert metrics['state_error_mean'] is None
