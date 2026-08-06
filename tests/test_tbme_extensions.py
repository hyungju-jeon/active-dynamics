from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_current_tbme_catalog_configures_initial_parameter_distribution():
    from experiments.experiment_definitions import (
        DEFAULT_MODEL_CATALOG_PATHS,
        DEFAULT_SUITE_CATALOG_PATHS,
        configure_catalogs,
        get_environment_preset,
    )

    env_paths = [
        REPO_ROOT / "experiments" / "experiment_env.yaml",
        REPO_ROOT / "experiments" / "tbme" / "config" / "experiment_env.yaml",
    ]
    configure_catalogs(
        env_catalog_paths=env_paths,
        model_catalog_paths=DEFAULT_MODEL_CATALOG_PATHS,
        suite_catalog_paths=DEFAULT_SUITE_CATALOG_PATHS,
    )
    try:
        preset = get_environment_preset("tbme_gated_duffing")
        assert np.allclose(preset.initial_parameter_mean_vector(), np.zeros(4))
        assert preset.initial_parameter_variance == pytest.approx(0.25)
    finally:
        configure_catalogs()


def test_confounded_gate_matches_hidden_nuisance_design():
    from actdyn.environment.vectorfield import (
        jacobian_param_torch,
        jacobian_state_torch,
    )

    params = torch.tensor([0.5], dtype=torch.float32)
    states = torch.tensor([[-0.5, 0.0, 0.25], [-0.32, 0.0, 0.25]], dtype=torch.float32)

    param_jacobian = jacobian_param_torch(
        "confounded_gate",
        states,
        params.expand(states.shape[0], -1),
        dynamics_alpha=1.0,
    )
    state_jacobian = jacobian_state_torch(
        "confounded_gate", states, params, dynamics_alpha=1.0
    )

    assert param_jacobian.shape == (2, 3, 1)
    assert torch.allclose(param_jacobian[:, (0, 2), 0], torch.zeros(2, 2))
    assert param_jacobian[0, 1, 0] == pytest.approx(20.0, abs=1e-3)
    assert param_jacobian[1, 1, 0] == pytest.approx(10.0, abs=2e-3)
    assert state_jacobian.shape == (2, 3, 3)
    assert float(state_jacobian[0, 1, 2].detach()) == pytest.approx(20.0, abs=1e-5)
    assert abs(float(state_jacobian[1, 1, 2].detach())) < 2e-3
    assert torch.allclose(state_jacobian[:, 2], torch.zeros(2, 3))


def test_confounded_gate_canonical_rollouts_separate_objectives():
    from actdyn.environment.vectorfield import (
        jacobian_param_torch,
        jacobian_state_torch,
    )
    from actdyn.utils.torch_utils import attenuated_state_information

    params = torch.tensor([0.5], dtype=torch.float32)
    state_by_gate = {
        "ambiguity": torch.tensor([[-0.5, 0.0, 0.0]], dtype=torch.float32),
        "informative": torch.tensor([[-0.32, 0.0, 0.0]], dtype=torch.float32),
    }
    dt = 0.2
    horizon = 40
    state_information = torch.diag(torch.tensor([0.0, 1.0, 0.0]))
    eye = torch.eye(3)
    scores = {}

    for gate, state in state_by_gate.items():
        f_theta = jacobian_param_torch(
            "confounded_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        f_state = jacobian_state_torch(
            "confounded_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        sensitivity = torch.zeros(3, 1)
        covariance = 4.0 * eye
        gate_scores = {
            "paldi": 0.0,
            "fully_observable": 0.0,
            "dynamics": 0.0,
            "state_information": 0.0,
        }
        for _ in range(horizon):
            transition = eye + f_state * dt
            sensitivity = transition @ sensitivity + f_theta * dt
            attenuated = attenuated_state_information(covariance, state_information)
            gate_scores["paldi"] += float(
                (sensitivity.T @ attenuated @ sensitivity).item()
            )
            gate_scores["fully_observable"] += float(
                (sensitivity.T @ state_information @ sensitivity).item()
            )
            gate_scores["dynamics"] += float(
                (sensitivity.T @ covariance @ sensitivity).item()
            )
            gate_scores["state_information"] += float(
                0.5 * torch.logdet(eye + covariance @ state_information).item()
            )
            covariance = transition @ covariance @ transition.T + 0.01 * eye
        scores[gate] = gate_scores

    assert scores["informative"]["paldi"] > scores["ambiguity"]["paldi"]
    for ablation in ("fully_observable", "dynamics", "state_information"):
        assert scores["ambiguity"][ablation] > scores["informative"][ablation]


def test_rank_imbalanced_gate_separates_logdet_and_e_optimality():
    from actdyn.environment.vectorfield import (
        jacobian_param_torch,
        jacobian_state_torch,
    )
    from actdyn.utils.torch_utils import attenuated_state_information

    params = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    state_by_gate = {
        "balanced": torch.tensor([[-0.5, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "main": torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    }
    dt = 0.2
    horizon = 40
    eye_state = torch.eye(4)
    eye_parameter = torch.eye(3)
    state_information = torch.diag(torch.tensor([0.0, 1.0, 1.0, 1.0]))
    scores = {}

    for gate, state in state_by_gate.items():
        f_theta = jacobian_param_torch(
            "rank_imbalanced_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        f_state = jacobian_state_torch(
            "rank_imbalanced_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        sensitivity = torch.zeros(4, 3)
        covariance = 0.25 * eye_state
        information = torch.zeros(3, 3)
        for _ in range(horizon):
            transition = eye_state + f_state * dt
            sensitivity = transition @ sensitivity + f_theta * dt
            attenuated = attenuated_state_information(covariance, state_information)
            information += sensitivity.T @ attenuated @ sensitivity
            covariance = transition @ covariance @ transition.T + 0.01 * eye_state

        scores[gate] = {
            "paldi": float(0.5 * torch.logdet(eye_parameter + information).item()),
            "e_optimality": float(torch.linalg.eigvalsh(information)[0].item()),
        }

    assert scores["main"]["paldi"] > scores["balanced"]["paldi"]
    assert scores["balanced"]["e_optimality"] > scores["main"]["e_optimality"]


def test_compound_tri_gate_canonical_fixed_gate_rankings():
    """Each objective selects its intended gate under the configured state Fisher."""
    from actdyn.environment.vectorfield import (
        jacobian_param_torch,
        jacobian_state_torch,
    )
    from actdyn.utils.torch_utils import attenuated_state_information

    params = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
    state_by_gate = {
        "ambiguity": torch.tensor([[-0.5, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "balanced": torch.tensor([[-0.32, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "main": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    }
    dt = 0.1
    horizon = 40
    eye_state = torch.eye(5)
    eye_parameter = torch.eye(3)
    state_information = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, 0.01]))
    scores = {}

    for gate, state in state_by_gate.items():
        f_theta = jacobian_param_torch(
            "compound_tri_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        f_state = jacobian_state_torch(
            "compound_tri_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        sensitivity = torch.zeros(5, 3)
        covariance = eye_state.clone()
        information = torch.zeros(3, 3)
        fully_observed_information = torch.zeros(3, 3)
        dynamics_score = 0.0
        state_information_score = 0.0
        for _ in range(horizon):
            transition = eye_state + f_state * dt
            sensitivity = transition @ sensitivity + f_theta * dt
            attenuated = attenuated_state_information(covariance, state_information)
            information += sensitivity.T @ attenuated @ sensitivity
            fully_observed_information += (
                sensitivity.T @ state_information @ sensitivity
            )
            dynamics_score += float(
                torch.trace(sensitivity.T @ covariance @ sensitivity)
            )
            state_information_score += float(
                0.5 * torch.logdet(eye_state + covariance @ state_information)
            )
            covariance = transition @ covariance @ transition.T + 0.01 * eye_state

        scores[gate] = {
            "paldi": float(0.5 * torch.logdet(eye_parameter + information)),
            "fully_observable": float(
                0.5 * torch.logdet(eye_parameter + fully_observed_information)
            ),
            "e_optimality": float(torch.linalg.eigvalsh(information)[0]),
            "dynamics": dynamics_score,
            "state_information": state_information_score,
            "variance": float(torch.trace(f_theta.T @ f_theta)),
        }

    assert scores["main"]["paldi"] > scores["ambiguity"]["paldi"]
    assert scores["main"]["paldi"] > scores["balanced"]["paldi"]
    assert scores["ambiguity"]["fully_observable"] > scores["main"]["fully_observable"]
    assert scores["balanced"]["e_optimality"] > scores["main"]["e_optimality"]
    for objective in ("dynamics", "state_information", "variance"):
        assert scores["ambiguity"][objective] > scores["main"][objective]

    ambiguity_param = jacobian_param_torch(
        "compound_tri_gate", state_by_gate["ambiguity"], params, dynamics_alpha=1.0
    )[0]
    ambiguity_state = jacobian_state_torch(
        "compound_tri_gate", state_by_gate["ambiguity"], params, dynamics_alpha=1.0
    )[0]
    # theta_1 and h are collinear in the same response equation, with h amplified.
    assert float(ambiguity_state[1, 4].detach()) > 7.0 * float(
        ambiguity_param[1, 0].detach()
    )
    assert torch.linalg.svdvals(ambiguity_param)[1] < 0.01


def test_three_gate_diagnostic_fixed_gate_rankings_and_full_rank_main_gate():
    """Every nonzero true parameter remains identifiable at the full-rank M gate."""
    from actdyn.environment.vectorfield import (
        jacobian_param_torch,
        jacobian_state_torch,
    )
    from actdyn.utils.torch_utils import attenuated_state_information

    params = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    state_by_gate = {
        "ambiguity": torch.tensor([[-0.5, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "balanced": torch.tensor([[-0.1, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "main": torch.tensor([[0.3, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    }
    dt = 0.1
    horizon = 40
    eye_state = torch.eye(5)
    eye_parameter = torch.eye(3)
    # Ten paired Poisson neurons at z=0: two neurons per coordinate.
    state_information = torch.diag(torch.tensor([0.09, 0.09, 0.09, 0.09, 0.0025]))
    scores = {}

    for gate, state in state_by_gate.items():
        f_theta = jacobian_param_torch(
            "three_gate_diagnostic", state, params, dynamics_alpha=1.0
        )[0].detach()
        f_state = jacobian_state_torch(
            "three_gate_diagnostic", state, params, dynamics_alpha=1.0
        )[0].detach()
        sensitivity = torch.zeros(5, 3)
        covariance = eye_state.clone()
        information = torch.zeros(3, 3)
        fully_observed_information = torch.zeros(3, 3)
        dynamics_score = 0.0
        state_information_score = 0.0
        for _ in range(horizon):
            transition = eye_state + f_state * dt
            sensitivity = transition @ sensitivity + f_theta * dt
            attenuated = attenuated_state_information(covariance, state_information)
            information += sensitivity.T @ attenuated @ sensitivity
            fully_observed_information += (
                sensitivity.T @ state_information @ sensitivity
            )
            dynamics_score += float(
                torch.trace(sensitivity.T @ covariance @ sensitivity)
            )
            state_information_score += float(
                0.5 * torch.logdet(eye_state + covariance @ state_information)
            )
            covariance = transition @ covariance @ transition.T + 0.01 * eye_state

        scores[gate] = {
            "paldi": float(0.5 * torch.logdet(eye_parameter + information)),
            "fully_observable": float(
                0.5 * torch.logdet(eye_parameter + fully_observed_information)
            ),
            "e_optimality": float(torch.linalg.eigvalsh(information)[0]),
            "dynamics": dynamics_score,
            "state_information": state_information_score,
            "variance": float(torch.trace(f_theta.T @ f_theta)),
        }

    assert scores["main"]["paldi"] > scores["ambiguity"]["paldi"]
    assert scores["main"]["paldi"] > scores["balanced"]["paldi"]
    assert scores["ambiguity"]["fully_observable"] > scores["main"]["fully_observable"]
    assert scores["balanced"]["e_optimality"] > scores["main"]["e_optimality"]
    for objective in ("dynamics", "state_information", "variance"):
        assert scores["ambiguity"][objective] > scores["main"][objective]

    ambiguity_param = jacobian_param_torch(
        "three_gate_diagnostic", state_by_gate["ambiguity"], params, dynamics_alpha=1.0
    )[0]
    ambiguity_state = jacobian_state_torch(
        "three_gate_diagnostic", state_by_gate["ambiguity"], params, dynamics_alpha=1.0
    )[0]
    main_param = jacobian_param_torch(
        "three_gate_diagnostic", state_by_gate["main"], params, dynamics_alpha=1.0
    )[0]
    assert float(ambiguity_state[1, 4].detach()) == pytest.approx(100.0, abs=1e-4)
    nuisance_ratio = float(ambiguity_state[1, 4].detach()) / float(
        ambiguity_param[1, 0].detach()
    )
    assert nuisance_ratio == pytest.approx(5.0, abs=1e-4)
    assert torch.linalg.svdvals(main_param)[2] > 0.73
