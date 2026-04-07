from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyNoise:
    pass


class _DummyDecoder:
    def __init__(self):
        self.noise = _DummyNoise()

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = z.shape
        eye = torch.eye(2, dtype=z.dtype, device=z.device).reshape(1, 1, 2, 2)
        return eye.expand(batch, steps, -1, -1)

    def var(self, z: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = z.shape
        return torch.ones(batch, steps, 2, dtype=z.dtype, device=z.device)


class _DummyDynamics:
    def __init__(self):
        self.logvar = torch.nn.Parameter(torch.log(torch.ones(1, 2) * 1e-6))


class _SamplingDynamics:
    def __init__(self):
        self.logvar = torch.nn.Parameter(torch.log(torch.ones(1, 2) * 1e-6))
        self.theta = torch.zeros(1, 2)

    def set_params(self, theta: torch.Tensor) -> None:
        self.theta = theta.detach().clone()


def _make_model() -> SimpleNamespace:
    return SimpleNamespace(
        e={
            "m": torch.zeros(1, 2),
            "P": torch.diag(torch.tensor([2.0, 3.0])).unsqueeze(0),
        },
        z={
            "P": torch.diag(torch.tensor([4.0, 5.0])).unsqueeze(0),
        },
        decoder=_DummyDecoder(),
        dynamics=_DummyDynamics(),
        dt=1.0,
        device="cpu",
    )


def _fe(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    batch, steps, _, = z.shape
    eye = torch.eye(2, dtype=z.dtype, device=z.device).reshape(1, 1, 2, 2)
    return eye.expand(batch, steps, -1, -1)


def _fz(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    batch, steps, _, = z.shape
    return torch.zeros(batch, steps, 2, 2, dtype=z.dtype, device=z.device)


class _SamplingModel:
    def __init__(self) -> None:
        self.e = {
            "m": torch.tensor([[0.5, -0.25]], dtype=torch.float32),
            "P": torch.diag(torch.tensor([0.2, 0.1], dtype=torch.float32)).unsqueeze(0),
        }
        self.z = {"P": torch.diag(torch.tensor([1.0, 1.0], dtype=torch.float32)).unsqueeze(0)}
        self.decoder = lambda z: z
        self.dynamics = _SamplingDynamics()
        self.dt = 1.0
        self.device = "cpu"
        self._state = torch.tensor([[[1.0, -1.0]]], dtype=torch.float32)
        self.action_encoder = lambda action: action

    def get_state(self) -> torch.Tensor:
        return self._state

    def set_params(self, theta: torch.Tensor) -> None:
        self.e["m"] = theta.detach().clone()
        self.dynamics.set_params(theta)

    def predict(self, action: torch.Tensor) -> torch.Tensor:
        theta = self.dynamics.theta.to(action.device, dtype=action.dtype)
        state = self._state.expand(action.shape[0], -1, -1).to(action.device, dtype=action.dtype)
        preds = []
        for t in range(action.shape[1]):
            state = state + theta.unsqueeze(1) + action[:, t : t + 1, :]
            preds.append(state)
        return torch.cat(preds, dim=1)


def test_parameter_eig_and_fully_observable_match_manual_one_step():
    module = _load_module("cosyne_objectives_v2_param", "experiments/cosyne/objectives.py")
    model = _make_model()
    rollout = {"model_state": torch.zeros(1, 1, 2)}
    metric = module.parameter_eig(model=model, Fe_net=_fe, Fz_net=_fz, gamma=1.0, device="cpu")
    fo_metric = module.fully_observable_parameter_eig(model=model, Fe_net=_fe, Fz_net=_fz, gamma=1.0, device="cpu")
    value = float(metric(rollout).item())
    fo_value = float(fo_metric(rollout).item())
    p_theta = torch.diag(torch.tensor([2.0, 3.0]))
    p_z = torch.diag(torch.tensor([4.0, 5.0]))
    atten = torch.linalg.solve(torch.eye(2) + p_z, torch.eye(2))
    expected = -0.5 * torch.logdet(torch.eye(2) + p_theta @ atten)
    expected_fo = -0.5 * torch.logdet(torch.eye(2) + p_theta)
    assert torch.isclose(torch.tensor(value), expected, atol=1e-5)
    assert torch.isclose(torch.tensor(fo_value), expected_fo, atol=1e-5)
    assert fo_value < value


def test_e_optimality_matches_manual_one_step():
    module = _load_module("cosyne_objectives_v2_eopt", "experiments/cosyne/objectives.py")
    model = _make_model()
    rollout = {"model_state": torch.zeros(1, 1, 2)}
    metric = module.e_optimality(model=model, Fe_net=_fe, Fz_net=_fz, gamma=1.0, device="cpu")
    value = float(metric(rollout).item())
    atten = torch.linalg.solve(torch.eye(2) + torch.diag(torch.tensor([4.0, 5.0])), torch.eye(2))
    expected = -float(torch.min(torch.linalg.eigvalsh(torch.diag(torch.tensor([2.0, 3.0])) @ atten)))
    assert torch.isclose(torch.tensor(value), torch.tensor(expected), atol=1e-5)


def test_state_information_matches_manual_one_step():
    module = _load_module("cosyne_objectives_v2_state", "experiments/cosyne/objectives.py")
    model = _make_model()
    rollout = {"model_state": torch.zeros(1, 1, 2)}
    metric = module.state_information(model=model, Fe_net=_fe, Fz_net=_fz, gamma=1.0, device="cpu")
    value = float(metric(rollout).item())
    expected = -0.5 * torch.logdet(torch.eye(2) + torch.diag(torch.tensor([4.0, 5.0])))
    assert torch.isclose(torch.tensor(value), expected, atol=1e-5)


def test_dynamics_metric_matches_manual_one_step():
    module = _load_module("cosyne_objectives_v2_dyn", "experiments/cosyne/objectives.py")
    model = _make_model()
    rollout = {"model_state": torch.zeros(1, 1, 2)}
    metric = module.dynamics(model=model, Fe_net=_fe, Fz_net=_fz, gamma=1.0, device="cpu")
    value = float(metric(rollout).item())
    expected = -(4.0 + 5.0)
    assert torch.isclose(torch.tensor(value), torch.tensor(expected), atol=1e-5)


def test_sampling_variance_matches_manual_parameter_samples():
    module = _load_module("cosyne_objectives_v2_sampling", "experiments/cosyne/objectives.py")
    model = _SamplingModel()
    rollout = {"action": torch.tensor([[[0.1, -0.2], [0.0, 0.05]]], dtype=torch.float32)}
    metric = module.sampling_variance(
        model=model,
        Fe_net=_fe,
        Fz_net=_fz,
        gamma=0.5,
        device="cpu",
        num_parameter_samples=4,
        sample_seed=7,
    )
    value = float(metric(rollout).item())

    mean = model.e["m"][0]
    cov = model.e["P"][0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    noise = torch.randn(4, 2, generator=generator)
    theta_samples = mean.unsqueeze(0) + noise @ torch.linalg.cholesky(cov + 1e-8 * torch.eye(2))

    lam_samples = []
    for theta in theta_samples:
        state = model.get_state().clone()
        traj = []
        for t in range(rollout["action"].shape[1]):
            state = state + theta.view(1, 1, -1) + rollout["action"][:, t : t + 1, :]
            traj.append(state)
        lam_samples.append(torch.cat(traj, dim=1))
    lam_stack = torch.stack(lam_samples, dim=0)
    var_diag = torch.var(lam_stack, dim=0, unbiased=True)
    logdet_diag = torch.log1p(var_diag).sum(dim=-1)
    expected = -(logdet_diag[:, 0] + 0.5 * logdet_diag[:, 1]).item()
    assert torch.isclose(torch.tensor(value), torch.tensor(expected), atol=1e-5)
