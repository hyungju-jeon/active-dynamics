"""Campaign budget changes must preserve every other resolved policy field."""
from dataclasses import asdict

import pytest

from experiments.tnsre import snn_weak_evidence as w
from experiments.experiment_definitions import get_policy_spec, get_schedule_spec
from experiments.run import _mpc_budget


@pytest.mark.parametrize('policy', w.POLICIES)
def test_budget_changes_only_two_planners(tmp_path, policy):
    scenario = w.SCENARIOS['weak2_e5_x10_ball']
    w.configure_catalogs('ball')
    before = asdict(get_policy_spec(policy))
    schedule = asdict(get_schedule_spec(before['schedule_id']))
    w.configure_identification_budget(tmp_path, scenario, policy, 0, [48, 4, 6])
    after = asdict(get_policy_spec(policy))
    expected = dict(before)
    if policy in ('adaptive_c2h20', 'active_myopic'):
        expected.update(mpc_num_samples=48, mpc_num_iterations=4, mpc_num_elite=6)
        assert _mpc_budget(get_policy_spec(policy)) == (48, 4, 6)
    assert after == expected
    assert asdict(get_schedule_spec(after['schedule_id'])) == schedule
    w.configure_catalogs('ball')


def test_default_budget_is_unchanged(tmp_path):
    w.configure_identification_budget(tmp_path, w.SCENARIOS['weak2_e5_x10_ball'], 'adaptive_c2h20', 0, None)
    assert _mpc_budget(get_policy_spec('adaptive_c2h20')) == (24, 4, 6)


def test_invalid_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match='elites <= samples'):
        w.configure_identification_budget(tmp_path, w.SCENARIOS['weak2_e5_x10_ball'], 'active_myopic', 0, [4, 4, 6])
