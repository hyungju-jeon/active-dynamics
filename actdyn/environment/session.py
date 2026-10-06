"""Decision sessions: decide, wait, reset.

A session starts from the quiet initial point of the decision circuit. The
choice is made when one pool leads the other by more than ``decision_gap`` in
the latent ``z = 4 (s - 1/2)``; ``post_decision_bins`` later the circuit is reset
to the initial point, and a session without a decision ends after
``max_session_bins``. The environment applies the rule to the true latent. Agents
learn of a reset only when it happens; their filters then jump to the known reset
state. Planners do not simulate resets: a model rollout stays in the current session.

A rule with ``evidence_amplitude > 0`` adds sensory evidence to every session from
``evidence_first_session`` up to (not including) ``evidence_last_session``: the input
``evidence_amplitude`` drives one pool, drawn per session, during the first
``evidence_bins`` bins, on top of the agent's input. Other sessions have no evidence
(pool ``NO_EVIDENCE``). Agents know the schedule and the pool of the current session.
From ``fixed_first_session`` on, a decision no longer ends a session: it runs the full
``max_session_bins`` (fixed-length sessions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SessionRule:
    """Session protocol shared by the environment and the agents.

    Attributes:
        decision_gap: A decision is made when ``|z_1 - z_2|`` exceeds this value.
        post_decision_bins: Bins between the decision and the reset.
        max_session_bins: A session without a decision ends after this many bins.
        reset_state: Latent state after a reset, shape (2,) (all gating zero).
        reset_variance: Variance of the agents' state belief right after a reset.
        max_sessions: The environment terminates after this many sessions (None: never).
        evidence_amplitude: Evidence input to the evidence pool, in action units (0: none).
        evidence_bins: Bins from the session start during which the evidence is on.
        evidence_first_session: Sessions with a lower index have no evidence.
        evidence_last_session: Sessions with this or a higher index have no evidence (None: no limit).
        fixed_first_session: Sessions with this or a higher index end only at ``max_session_bins``
            (None: never).
    """

    decision_gap: float = 2.0
    post_decision_bins: int = 40
    max_session_bins: int = 400
    reset_state: tuple[float, float] = (-2.0, -2.0)
    reset_variance: float = 0.01
    max_sessions: int | None = None
    evidence_amplitude: float = 0.0
    evidence_bins: int = 0
    evidence_first_session: int = 0
    evidence_last_session: int | None = None
    fixed_first_session: int | None = None


NO_EVIDENCE = -2  # evidence pool of a session without evidence (-1: pool not known yet)


def session_has_evidence(rule: SessionRule, session_index: int) -> bool:
    """Whether session ``session_index`` (0-based) receives evidence under ``rule``."""
    k = int(session_index)
    last = rule.evidence_last_session
    return float(rule.evidence_amplitude) > 0.0 and k >= int(rule.evidence_first_session) and (last is None or k < int(last))


def session_post_decision_bins(rule: SessionRule, session_index: int) -> int:
    """Bins from the decision to the end of session ``session_index``; a fixed-length session never ends early."""
    first = rule.fixed_first_session
    if first is not None and int(session_index) >= int(first):
        return int(rule.max_session_bins)
    return int(rule.post_decision_bins)


def session_evidence_pool(session_seed: int) -> int:
    """Pool (0 or 1) that receives the evidence in the session with network seed ``session_seed``."""
    return int(np.random.default_rng([int(session_seed), 1]).integers(2))


def session_evidence(
    bins: torch.Tensor, pool: torch.Tensor, rule: SessionRule, step_bins: int = 1
) -> torch.Tensor:
    """Evidence input of the step that starts at session bin ``bins``, shape (B, 2), float32.

    Args:
        bins: Bins elapsed in the session at the start of the step, shape (B,).
        pool: Evidence pool per step, shape (B,): 0 or 1; -1 for a pool that is not
            known, which gets the expected evidence, half the amplitude to each pool; or
            ``NO_EVIDENCE`` for a session without evidence (zero).
        step_bins: Bins per step; a step that straddles the end of the evidence period
            gets the evidence averaged over its bins.
    """
    step = float(step_bins)
    on = ((float(rule.evidence_bins) - bins.to(torch.float32)) / step).clamp(0.0, 1.0)
    onehot = torch.nn.functional.one_hot(pool.clamp(min=0).long(), 2).to(torch.float32)
    split = torch.full_like(onehot, 0.5)
    share = torch.where((pool >= 0).unsqueeze(-1), onehot, split)
    share = torch.where((pool == NO_EVIDENCE).unsqueeze(-1), torch.zeros_like(share), share)
    return float(rule.evidence_amplitude) * on.unsqueeze(-1) * share
