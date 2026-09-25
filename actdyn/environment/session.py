"""Decision sessions: decide, wait, reset.

A session starts from the quiet initial point of the decision circuit. The
choice is made when one pool leads the other by more than ``decision_gap`` in
the latent ``z = 4 (s - 1/2)``; ``post_decision_bins`` later the circuit is reset
to the initial point, and a session without a decision ends after
``max_session_bins``. The environment applies the rule to the true latent; agents
are given the same rule, so their filters know the reset state and their
planners can simulate resets inside model rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """

    decision_gap: float = 2.0
    post_decision_bins: int = 40
    max_session_bins: int = 400
    reset_state: tuple[float, float] = (-2.0, -2.0)
    reset_variance: float = 0.01
    max_sessions: int | None = None


def new_session_clock(batch: int, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Clock of ``batch`` sessions that have just started."""
    return {
        "bins": torch.zeros(batch, dtype=torch.long, device=device),
        "decided": torch.zeros(batch, dtype=torch.bool, device=device),
        "since": torch.zeros(batch, dtype=torch.long, device=device),
    }


def session_clock_from_context(
    context: dict[str, int | bool] | None, batch: int, *, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """Clock for ``batch`` rollouts starting from the environment's current session state.

    ``context`` holds ``bins`` (bins elapsed in the session), ``decided``, and
    ``since`` (bins since the decision), as reported by the environment.
    """
    clock = new_session_clock(batch, device=device)
    if context:
        clock["bins"].fill_(int(context.get("bins", 0)))
        clock["decided"].fill_(bool(context.get("decided", False)))
        clock["since"].fill_(int(context.get("since", 0)))
    return clock


def advance_session_clock(
    clock: dict[str, torch.Tensor], gap: torch.Tensor, rule: SessionRule, step_bins: int = 1
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Advance session clocks by one step of ``step_bins`` bins.

    Args:
        clock: Per-rollout ``bins``, ``decided``, ``since``, each shape (B,).
        gap: ``z_1 - z_2`` after the step, shape (B,).

    Returns:
        ``(clock, decision_now, reset_now)``: the updated clock (zeroed where a
        reset happened), whether the decision was made in this step, and whether
        the state resets after this step, each shape (B,).
    """
    step = int(step_bins)
    bins = clock["bins"] + step
    decision_now = (~clock["decided"]) & (gap.abs() > float(rule.decision_gap))
    decided = clock["decided"] | decision_now
    since = torch.where(clock["decided"], clock["since"] + step, torch.zeros_like(clock["since"]))
    reset_now = (decided & (since >= int(rule.post_decision_bins))) | (bins >= int(rule.max_session_bins))
    zero = torch.zeros_like(bins)
    new_clock = {
        "bins": torch.where(reset_now, zero, bins),
        "decided": decided & ~reset_now,
        "since": torch.where(reset_now, zero, since),
    }
    return new_clock, decision_now, reset_now
