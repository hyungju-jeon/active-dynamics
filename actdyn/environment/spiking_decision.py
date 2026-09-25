"""Spiking decision circuit (Wang, 2002) as a closed-loop environment.

The network is the two-pool cortical decision model of Wang (2002), Neuron
36:955. Its published mean-field reduction is the two-variable Wong-Wang
circuit implemented in ``actdyn.utils.vectorfields_eqn.WongWang``; here the
full network of leaky integrate-and-fire neurons is the *environment* that the
learner interacts with, so the reduced model is only an approximation of the
generating dynamics.

Equations, parameters, and population structure follow the Brian2 reference
implementation distributed with Wang's textbook
(``xjwanglab/book``, ``chapter06-decision-making-spiking-network-model``).
Two changes make the circuit controllable online:

* the sensory Poisson inputs of the original protocol are replaced by an
  injected current ``I_stim`` per selective pool, set from the 2-D action at
  every control bin (positive current depolarizes);
* the network is advanced in control bins of ``control_bin_ms`` and reports the
  spike counts of a fixed subset of selective neurons for each bin.

Latent proxy. The pool-averaged NMDA gating variables ``s_1, s_2`` are the
state variables of the Wong-Wang reduction. They are exposed as the latent
state ``z = state_scale (s - 1/2)`` for logging and evaluation only; the
learner never sees them.
"""

from __future__ import annotations

import queue
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from actdyn.environment.session import SessionRule

# Wang (2002) time constant of NMDA gating, the unit of time of the reduced model.
TAU_NMDA_MS = 100.0


def _brian2():
    import brian2 as b2

    return b2


# Neuron and synapse equations from the Brian2 reference implementation. The
# excitatory group carries the additional injected current ``I_stim``.
_EQUATIONS = dict(
    E="""
    dV/dt         = (-(V - V_L) - Isyn/gE) / tau_m_E : volt (unless refractory)
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA - I_stim : amp
    I_AMPA_ext    = gAMPA_ext_E*sAMPA_ext*(V - V_E) : amp
    I_AMPA        = gAMPA_E*S_AMPA*(V - V_E) : amp
    I_NMDA        = gNMDA_E*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    I_GABA        = gGABA_E*S_GABA*(V - V_I) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsAMPA/dt     = -sAMPA/tauAMPA : 1
    dx/dt         = -x/tau_x : 1
    dsNMDA/dt     = -sNMDA/tauNMDA + alpha*x*(1 - sNMDA) : 1
    S_AMPA : 1
    S_NMDA : 1
    S_GABA : 1
    I_stim : amp
    """,
    I="""
    dV/dt         = (-(V - V_L) - Isyn/gI) / tau_m_I : volt (unless refractory)
    Isyn          = I_AMPA_ext + I_AMPA + I_NMDA + I_GABA : amp
    I_AMPA_ext    = gAMPA_ext_I*sAMPA_ext*(V - V_E) : amp
    I_AMPA        = gAMPA_I*S_AMPA*(V - V_E) : amp
    I_NMDA        = gNMDA_I*S_NMDA*(V - V_E)/(1 + exp(-a*V)/b) : amp
    I_GABA        = gGABA_I*S_GABA*(V - V_I) : amp
    dsAMPA_ext/dt = -sAMPA_ext/tauAMPA : 1
    dsGABA/dt     = -sGABA/tauGABA : 1
    S_AMPA : 1
    S_NMDA : 1
    S_GABA : 1
    """,
)


def _model_parameters(b2, *, n_e: int, n_i: int, f_sel: float, w_plus: float, background_rate_hz: float):
    """Wang (2002) parameters with recurrent conductances scaled by population size."""
    params = dict(
        V_L=-70 * b2.mV, Vth=-50 * b2.mV, Vreset=-55 * b2.mV,
        gE=25 * b2.nS, tau_m_E=20 * b2.ms, tau_ref_E=2 * b2.ms,
        gI=20 * b2.nS, tau_m_I=10 * b2.ms, tau_ref_I=1 * b2.ms,
        V_E=0 * b2.mV, V_I=-70 * b2.mV,
        a=0.062 / b2.mV, b=3.57,
        tauAMPA=2 * b2.ms, tau_x=2 * b2.ms, tauNMDA=TAU_NMDA_MS * b2.ms,
        alpha=0.5 * b2.kHz, tauGABA=5 * b2.ms, delay=0.5 * b2.ms,
        gAMPA_ext_E=2.1 * b2.nS, gAMPA_ext_I=1.62 * b2.nS,
        gAMPA_E=80 * b2.nS / n_e, gNMDA_E=264 * b2.nS / n_e, gGABA_E=520 * b2.nS / n_i,
        gAMPA_I=64 * b2.nS / n_e, gNMDA_I=208 * b2.nS / n_e, gGABA_I=400 * b2.nS / n_i,
        nu_ext=float(background_rate_hz) * b2.Hz,
        N_E=int(n_e), N_I=int(n_i),
    )
    n1 = int(round(f_sel * n_e))
    params["N1"] = n1
    params["N2"] = n1
    params["N0"] = int(n_e) - 2 * n1
    if params["N0"] <= 0:
        raise ValueError(f"f_sel={f_sel} leaves no nonselective neurons for n_e={n_e}.")
    w_minus = (1.0 - w_plus * f_sel) / (1.0 - f_sel)
    params["wp"] = float(w_plus)
    params["wm"] = float(w_minus)
    return params


class WangDecisionNetwork:
    """Wang (2002) two-pool spiking decision circuit stepped in control bins.

    Args:
        n_e, n_i: Excitatory and inhibitory population sizes (1600, 400 in Wang).
        f_sel: Fraction of excitatory neurons in each selective pool (0.15).
        w_plus: Hebb-strengthened within-pool weight (1.7).
        sim_dt_ms: Integration step of the network in ms.
        control_bin_ms: Duration of one environment step in ms.
        stim_gain_pa: Injected current per unit action, in pA, applied to every
            neuron of the targeted selective pool. At 20 pA a symmetric drive
            still resolves to one winner and an antisymmetric drive of 0.7 flips
            a committed choice within about 1 s; from 25 pA upward a symmetric
            drive co-activates both pools, a regime the reduced model lacks.
        observed_per_pool: Number of neurons recorded from each selective pool.
        background_rate_hz: Rate of the independent external Poisson drive.
        state_scale: Latent proxy scale, ``z = state_scale (s - 1/2)``.
        seed: Seed of the Brian2 random streams.
        codegen_target: Brian2 code generation target (``"cython"`` or ``"numpy"``).
        max_duration_ms: Length of the single background run (one hour by default);
            the worker stops early when the environment is closed.
    """

    def __init__(
        self,
        *,
        n_e: int = 1600,
        n_i: int = 400,
        f_sel: float = 0.15,
        w_plus: float = 1.7,
        sim_dt_ms: float = 0.1,
        control_bin_ms: float = 5.0,
        stim_gain_pa: float = 20.0,
        observed_per_pool: int = 40,
        background_rate_hz: float = 2400.0,
        state_scale: float = 4.0,
        seed: int = 0,
        codegen_target: str = "cython",
        max_duration_ms: float = 3.6e6,
    ) -> None:
        b2 = _brian2()
        b2.prefs.codegen.target = str(codegen_target)
        b2.defaultclock.dt = float(sim_dt_ms) * b2.ms
        self._b2 = b2
        self.control_bin_ms = float(control_bin_ms)
        self.stim_gain_pa = float(stim_gain_pa)
        self.state_scale = float(state_scale)
        self.observed_per_pool = int(observed_per_pool)
        self.seed = int(seed)
        params = _model_parameters(
            b2, n_e=n_e, n_i=n_i, f_sel=f_sel, w_plus=w_plus, background_rate_hz=background_rate_hz
        )
        self.params = params
        if self.observed_per_pool > params["N1"]:
            raise ValueError(
                f"observed_per_pool={observed_per_pool} exceeds pool size {params['N1']}."
            )
        # Synaptic weights between excitatory subpopulations (nonselective, pool 1, pool 2).
        self.W = np.asarray(
            [[1.0, 1.0, 1.0], [params["wm"], params["wp"], params["wm"]], [params["wm"], params["wm"], params["wp"]]]
        )

        net: "OrderedDict[str, Any]" = OrderedDict()
        for label in ("E", "I"):
            net[label] = b2.NeuronGroup(
                params["N_" + label],
                _EQUATIONS[label],
                method="rk2",
                threshold="V > Vth",
                reset="V = Vreset",
                refractory=params["tau_ref_" + label],
                namespace=params,
                name=f"group_{label}",
            )
        n0, n1 = params["N0"], params["N1"]
        self.pool_slices = (slice(0, n0), slice(n0, n0 + n1), slice(n0 + n1, n0 + 2 * n1))
        exc = OrderedDict((k, net["E"][s]) for k, s in enumerate(self.pool_slices))

        for label in ("E", "I"):
            net["pg" + label] = b2.PoissonGroup(params["N_" + label], params["nu_ext"])
            net["ic" + label] = b2.Synapses(net["pg" + label], net[label], on_pre="sAMPA_ext += 1", delay=params["delay"])
            net["ic" + label].connect(condition="i == j")
        net["icAMPA"] = b2.Synapses(net["E"], net["E"], on_pre="sAMPA += 1", delay=params["delay"])
        net["icAMPA"].connect(condition="i == j")
        net["icNMDA"] = b2.Synapses(net["E"], net["E"], on_pre="x += 1", delay=params["delay"])
        net["icNMDA"].connect(condition="i == j")
        net["icGABA"] = b2.Synapses(net["I"], net["I"], on_pre="sGABA += 1", delay=params["delay"])
        net["icGABA"].connect(condition="i == j")

        # Recurrent coupling through population sums, as compiled summed synapses.
        # Wang's implementation recomputes the pool sums in a Python callback at
        # every integration step; the three-unit "pool" group below holds the same
        # sums so the coupling stays inside Brian2's generated code.
        pools = b2.NeuronGroup(3, "sum_ampa : 1\nsum_nmda : 1", name="pool_sums")
        gaba_sum = b2.NeuronGroup(1, "sum_gaba : 1", name="gaba_sum")
        pool_of_neuron = np.zeros(params["N_E"], dtype=int)
        pool_of_neuron[self.pool_slices[1]] = 1
        pool_of_neuron[self.pool_slices[2]] = 2
        collect = b2.Synapses(
            net["E"], pools,
            "sum_ampa_post = sAMPA_pre : 1 (summed)\nsum_nmda_post = sNMDA_pre : 1 (summed)",
            name="collect_pool_sums",
        )
        collect.connect(i=np.arange(params["N_E"]), j=pool_of_neuron)
        broadcast_e = b2.Synapses(
            pools, net["E"],
            "w : 1\nS_AMPA_post = w * sum_ampa_pre : 1 (summed)\nS_NMDA_post = w * sum_nmda_pre : 1 (summed)",
            name="broadcast_pool_sums_E",
        )
        broadcast_e.connect(True)
        # W[post_pool, pre_pool] weights every (pre pool -> post neuron) pair.
        broadcast_e.w = self.W[pool_of_neuron[broadcast_e.j[:]], broadcast_e.i[:]]
        broadcast_i = b2.Synapses(
            pools, net["I"],
            "S_AMPA_post = sum_ampa_pre : 1 (summed)\nS_NMDA_post = sum_nmda_pre : 1 (summed)",
            name="broadcast_pool_sums_I",
        )
        broadcast_i.connect(True)
        collect_gaba = b2.Synapses(net["I"], gaba_sum, "sum_gaba_post = sGABA_pre : 1 (summed)", name="collect_gaba")
        collect_gaba.connect(True)
        gaba_to_e = b2.Synapses(gaba_sum, net["E"], "S_GABA_post = sum_gaba_pre : 1 (summed)", name="gaba_to_E")
        gaba_to_e.connect(True)
        gaba_to_i = b2.Synapses(gaba_sum, net["I"], "S_GABA_post = sum_gaba_pre : 1 (summed)", name="gaba_to_I")
        gaba_to_i.connect(True)
        coupling = [pools, gaba_sum, collect, broadcast_e, broadcast_i, collect_gaba, gaba_to_e, gaba_to_i]
        self._pools = pools
        self._gaba_sum = gaba_sum

        self.spike_monitor = b2.SpikeMonitor(net["E"], record=False, name="spikes_E")
        self.net = net
        self.exc = exc
        # Observed neurons: the first ``observed_per_pool`` neurons of each selective pool.
        self.observed_indices = np.concatenate(
            [np.arange(s.start, s.start + self.observed_per_pool) for s in self.pool_slices[1:]]
        )
        self._prev_count = np.zeros(params["N_E"], dtype=np.int64)
        self._stim = np.zeros(params["N_E"], dtype=np.float64)

        # Closed-loop bridge. Brian2's ``Network.run`` costs ~0.3 s of bookkeeping per
        # call, so instead of one run per control bin the whole session is a single
        # run driven from a worker thread; ``exchange`` fires once per control bin
        # and trades the bin's spike counts for the next action through two queues.
        self._to_net: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self._to_env: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._worker_error: Optional[BaseException] = None
        self._started = False
        network_ref = self

        @b2.network_operation(dt=self.control_bin_ms * b2.ms, when="end")
        def exchange():
            network_ref._exchange()

        self.network = b2.Network(list(net.values()) + coupling + [self.spike_monitor, exchange])
        self.max_duration_ms = float(max_duration_ms)

    # ------------------------------------------------------------ worker side
    def _apply_command(self, command: Any) -> bool:
        """Apply an env command inside the worker thread. Returns False to stop."""
        b2 = self._b2
        if command is None:
            self.network.stop()
            return False
        kind, payload = command
        if kind == "reset":
            self._reinit(int(payload))
            self._stim[:] = 0.0
        elif kind == "act":
            a = np.asarray(payload, dtype=np.float64).reshape(-1)
            self._stim[:] = 0.0
            self._stim[self.pool_slices[1]] = self.stim_gain_pa * a[0]
            self._stim[self.pool_slices[2]] = self.stim_gain_pa * a[1]
        else:
            raise ValueError(f"Unknown command {kind!r}")
        self.net["E"].I_stim = self._stim * b2.pA
        return True

    def _exchange(self) -> None:
        count = np.array(self.spike_monitor.count[:], dtype=np.int64)
        counts = (count - self._prev_count)[self.observed_indices].astype(np.float32)
        self._prev_count = count
        self._to_env.put((counts, self.latent_proxy()))
        while True:
            command = self._to_net.get()
            keep_going = self._apply_command(command)
            if not keep_going:
                return
            if command[0] == "act":
                return
            # A reset is acknowledged immediately so the env sees the fresh state.
            self._prev_count = np.array(self.spike_monitor.count[:], dtype=np.int64)
            self._to_env.put((np.zeros(self.n_observed, dtype=np.float32), self.latent_proxy()))

    def _reinit(self, seed: int) -> None:
        b2 = self._b2
        b2.seed(int(seed))
        rng = np.random.default_rng(int(seed))
        p = self.params
        for label in ("E", "I"):
            group = self.net[label]
            v = rng.uniform(float(p["Vreset"] / b2.volt), float(p["Vth"] / b2.volt), size=p["N_" + label])
            group.V = v * b2.volt
        for name in ("sAMPA_ext", "sAMPA", "x", "sNMDA"):
            setattr(self.net["E"], name, 0)
        for name in ("sAMPA_ext", "sGABA"):
            setattr(self.net["I"], name, 0)
        # Summed coupling variables are refreshed after the next state update, so
        # they must be set to the sums of the reset state (all gating is zero) or
        # the first step integrates with the recurrent input of the previous trial.
        for label in ("E", "I"):
            for name in ("S_AMPA", "S_NMDA", "S_GABA"):
                setattr(self.net[label], name, 0)
        self._pools.sum_ampa = 0
        self._pools.sum_nmda = 0
        self._gaba_sum.sum_gaba = 0
        # Clear refractory state so a reset does not inherit pending refractory periods.
        for label in ("E", "I"):
            self.net[label].not_refractory = True
            self.net[label].lastspike = -1e9 * b2.second
        # Drop spikes still in flight in the 0.5 ms synaptic delay queues. Each queue
        # keeps its slot count (restoring fewer slots would break ``push``), so after
        # a reset with the same seed the network repeats its trajectory exactly.
        for obj in self.net.values():
            for pathway in getattr(obj, "_pathways", ()):
                queue_state = pathway.queue._full_state()
                pathway.queue._restore_from_full_state((0, [[] for _ in queue_state[1]]))

    def _worker(self) -> None:
        b2 = self._b2
        try:
            self.network.run(self.max_duration_ms * b2.ms)
        except BaseException as exc:  # surfaced to the env thread on its next call
            self._worker_error = exc
            try:
                self._to_env.put(exc)
            except Exception:
                pass

    # --------------------------------------------------------------- env side
    def _receive(self) -> Tuple[np.ndarray, np.ndarray]:
        item = self._to_env.get()
        if isinstance(item, BaseException):
            raise RuntimeError("Spiking network worker failed") from item
        return item

    def start(self) -> None:
        if self._started:
            return
        self._thread = threading.Thread(target=self._worker, name="wang2002-brian2", daemon=True)
        self._thread.start()
        self._started = True
        # The first exchange (t = 0) reports an empty bin; consume it.
        self._receive()

    def close(self) -> None:
        if self._started and self._thread is not None and self._thread.is_alive():
            try:
                self._to_net.put(None, timeout=5.0)
            except queue.Full:
                pass
            self._thread.join(timeout=30.0)
        self._started = False

    def __del__(self) -> None:  # best effort; the worker thread is a daemon anyway
        try:
            self.close()
        except Exception:
            pass

    @property
    def n_observed(self) -> int:
        return int(self.observed_indices.shape[0])

    @property
    def dt_latent(self) -> float:
        """Control bin in units of the reduced model (tau_NMDA = 1)."""
        return self.control_bin_ms / TAU_NMDA_MS

    def gating(self) -> np.ndarray:
        """Pool-averaged NMDA gating ``(s_1, s_2)`` of the two selective pools."""
        return np.asarray(
            [float(np.mean(self.exc[1].sNMDA[:])), float(np.mean(self.exc[2].sNMDA[:]))],
            dtype=np.float64,
        )

    def latent_proxy(self) -> np.ndarray:
        """Latent proxy ``z = state_scale (s - 1/2)`` with shape (2,)."""
        return (self.state_scale * (self.gating() - 0.5)).astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Re-initialize membrane potentials and synapses; return the latent proxy."""
        if seed is not None:
            self.seed = int(seed)
        self.start()
        self._to_net.put(("reset", self.seed))
        _counts, z = self._receive()
        return z

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Advance one control bin with pool currents ``stim_gain_pa * action``.

        Args:
            action: Shape (2,), dimensionless drive to pool 1 and pool 2.

        Returns:
            ``(counts, z)``: spike counts of the observed neurons in this bin with
            shape (n_observed,), and the latent proxy after the bin with shape (2,).
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f"action must have 2 entries, got shape {a.shape}.")
        self.start()
        self._to_net.put(("act", a))
        return self._receive()


class SpikingDecisionEnv(gym.Env):
    """Gym environment around :class:`WangDecisionNetwork`.

    Observations returned by ``step`` are the latent proxy (so the shared
    wrapper logs it as ``latent_state``); the spike counts of the bin are kept
    in ``last_counts`` and in ``info["spike_counts"]`` for the paired
    :class:`SpikeCountObservation`.

    With a ``session_rule`` the environment runs decision sessions: it reports
    the decision (``info["decision"]``: 1 or 2 once made, else 0) and the session
    clock (``info["session_context"]``), and after the post-decision delay or the
    session time limit it resets the network itself. The step that ends a
    session returns the spike counts of its last bin, ``info["session_end"] =
    True``, the post-reset latent as ``latent_state``, and the latent before the
    reset as ``session_pre_reset_state``. Session ``k`` of a run
    reseeded with ``seed`` uses network seed ``session_seed(seed, k)``. After
    ``rule.max_sessions`` sessions the step reports ``terminated``.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        network: WangDecisionNetwork,
        *,
        action_max: float = 1.0,
        reference_params: Optional[np.ndarray] = None,
        session_rule: Optional[SessionRule] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.network = network
        self.dt = float(network.dt_latent)
        self.device = torch.device(device)
        self.action_space = spaces.Box(-float(action_max), float(action_max), shape=(2,), dtype=np.float32)
        lim = float(network.state_scale)
        self.observation_space = spaces.Box(-lim, lim, shape=(2,), dtype=np.float32)
        self.reference_params = (
            None if reference_params is None else np.asarray(reference_params, dtype=np.float32)
        )
        self.session_rule = session_rule
        self.last_counts = np.zeros(network.n_observed, dtype=np.float32)
        self.state = torch.as_tensor(network.latent_proxy(), device=self.device)
        self._run_seed = int(network.seed)
        self._reset_session_counters(0)
        self.session_log: list[dict[str, int]] = []

    @staticmethod
    def session_seed(run_seed: int, session_index: int) -> int:
        """Network seed of session ``session_index`` in a run seeded with ``run_seed``."""
        return 1_000_000 + 10_000 * int(run_seed) + int(session_index)

    def _reset_session_counters(self, session_index: int) -> None:
        self.session_index = int(session_index)
        self.session_bins = 0
        self.decision = 0
        self.decision_bin = -1

    def get_params(self) -> torch.Tensor:
        """Reference parameters of the reduced model (not parameters of the network)."""
        if self.reference_params is None:
            return torch.zeros(0, device=self.device)
        return torch.as_tensor(self.reference_params, device=self.device)

    def set_params(self, *args: Any, **kwargs: Any) -> None:  # The network has no reduced parameters.
        return None

    def session_context(self) -> Dict[str, Any]:
        """Session clock as the agents see it (bins elapsed, decision made, bins since)."""
        decided = self.decision != 0
        return {"bins": self.session_bins, "decided": decided,
                "since": self.session_bins - self.decision_bin if decided else 0}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._run_seed = int(seed)
        self._reset_session_counters(0)
        self.session_log = []
        network_seed = (self.session_seed(self._run_seed, 0) if self.session_rule is not None
                        else seed)
        z = self.network.reset(network_seed)
        self.last_counts = np.zeros(self.network.n_observed, dtype=np.float32)
        self.state = torch.as_tensor(z, device=self.device)
        info = {"latent_state": z, "spike_counts": self.last_counts}
        if self.session_rule is not None:
            info.update(session_index=0, session_end=False, decision=0, session_context=self.session_context())
        return z, info

    def step(self, action: Any):
        a = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        counts, z = self.network.step(a.reshape(-1))
        self.last_counts = counts
        info: Dict[str, Any] = {"latent_state": z, "spike_counts": counts}
        terminated = False
        rule = self.session_rule
        if rule is not None:
            self.session_bins += 1
            gap = float(z[0] - z[1])
            if self.decision == 0 and abs(gap) > float(rule.decision_gap):
                self.decision = 1 if gap > 0 else 2
                self.decision_bin = self.session_bins
            decided = self.decision != 0
            end = (decided and self.session_bins - self.decision_bin >= int(rule.post_decision_bins)) or (
                self.session_bins >= int(rule.max_session_bins))
            info.update(session_index=self.session_index, decision=self.decision, session_end=bool(end))
            if end:
                self.session_log.append({"session": self.session_index, "bins": self.session_bins,
                                         "decision": self.decision, "decision_bin": self.decision_bin})
                next_index = self.session_index + 1
                info["session_pre_reset_state"] = z
                z = self.network.reset(self.session_seed(self._run_seed, next_index))
                self._reset_session_counters(next_index)
                info["latent_state"] = z
                info["session_reset_state"] = np.asarray(rule.reset_state, dtype=np.float32)
                info["session_reset_variance"] = float(rule.reset_variance)
                terminated = rule.max_sessions is not None and next_index >= int(rule.max_sessions)
            info["session_context"] = self.session_context()
        self.state = torch.as_tensor(z, device=self.device)
        return z, 0.0, bool(terminated), False, info


class SpikeCountObservation(torch.nn.Module):
    """Observation model that returns the network's spike counts of the current bin.

    ``network`` holds the calibrated log-linear readout ``lambda(z) = exp(C z + b)``
    scaled by ``dt`` so the estimator's decoder can copy it with
    ``Decoder.set_params``; ``observe`` ignores its argument and returns the
    counts produced by the spiking network for the bin that was just simulated.
    """

    def __init__(self, env: SpikingDecisionEnv, *, weight: torch.Tensor, bias: torch.Tensor, dt: float, device: str = "cpu"):
        super().__init__()
        from actdyn.environment.observation import Exp, Scale

        self.env = env
        self.device = torch.device(device)
        self.d_latent = int(weight.shape[1])
        self.d_obs = int(weight.shape[0])
        self.noise_type = "poisson"
        self.R = 0.0
        self.dt = float(dt)
        linear = torch.nn.Linear(self.d_latent, self.d_obs)
        linear.weight = torch.nn.Parameter(weight.to(self.device, torch.float32).clone())
        linear.bias = torch.nn.Parameter(bias.to(self.device, torch.float32).clone())
        self.network = torch.nn.Sequential(linear, Exp(), Scale(self.dt)).to(self.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # Expected counts of the calibrated readout.
        return self.network(z)

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        counts = torch.as_tensor(self.env.last_counts, dtype=torch.float32, device=self.device)
        return counts.reshape(1, 1, -1)


def fit_loglinear_readout(
    latents: np.ndarray,
    counts: np.ndarray,
    *,
    dt: float,
    iterations: int = 500,
    learning_rate: float = 0.05,
    ridge: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit ``counts ~ Poisson(dt exp(C z + b))`` by penalized maximum likelihood.

    Args:
        latents: Shape (T, d_z) latent proxies.
        counts: Shape (T, n) spike counts per bin.
        dt: Bin length in latent time units.
        iterations, learning_rate: Adam settings; the fit is deterministic
            (zero initialization, full-batch gradients).
        ridge: L2 penalty on ``C`` that keeps unresponsive neurons finite.

    Returns:
        ``(C, b)`` with shapes (n, d_z) and (n,) as float32 tensors.
    """
    z = torch.as_tensor(np.asarray(latents), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(counts), dtype=torch.float64)
    if z.ndim != 2 or y.ndim != 2 or z.shape[0] != y.shape[0]:
        raise ValueError(f"latents {tuple(z.shape)} and counts {tuple(y.shape)} must share T.")
    n, d = int(y.shape[1]), int(z.shape[1])
    c = torch.zeros((n, d), dtype=torch.float64, requires_grad=True)
    mean_rate = torch.clamp(y.mean(dim=0), min=1e-3) / float(dt)
    b = torch.log(mean_rate).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([c, b], lr=float(learning_rate))
    log_dt = float(np.log(dt))
    for _ in range(int(iterations)):
        optimizer.zero_grad()
        log_lam = z @ c.T + b + log_dt
        nll = torch.sum(torch.exp(log_lam) - y * log_lam) / z.shape[0]
        loss = nll + float(ridge) * torch.sum(c * c)
        loss.backward()
        optimizer.step()
    return c.detach().to(torch.float32), b.detach().to(torch.float32)


def calibration_input_program(
    steps: int, *, hold_steps: int, seed: int, amplitude: float = 1.0
) -> np.ndarray:
    """Piecewise-constant pool drives that visit both choices, shape (steps, 2).

    Each pool independently holds a level from ``{-1, 0, +1} * amplitude`` for
    ``hold_steps`` bins; levels are drawn with ``numpy.random.default_rng(seed)``.
    """
    rng = np.random.default_rng(int(seed))
    n_blocks = int(np.ceil(steps / hold_steps))
    levels = rng.choice([-1.0, 0.0, 1.0], size=(n_blocks, 2)) * float(amplitude)
    return np.repeat(levels, hold_steps, axis=0)[:steps].astype(np.float32)


def calibrate_spike_readout(
    network: WangDecisionNetwork,
    *,
    steps: int,
    hold_steps: int,
    seed: int,
    action_max: float = 1.0,
    iterations: int = 500,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Run an open-loop calibration episode and fit the log-linear readout.

    The network is reset with ``seed``, driven by :func:`calibration_input_program`,
    and reset again afterwards so the experiment starts from a fresh state.
    Returns ``(C, b, diagnostics)`` where diagnostics report mean rate and the
    latent proxy range visited.
    """
    program = calibration_input_program(steps, hold_steps=hold_steps, seed=seed, amplitude=action_max)
    network.reset(seed)
    latents = np.empty((steps, 2), dtype=np.float64)
    counts = np.empty((steps, network.n_observed), dtype=np.float64)
    for t in range(steps):
        y, z = network.step(program[t])
        counts[t] = y
        latents[t] = z
    c, b = fit_loglinear_readout(latents, counts, dt=network.dt_latent, iterations=iterations)
    diagnostics = {
        "calibration_mean_rate_hz": float(counts.mean() / (network.control_bin_ms / 1000.0)),
        "calibration_z_min": float(latents.min()),
        "calibration_z_max": float(latents.max()),
        "calibration_steps": float(steps),
    }
    network.reset(seed)
    return c, b, diagnostics


def build_spiking_environment(
    env_preset: Any,
    *,
    seed: int,
    action_max: float,
    device: str = "cpu",
) -> Tuple[SpikingDecisionEnv, SpikeCountObservation, Dict[str, float]]:
    """Build the Wang (2002) environment and its calibrated spike-count observation.

    The preset supplies the network configuration (``spiking_*`` fields), the
    latent step ``dt`` in units of tau_NMDA (control bin ``dt * 100 ms``), and the
    reduced-model reference parameters. The readout is calibrated on an
    open-loop episode seeded by ``seed`` before the experiment starts.
    """
    dt = float(env_preset.dt)
    network = WangDecisionNetwork(
        n_e=int(env_preset.spiking_n_e),
        n_i=int(env_preset.spiking_n_i),
        f_sel=float(env_preset.spiking_f_sel),
        w_plus=float(env_preset.spiking_w_plus),
        sim_dt_ms=float(env_preset.spiking_sim_dt_ms),
        control_bin_ms=dt * TAU_NMDA_MS,
        stim_gain_pa=float(env_preset.spiking_stim_gain_pa),
        observed_per_pool=int(env_preset.spiking_observed_per_pool),
        background_rate_hz=float(env_preset.spiking_background_rate_hz),
        seed=int(seed),
        codegen_target=str(env_preset.spiking_codegen_target),
    )
    if int(env_preset.observation_dim) != network.n_observed:
        raise ValueError(
            f"observation_dim={env_preset.observation_dim} must equal 2 * spiking_observed_per_pool "
            f"= {network.n_observed} for {env_preset.preset_id}."
        )
    weight, bias, diagnostics = calibrate_spike_readout(
        network,
        steps=int(env_preset.spiking_calibration_steps),
        hold_steps=int(env_preset.spiking_calibration_hold_steps),
        seed=int(seed),
        action_max=float(action_max),
    )
    reference = np.asarray(env_preset.resolved_true_params(), dtype=np.float32)
    env = SpikingDecisionEnv(network, action_max=action_max, reference_params=reference,
                             session_rule=session_rule_from_preset(env_preset, network.state_scale),
                             device=device)
    obs_model = SpikeCountObservation(env, weight=weight, bias=bias, dt=dt, device=device)
    return env, obs_model, diagnostics


def session_rule_from_preset(env_preset: Any, state_scale: float) -> Optional[SessionRule]:
    """Session rule of a preset, or None when the preset runs one continuous trial.

    The reset state is the latent of all-zero gating, ``z = -state_scale / 2``.
    """
    n_sessions = getattr(env_preset, "spiking_sessions", None)
    if n_sessions is None:
        return None
    return SessionRule(
        decision_gap=float(env_preset.spiking_decision_gap),
        post_decision_bins=int(env_preset.spiking_post_decision_bins),
        max_session_bins=int(env_preset.spiking_max_session_bins),
        reset_state=(-float(state_scale) / 2.0,) * 2,
        reset_variance=float(env_preset.spiking_reset_variance),
        max_sessions=int(n_sessions),
    )

