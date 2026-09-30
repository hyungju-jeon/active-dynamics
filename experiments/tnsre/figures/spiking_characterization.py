"""Fitted reduced-model phase portrait and detailed network schematic.

The vector field is the saved reduced fit at zero input, not the true SNN drift.
Decision thresholds are task stopping rules, not basin separatrices.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from scipy.special import expit
from .assets import _ASSET_LABEL_SIZE, _ASSET_TICK_SIZE, _apply_asset_style, save_figure
from .theme import STROKE_COLOR, style_experiment_axis
from actdyn.utils.figure_io import load_plotting


def reduced_drift(z: np.ndarray, params: dict) -> np.ndarray:
    """Zero-input drift dz/d(t/100 ms), state shape (..., 2), float64."""
    z = np.asarray(z, dtype=float)
    s = z / 4 + 0.5
    a = (params['w_plus'] * z - params['w_minus'] * z[..., ::-1]) / 4 + params['h']
    return 4 * (-s + params['gamma'] * (1 - s) * expit(6 * a))


def fixed_points(params: dict) -> list[dict]:
    """Find distinct equilibria from a deterministic 15x15 grid; classify by Jacobian."""
    points = []
    for x in np.linspace(-2, 2, 15):
        for y in np.linspace(-2, 2, 15):
            r = root(lambda z: reduced_drift(z, params), [x, y])
            if (not r.success or np.linalg.norm(reduced_drift(r.x, params)) > 1e-8
                    or np.max(np.abs(r.x)) > 2 or any(np.linalg.norm(r.x-q) < 1e-5 for q in points)):
                continue
            points.append(r.x)
    records = []
    for q in sorted(points, key=lambda z: z[0]):
        eps = np.eye(2) * 1e-5
        jac = np.column_stack([(reduced_drift(q+d, params)-reduced_drift(q-d, params))/2e-5 for d in eps])
        eig = np.linalg.eigvals(jac)
        stable = bool(np.max(eig.real) < 0)
        records.append(dict(state=q.tolist(), eigenvalues=eig.real.tolist(), stable=stable,
                            residual=float(np.linalg.norm(reduced_drift(q, params)))))
    return records


def draw_phase_portrait(ax, experiment_dir: Path) -> list[dict]:
    params = json.loads((experiment_dir/'reference/m2_fit.json').read_text())['physical_parameters']
    xx, yy = np.meshgrid(np.linspace(-2, 2, 61), np.linspace(-2, 2, 61))
    v = reduced_drift(np.stack([xx, yy], axis=-1), params)
    ax.streamplot(xx, yy, v[..., 0], v[..., 1], density=0.7, color='#A6A6A6',
                  linewidth=0.45, arrowsize=0.55)
    ax.plot([-2, 0], [0, 2], '--', color=STROKE_COLOR, lw=1.3, zorder=4)
    ax.plot([0, 2], [-2, 0], '--', color=STROKE_COLOR, lw=1.3, zorder=4)
    records = fixed_points(params)
    ax.plot(-2,-2,marker='x',ms=4,color=STROKE_COLOR,mew=.8,zorder=7)
    for r in records:
        difference = r['state'][0] - r['state'][1]
        color = '#3E6FB0' if difference > 1e-5 else '#C84747' if difference < -1e-5 else STROKE_COLOR
        ax.plot(*r['state'], 'o' if r['stable'] else 'D', ms=5 if r['stable'] else 4,
                mfc=color if r['stable'] else 'white', mec=color, mew=0.7, zorder=6)
    ax.set(xlim=(-2.12, 2.12), ylim=(-2.12, 2.12), xlabel=r'$z_1$', ylabel=r'$z_2$')
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2]); ax.set_aspect('equal')
    ax.set_title('Fitted model', fontsize=_ASSET_LABEL_SIZE, pad=19)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color=STROKE_COLOR,ms=3,label='stable'),
                       Line2D([],[],marker='D',ls='',mfc='white',color=STROKE_COLOR,ms=3,label='saddle'),
                       Line2D([],[],ls='--',color=STROKE_COLOR,lw=1.3,label='threshold'),
                       Line2D([],[],marker='x',ls='',color=STROKE_COLOR,ms=3,label='reset')],
              loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=_ASSET_TICK_SIZE,
              frameon=False,ncol=4,columnspacing=.45,handlelength=.9,handletextpad=.3)
    style_experiment_axis(ax)
    return records


def generate_circuit(experiment_dir: Path, output: Path) -> Path:
    """Population schematic in the style of Wilson-Cowan Fig. 2A.

    Counts, weights, currents, and recording selection follow WangDecisionNetwork
    and the saved experiment preset. The spike glyph is schematic, not data.
    Paired edges are straight and parallel, so every tip and bar ends on its
    target rim; the layout has no crossing edges.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path as MPath
    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    width, height = 516/72.27, 4.0  # inches; data units are 16 across and 4 high
    fig = plt.figure(figsize=(width, height/16*width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 16), ylim=(0, height), aspect='equal', xticks=[], yticks=[])
    ax.axis('off')
    blue, orange, inhibitory, grey = '#3E6FB0', '#D4512E', '#686078', '#777777'
    excite = ArrowStyle('-|>', head_length=.55, head_width=.25)
    inhibit = ArrowStyle('-[', widthB=.5, lengthB=0)
    gap = .05  # space between a tip or bar and the target rim
    nodes = {'p1': ((2.75, 2.45), .58, blue, 'Pool 1', '240 E'),
             'p2': ((6.65, 2.45), .58, orange, 'Pool 2', '240 E'),
             'i': ((4.7, .98), .63, inhibitory, 'Inhibitory', '400 I'),
             'n': ((1.05, .98), .85, grey, 'Nonselective', '1120 E')}
    # Network block: inputs enter from above and read-outs leave on the right.
    ax.add_patch(FancyBboxPatch((.1, .07), 8.25, 3.2, boxstyle='round,pad=0,rounding_size=.2',
                                facecolor='#F7F6F3', edgecolor='#CFCAC2', lw=.6, zorder=0))
    for (x, y), r, color, name, count in nodes.values():
        ax.add_patch(Circle((x, y), r, facecolor=to_rgba(color, .15), edgecolor=color, lw=.85, zorder=3))
        ax.text(x, y+.03, name, ha='center', va='bottom', fontsize=_ASSET_LABEL_SIZE, zorder=4)
        ax.text(x, y-.06, count, ha='center', va='top', fontsize=_ASSET_TICK_SIZE, zorder=4)

    def arrow(tail, tip, style=excite, color=STROKE_COLOR, lw=.8):
        ax.add_patch(FancyArrowPatch(tail, tip, arrowstyle=style, mutation_scale=7,
                                     shrinkA=0, shrinkB=0, lw=lw, color=color, zorder=2))

    def edge(start, end, offset, *, inhibitory_edge=False):
        """Straight edge parallel to the centre line, from rim to rim."""
        (a, ra, color, *_), (b, rb, *_) = nodes[start], nodes[end]
        a, b = np.asarray(a), np.asarray(b)
        u = (b-a)/np.linalg.norm(b-a)
        n = np.array([-u[1], u[0]])*offset
        arrow(a + (np.sqrt(ra**2-offset**2)+gap)*u + n, b - (np.sqrt(rb**2-offset**2)+gap)*u + n,
              inhibit if inhibitory_edge else excite, inhibitory if inhibitory_edge else color)

    def loop(key, angle, sign, *, inhibitory_edge=False, spread=36, tilt=18, reach=.42):
        """Self-connection around the outward direction `angle`; sign=+1 runs clockwise."""
        (x, y), r, color, *_ = nodes[key]
        ray = lambda deg: np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        start = np.array([x, y]) + (r+gap)*ray(angle+sign*spread)
        end = np.array([x, y]) + (r+gap)*ray(angle-sign*spread)
        path = MPath([start, start+reach*ray(angle+sign*tilt), end+reach*ray(angle-sign*tilt), end],
                     [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle=inhibit if inhibitory_edge else excite,
                                     mutation_scale=7, lw=.8 if inhibitory_edge else 1.0,
                                     color=inhibitory if inhibitory_edge else color, zorder=2))

    # Selective pools excite each other weakly (w_-) and themselves strongly (w_+).
    edge('p1', 'p2', .13)
    edge('p2', 'p1', .13)
    ax.text(4.7, 2.66, r'$w_-$', ha='center', va='bottom', fontsize=_ASSET_LABEL_SIZE)
    for key, angle, side in (('p1', 180, -1), ('p2', 0, 1)):
        loop(key, angle, side)
        (x, y), r, color, *_ = nodes[key]
        ax.text(x+side*(r+.66), y, r'$w_+$', ha='center', va='center', fontsize=_ASSET_LABEL_SIZE, color=color)
    # Shared inhibition: every excitatory population drives it and receives GABA_A back.
    for key in ('p1', 'p2', 'n'):
        edge(key, 'i', .1)
        edge('i', key, .1, inhibitory_edge=True)
    loop('i', -35, 1, inhibitory_edge=True, spread=32, tilt=16, reach=.36)

    # External current into each selective pool.
    for key, idx in (('p1', 1), ('p2', 2)):
        (x, y), r, *_ = nodes[key]
        arrow((x, 3.58), (x, y+r+gap))
        ax.text(x, 3.62, rf'$I_{idx}$', ha='center', va='bottom', fontsize=_ASSET_LABEL_SIZE)
    ax.text(4.7, 3.55, r'$I_k=20\,u_k$ pA per neuron', ha='center', va='center', fontsize=_ASSET_TICK_SIZE)

    # Synapse legend and background drive.
    for y, style, color, label in ((1.22, excite, blue, 'AMPA + NMDA'), (.9, inhibit, inhibitory, r'GABA$_A$')):
        arrow((6.0, y), (6.52, y), style, color)
        ax.text(6.64, y, label, va='center', fontsize=_ASSET_TICK_SIZE)
    ax.text(6.0, .45, 'Poisson drive:\n2400 Hz per neuron', va='center', fontsize=_ASSET_TICK_SIZE,
            linespacing=1.1)

    # Two read-outs: observed spike counts (solid) and the latent proxy (dashed).
    top, bottom, x0 = 2.45, 1.0, 8.35
    arrow((x0, top), (9.35, top))
    rng = np.random.default_rng(3)  # jitter for the schematic spike marks
    for pool, color in enumerate((blue, orange)):
        for row in range(3):
            y = top+.42-pool*.5-row*.17
            for x in np.linspace(9.62, 11.28, 6) + rng.uniform(-.1, .1, 6):
                ax.plot([x, x], [y-.055, y+.055], color=color, lw=.7, solid_capstyle='butt')
    ax.text(10.45, top+.6, '40 neurons per pool', ha='center', va='bottom', fontsize=_ASSET_TICK_SIZE)
    arrow((11.65, top), (12.7, top))
    ax.text(12.18, top+.1, '5-ms bins', ha='center', va='bottom', fontsize=_ASSET_TICK_SIZE)
    box = dict(boxstyle='round,pad=.06,rounding_size=.14', lw=.6, edgecolor=STROKE_COLOR)
    ax.add_patch(FancyBboxPatch((12.85, top-.33), 2.95, .66, facecolor='#F2F0EC', **box))
    ax.text(14.32, top, r'Counts $\mathbf{y}_t\in\mathbb{N}^{80}$', ha='center', va='center',
            fontsize=_ASSET_LABEL_SIZE)
    # Dashed shaft with a solid head, so the head keeps a clean outline.
    ax.plot([x0, 12.45], [bottom, bottom], color=STROKE_COLOR, lw=.8, linestyle=(0, (3, 2)))
    arrow((12.3, bottom), (12.7, bottom))
    ax.text(10.45, bottom+.1, 'Pool-averaged NMDA gating', ha='center', va='bottom', fontsize=_ASSET_LABEL_SIZE)
    ax.add_patch(FancyBboxPatch((12.85, bottom-.33), 2.95, .66, facecolor='white', linestyle=(0, (3, 2)), **box))
    ax.text(14.32, bottom, r'$\mathbf{z}_t=4(\mathbf{s}_t-\frac{1}{2}\mathbf{1})$', ha='center',
            va='center', fontsize=_ASSET_LABEL_SIZE)
    ax.text(14.32, bottom-.45, 'Latent proxy for calibration\nand evaluation', ha='center', va='top',
            fontsize=_ASSET_TICK_SIZE, linespacing=1.1)
    return save_figure(fig, output, plt_module=plt)


def draw_control_phase(ax, experiment_dir: Path, phase_dir: Path) -> None:
    """Saved SNN control trajectories over the fitted zero-input reduced drift.

    Trajectories include known evidence and time-varying control; the background
    is a reference geometry, not the vector field generating those paths.
    """
    from .assets import _asset_baseline_policy_color
    from matplotlib import patheffects
    from matplotlib.lines import Line2D
    from experiments.tnsre.eval_spiking_sessions import EXAMPLES
    manifest = json.loads((phase_dir/'manifest.json').read_text())
    assert manifest['verified'] and manifest['training_seed'] == EXAMPLES['training_seed']
    assert [manifest['task_seed'], manifest['budget_normalized']] == EXAMPLES['sessions']['overturn']
    params = json.loads((experiment_dir/'reference/m2_fit.json').read_text())['physical_parameters']
    grid = np.linspace(-2, 1.5, 61)
    xx, yy = np.meshgrid(grid, grid)
    v = reduced_drift(np.stack([xx, yy], axis=-1), params)
    ax.streamplot(xx, yy, v[...,0], v[...,1], density=.45, color='#B0B0B0',
                  linewidth=.55, arrowsize=.65, zorder=1)
    ax.plot([0,1.5],[-2,-.5], '--', color=STROKE_COLOR, lw=.6, zorder=2)
    for point in fixed_points(params):
        difference = point['state'][0] - point['state'][1]
        color = '#3E6FB0' if difference > 1e-5 else '#C84747' if difference < -1e-5 else STROKE_COLOR
        ax.plot(*point['state'], 'o' if point['stable'] else 'D', ms=3.2,
                mfc=color if point['stable'] else 'white', mec=color, mew=.7, zorder=5)
    styles = {'adaptive':'-', 'spread':'--'}
    records = {r['policy']:r for r in manifest['records']}
    # Draw Constant-amplitude last to separate the two paths where they initially overlap.
    for policy in ['adaptive','spread']:
        r = records[policy]
        z = np.load(phase_dir/f'{policy}.npz')['z'][r['onset']-1:]
        assert np.all(z >= -2.1) and np.all(z <= 1.6), (policy, 'trajectory outside axes')
        color = _asset_baseline_policy_color(policy) if policy=='adaptive' else '#4F4F4F'
        ax.plot(z[:,0], z[:,1], color=color, ls=styles[policy], lw=1.65, zorder=4,
                path_effects=[patheffects.Stroke(linewidth=2.5, foreground='white'), patheffects.Normal()])
        ax.plot(*z[-1], marker='o' if r['success'] else 'x', ms=3.3,
                color=color, mew=.8, zorder=6)
        if policy == 'adaptive':
            ax.plot(*z[0], marker='s', ms=3.2, mfc='white', mec=STROKE_COLOR, mew=.7, zorder=7)
    ax.set(xlim=(-2.1,1.6), ylim=(-2.1,1.6), xlabel=r'$z_1$', ylabel=r'$z_2$')
    ax.set_xticks([-2,0,1]); ax.set_yticks([-2,0,1]); ax.set_aspect('equal')
    ax.set_title('Fitted model', fontsize=_ASSET_LABEL_SIZE, pad=2)
    ax.legend(handles=[Line2D([],[],color=_asset_baseline_policy_color('adaptive'),lw=1.65,label='PALDI'),
                       Line2D([],[],color='#4F4F4F',lw=1.65,ls='--',label='Const.')],
              loc='upper right',fontsize=_ASSET_TICK_SIZE,frameon=True,facecolor='white',
              edgecolor='none',framealpha=.9,handlelength=1.4,handletextpad=.4,borderpad=.25)
    style_experiment_axis(ax)
