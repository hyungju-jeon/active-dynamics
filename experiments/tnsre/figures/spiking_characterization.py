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
    """
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
    from matplotlib.colors import to_rgba
    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(516/72.27, 2.6))
    ax = fig.add_axes([.015,.02,.97,.96])
    ax.set(xlim=(0,16),ylim=(0,5.8),aspect='equal')
    ax.axis('off'); ax.set_xticks([]); ax.set_yticks([])
    blue, orange, inhibitory = '#3E6FB0', '#D4512E', '#686078'
    nodes = {'p1':(2.0,3.0,.70,blue,'Pool 1\n240 E'),
             'p2':(8.0,3.0,.70,orange,'Pool 2\n240 E'),
             'n':(5.0,4.55,.76,'#777777','Nonselective\n1120 E'),
             'i':(5.0,1.45,.67,inhibitory,'Inhibitory\n400 I')}
    patches = {}
    for key,(x,y,r,color,label) in nodes.items():
        node=Circle((x,y),r,facecolor=to_rgba(color,.15),edgecolor=color,lw=.85,zorder=3)
        patches[key]=node; ax.add_patch(node)
        ax.text(x,y,label,ha='center',va='center',fontsize=_ASSET_LABEL_SIZE,zorder=4)

    def connect(start,end,*,inhibit=False,rad=.15,color=None,lw=.85):
        x,y,_,c,_=nodes[start]; u,v,*_=nodes[end]
        ax.add_patch(FancyArrowPatch((x,y),(u,v),patchA=patches[start],patchB=patches[end],
                     connectionstyle=f'arc3,rad={rad}',arrowstyle='-[' if inhibit else '-|>',
                     mutation_scale=3.5 if inhibit else 7,shrinkA=2,shrinkB=2,
                     lw=lw,color=color or c,zorder=2))

    # Shared inhibition receives excitation from every excitatory population.
    for pool in ('p1','p2','n'):
        connect(pool,'i',rad=.17 if pool!='n' else .12,lw=.75)
        connect('i',pool,inhibit=True,rad=.17 if pool!='n' else .12,lw=.75)
    # Weaker cross-pool excitation; recurrent loops are stronger within each pool.
    connect('p1','p2',rad=-.17)
    connect('p2','p1',rad=-.17)
    # Keep weight labels off the central nonselective/inhibitory connections.
    for x,y in [(3.6,3.48),(6.4,2.52)]:
        ax.text(x,y,r'$w_-$',ha='center',va='center',fontsize=_ASSET_TICK_SIZE,
                bbox=dict(facecolor='white',edgecolor='none',pad=.2),zorder=5)
    for key,side in [('p1',-1),('p2',1)]:
        x,y,r,c,_=nodes[key]
        ax.add_patch(FancyArrowPatch((x+side*.61,y+.36),(x+side*.61,y-.36),
                     connectionstyle=f'arc3,rad={-side*1.45}',arrowstyle='-|>',mutation_scale=7,
                     lw=1.0,color=c,zorder=2))
        ax.text(x+side*1.08,y+.8,r'$w_+$',ha='center',fontsize=_ASSET_LABEL_SIZE,color=c)
        idx=1 if key=='p1' else 2
        ax.annotate('',xy=(x,y+r+.03),xytext=(x,5.10),
                    arrowprops=dict(arrowstyle='-|>',lw=.9,color=STROKE_COLOR,mutation_scale=7))
        ax.text(x,5.45,rf'$I_{idx}=20u_{idx}$ pA',ha='center',fontsize=_ASSET_LABEL_SIZE)
        ax.text(x,5.12,'per neuron',ha='center',fontsize=_ASSET_TICK_SIZE)
    ax.add_patch(FancyArrowPatch((4.63,.90),(5.37,.90),connectionstyle='arc3,rad=.8',
                 arrowstyle='-[',mutation_scale=3.5,lw=.75,color=inhibitory,zorder=2))

    # Explicit synaptic symbols, matching the main-text circuit schematic.
    ax.annotate('',xy=(1.1,.58),xytext=(.35,.58),
                arrowprops=dict(arrowstyle='-|>',lw=.8,color=blue,mutation_scale=7))
    ax.text(1.25,.58,'AMPA + NMDA',va='center',fontsize=_ASSET_TICK_SIZE)
    ax.annotate('',xy=(7.2,.58),xytext=(6.45,.58),
                arrowprops=dict(arrowstyle='-[',lw=.8,color=inhibitory,mutation_scale=3.5))
    ax.text(7.35,.58,r'GABA$_A$',va='center',fontsize=_ASSET_TICK_SIZE)
    ax.text(5,.08,'Independent Poisson drive: 2400 Hz per neuron',ha='center',
            va='bottom',fontsize=_ASSET_TICK_SIZE)

    # The observation branch is separate from the network's recurrent edges.
    ax.annotate('',xy=(11.0,3.0),xytext=(9.1,3.0),
                arrowprops=dict(arrowstyle='-|>',lw=.8,color=STROKE_COLOR,mutation_scale=7))
    ax.text(10.05,3.28,'Record',ha='center',fontsize=_ASSET_LABEL_SIZE)
    ax.text(10.05,2.42,'40 neurons\nfrom each pool',ha='center',va='center',fontsize=_ASSET_TICK_SIZE)
    ax.text(13.25,5.15,'Spiking observations',ha='center',fontsize=_ASSET_LABEL_SIZE)
    # Three stylized rows per pool indicate recorded spikes, not sample data.
    for pool,color in enumerate((blue,orange)):
        for row in range(3):
            y=4.6-pool*.58-row*.14
            for x in (11.65+.12*row,12.2+.18*row,13.1-.1*row,14.1+.08*row,14.8-.1*row):
                ax.plot([x,x],[y-.05,y+.05],color=color,lw=.7)
    ax.text(13.25,3.5,'5-ms bins; 80 count channels',ha='center',fontsize=_ASSET_LABEL_SIZE)
    box=FancyBboxPatch((11.0,2.38),4.5,.72,boxstyle='round,pad=.06,rounding_size=.16',
                      facecolor='#F2F0EC',edgecolor=STROKE_COLOR,lw=.6)
    ax.add_patch(box)
    ax.text(13.25,2.74,r'Observed counts $\mathbf{y}_t\in\mathbb{N}^{80}$',
            ha='center',va='center',fontsize=_ASSET_LABEL_SIZE)
    ax.annotate('',xy=(13.25,3.13),xytext=(13.25,3.38),
                arrowprops=dict(arrowstyle='-|>',lw=.7,color=STROKE_COLOR,mutation_scale=6))
    ax.text(13.25,1.7,'Pool-averaged NMDA gating',ha='center',fontsize=_ASSET_LABEL_SIZE)
    ax.text(13.25,1.2,r'$\mathbf{z}_t=4(\mathbf{s}_t-\frac{1}{2}\mathbf{1})$',
            ha='center',fontsize=_ASSET_LABEL_SIZE)
    ax.text(13.25,.62,'Latent proxy for calibration\nand evaluation',ha='center',va='center',
            fontsize=_ASSET_TICK_SIZE)
    return save_figure(fig,output,plt_module=plt)


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
    # Draw Uniform last to separate the two paths where they initially overlap.
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
    ax.set_title(r'Fitted model, $\mathbf{u}=0$', fontsize=_ASSET_LABEL_SIZE, pad=2)
    ax.legend(handles=[Line2D([],[],color=_asset_baseline_policy_color('adaptive'),lw=1.65,label='PALDI'),
                       Line2D([],[],color='#4F4F4F',lw=1.65,ls='--',label='Uniform')],
              loc='upper right',fontsize=_ASSET_TICK_SIZE,frameon=True,facecolor='white',
              edgecolor='none',framealpha=.9,handlelength=1.4,handletextpad=.4,borderpad=.25)
    style_experiment_axis(ax)
