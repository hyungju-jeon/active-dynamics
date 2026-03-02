#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _load_trace(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / 'parameter_error_trace.csv').sort_values('step').reset_index(drop=True)
    return df

def _load_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / 'run_metadata.json').read_text())

def _estimate_param_path(df: pd.DataFrame, meta: dict) -> np.ndarray:
    theta0 = np.array([1.0, 1.0], float)
    thetaT = np.array(meta.get('embedding_estimate', [1.0, 1.0]), float)
    err = df['parameter_error'].to_numpy(float)
    e0, eT = float(err[0]), float(err[-1])
    prog = np.clip((e0 - err) / max(e0 - eT, 1e-8), 0, 1)
    return theta0[None, :] + prog[:, None] * (thetaT[None, :] - theta0[None, :])

def _info_gain_proxy_grid(theta: np.ndarray, X: np.ndarray, V: np.ndarray) -> np.ndarray:
    a, b = float(theta[0]), float(theta[1])
    A = np.abs(a * V - b * X - 0.1 * X**3)
    S = np.sqrt(V**2 + X**2)
    return A * (1.0 + 0.2 * S)

def make_info_gain_video(run_dir: Path, out_path: Path, stride: int = 4):
    df, meta = _load_trace(run_dir), _load_meta(run_dir)
    path = _estimate_param_path(df, meta)
    x = np.linspace(-4, 4, 100); v = np.linspace(-4, 4, 100); X, V = np.meshgrid(x, v)
    frames=[]
    for i in range(0, len(df), max(1, stride)):
        G = _info_gain_proxy_grid(path[i], X, V)
        fig, ax = plt.subplots(figsize=(6,5))
        im=ax.imshow(G, extent=[x.min(),x.max(),v.min(),v.max()], origin='lower', cmap='magma', aspect='auto')
        ax.set_xlabel('x'); ax.set_ylabel('v'); ax.set_title(f'Information-gain proxy map (step={int(df.loc[i,"step"])})')
        fig.colorbar(im, ax=ax, label='proxy IG'); fig.tight_layout(); fig.canvas.draw()
        frame=np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame); plt.close(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True); imageio.mimsave(out_path, frames, fps=10)

def make_latent_trajectory_video(run_dir: Path, out_path: Path, stride: int = 4):
    df, meta = _load_trace(run_dir), _load_meta(run_dir)
    theta_true=np.array(meta.get('embedding_true',[0,0]),float); path=_estimate_param_path(df,meta)
    xmin=min(path[:,0].min(),theta_true[0])-0.5; xmax=max(path[:,0].max(),theta_true[0])+0.5
    ymin=min(path[:,1].min(),theta_true[1])-0.5; ymax=max(path[:,1].max(),theta_true[1])+0.5
    frames=[]
    for i in range(1,len(df),max(1,stride)):
        fig,ax=plt.subplots(figsize=(6,5))
        ax.plot(path[:i,0],path[:i,1],color='tab:blue',lw=2,label='estimated trajectory')
        ax.scatter([theta_true[0]],[theta_true[1]],color='tab:red',s=70,label='true')
        est=path[i]; err=theta_true-est
        ax.quiver(est[0],est[1],err[0],err[1],angles='xy',scale_units='xy',scale=1,color='tab:orange',width=0.007,label='true-est vector')
        ax.scatter([est[0]],[est[1]],color='tab:blue',s=40)
        ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_xlabel('a parameter'); ax.set_ylabel('b parameter')
        ax.set_title(f'Latent parameter trajectory (step={int(df.loc[i,"step"])})'); ax.grid(alpha=0.25); ax.legend(loc='best')
        fig.tight_layout(); fig.canvas.draw()
        frame=np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame); plt.close(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True); imageio.mimsave(out_path, frames, fps=10)

def make_extra_figure(summary_dir: Path):
    df=pd.read_csv(summary_dir/'trajectory_r2_over_steps.csv').dropna(subset=['cpu_time_sec_mean','trajectory_r2_mean'])
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for (m,e),sub in df.groupby(['model_tag','exp_id']): ax.plot(sub['cpu_time_sec_mean'], sub['trajectory_r2_mean'], label=f'{m}:{e}')
    ax.set_xlabel('CPU Time (sec)'); ax.set_ylabel('Trajectory R2'); ax.set_title('Trajectory R2 over CPU time'); ax.grid(alpha=0.25); ax.legend(loc='best',fontsize=8)
    fig.tight_layout(); out=summary_dir/'figures'/'trajectory_r2_over_cpu_time.png'; out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out,dpi=150); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',required=True); ap.add_argument('--summary-dir',required=True)
    ap.add_argument('--model-tag',default='updated'); ap.add_argument('--exp-id',default='active_short'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--repeat',default='repeat_01'); args=ap.parse_args()
    base=Path(args.base_dir); run_dir=base/'tracks'/args.model_tag/args.exp_id/f'seed_{args.seed}'/args.repeat; media=Path(args.summary_dir)/'media'
    make_info_gain_video(run_dir, media/'info_gain_proxy_over_time.mp4'); make_latent_trajectory_video(run_dir, media/'latent_parameter_trajectory.mp4'); make_extra_figure(Path(args.summary_dir))
    print(f'Generated media in: {media}')

if __name__=='__main__': main()
