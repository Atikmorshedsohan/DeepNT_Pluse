"""Figures and tables built from the files written by run.run_all. No packet-loss plot: there are no loss labels."""
from pathlib import Path


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .metrics import reliability_curve

FULL = 'DeepNT+ (full)'


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)          # otherwise Jupyter auto-displays it a second time, on top of the explicit display() call
    return path


def loss_curve(history, path):
    h = history[history['model'] == FULL].groupby('epoch')[['train_loss', 'val_MAE']].mean()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h.index, h['train_loss']); ax[0].set(title='Training loss (mean over seeds)', xlabel='epoch')
    ax[1].plot(h.index, h['val_MAE'], color='tab:orange'); ax[1].set(title='Validation MAE (ms)', xlabel='epoch')
    return _save(fig, path)


def delay_prediction(pred, path, model=FULL, split='test'):
    d = pred[(pred['model'] == model) & (pred['split'] == split)]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = [0, max(d['y_true'].max(), d['y_pred'].max()) * 1.05]
    sc = ax.scatter(d['y_true'], d['y_pred'], s=10, c=d['sigma_log'] if d['sigma_log'].notna().all() else None, alpha=0.7)
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set(xlabel='true RTT (ms)', ylabel='predicted RTT (ms)', xlim=lim, ylim=lim,
           title=f'{model}: hidden paths, {split} graphs')
    if d['sigma_log'].notna().all():
        fig.colorbar(sc, label='predicted sigma (log space)')
    return _save(fig, path)


def calibration_plot(pred, path, split='test', dist='laplace'):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect')
    for model, d in pred[pred['split'] == split].groupby('model'):
        if d['sigma_log'].notna().all():
            lv, cov = reliability_curve(d['y_true'], d['mu_log'], d['sigma_log'], dist=dist)
            ax.plot(lv, cov, marker='o', ms=3, label=f"{model} ({dist.title()})")
    ax.set(xlabel='nominal interval level', ylabel='empirical coverage', title=f'Uncertainty calibration ({split}, {dist.title()})')
    ax.legend(fontsize=7)
    return _save(fig, path)


def model_comparison(summary, path, split='test'):
    s = summary[summary['split'] == split].sort_values('MAE_mean' if 'MAE_mean' in summary else 'MAE')
    mean = s['MAE_mean'] if 'MAE_mean' in s else s['MAE']
    err = s['MAE_std'].fillna(0) if 'MAE_std' in s else None
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(s) + 1.5))
    ax.barh(s['model'], mean, xerr=err, color='tab:blue')
    ax.invert_yaxis()
    ax.set(xlabel='MAE on hidden paths (ms), lower is better', title=f'Model comparison ({split} graphs)')
    return _save(fig, path)


def active_learning_plot(curves, path):
    mean = curves.groupby(['strategy', 'measured'], as_index=False)['MAE'].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, d in mean.groupby('strategy'):
        ax.plot(d['measured'], d['MAE'], marker='o', ms=3, label=s)
    ax.set(xlabel='extra paths measured', ylabel='MAE on held-out hidden paths (ms)', title='Active measurement scheduling')
    ax.legend(fontsize=8)
    return _save(fig, path)


def make_all(out_dir, dist=None):
    out_dir = Path(out_dir)
    pred = pd.read_csv(out_dir / 'predictions.csv')
    summary = pd.read_csv(out_dir / 'metrics_summary.csv')
    if dist is None:
        dist = 'laplace'
        cfg_file = out_dir / 'config.json'
        if cfg_file.exists():
            try:
                import json
                with open(cfg_file) as f:
                    cfg_data = json.load(f)
                    dist = cfg_data.get('loss_type', 'laplace')
            except Exception:
                dist = 'laplace'
    files = [loss_curve(pd.read_csv(out_dir / 'history.csv'), out_dir / 'loss_curve.png'),
             delay_prediction(pred, out_dir / 'delay_prediction.png'),
             calibration_plot(pred, out_dir / 'calibration.png', dist=dist),
             model_comparison(summary, out_dir / 'model_comparison.png')]
    if (out_dir / 'active_learning.csv').exists():
        files.append(active_learning_plot(pd.read_csv(out_dir / 'active_learning.csv'), out_dir / 'active_learning.png'))
    return files
