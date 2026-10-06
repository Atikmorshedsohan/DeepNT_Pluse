"""Simulated active measurement scheduling.

A graph's hidden paths were all really measured, so "measuring" one means revealing its stored RTT and adding it
to the context. A fixed 25% of the hidden paths is held out for scoring and is never acquired, so the curves of
different strategies are comparable. Nothing is generated: every revealed value is a real measurement.
"""
import numpy as np
import pandas as pd
import torch

from .train import eval_episode, predict_frame

STRATEGIES = ('random', 'variance', 'bald', 'edge_coverage', 'longest_path')


def _scores(strategy, frame, g, ctx, pool, rng):
    n = len(pool)
    if strategy == 'random':
        return rng.random(n)
    if strategy == 'longest_path':
        return g.hops[torch.tensor(pool)].numpy()
    if strategy == 'edge_coverage':
        covered = (g.A[torch.tensor(ctx)].sum(0) > 0).float()
        return (g.A[torch.tensor(pool)] @ (1 - covered)).numpy()
    epi = frame['var_epistemic_z'].to_numpy()[:n]
    alea = np.nan_to_num(frame['var_aleatoric_z'].to_numpy()[:n], nan=0.0)
    if strategy == 'variance':                                  # a(P) = predictive variance
        return epi + alea
    if strategy == 'bald':                                      # mutual information of a Gaussian predictive
        return 0.5 * np.log1p(epi / np.maximum(alea, 1e-6)) if alea.any() else epi
    raise ValueError(strategy)


def simulate(model, g, cfg, strategy, rounds=6, frac=0.10, mc=20, seed=0):
    assert not cfg.causal, "active-learning simulation uses the static (non-causal) model"
    ctx0, hidden = eval_episode(g)
    hidden = hidden.tolist()
    if len(hidden) < 8:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    holdout_rng = np.random.default_rng(1234 + int(g.gid.split('_')[-1]))     # same hold-out for every strategy
    order = holdout_rng.permutation(len(hidden))
    n_eval = max(2, len(hidden) // 4)
    eval_idx = [hidden[i] for i in order[:n_eval]]
    pool = [hidden[i] for i in order[n_eval:]]
    ctx = ctx0.tolist()
    k = max(1, int(round(frac * len(pool))))
    rows = []
    for r in range(rounds + 1):
        use_mc = mc if strategy in ('variance', 'bald') else 0
        frame = predict_frame(model, g, cfg, ctx=torch.tensor(ctx), tgt=torch.tensor(pool + eval_idx), mc=use_mc)
        ev = frame.iloc[len(pool):]
        err = (ev['y_pred'] - ev['y_true']).to_numpy()
        rows.append({'graph_id': g.gid, 'strategy': strategy, 'seed': seed, 'round': r,
                     'measured': len(ctx) - len(ctx0), 'MAE': float(np.abs(err).mean()),
                     'RMSE': float(np.sqrt((err ** 2).mean()))})
        if r == rounds or len(pool) <= k:
            break
        pick = np.argsort(-_scores(strategy, frame, g, ctx, pool, rng))[:k]
        ctx += [pool[i] for i in pick]
        pool = [p for i, p in enumerate(pool) if i not in set(pick.tolist())]
    return pd.DataFrame(rows)


def run(model, graphs, cfg, split='test', strategies=STRATEGIES, random_seeds=5, **kw):
    frames = []
    for g in graphs.values():
        if g.split != split:
            continue
        for s in strategies:
            for seed in (range(random_seeds) if s == 'random' else [0]):
                frames.append(simulate(model, g, cfg, s, seed=seed, **kw))
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(curves):
    """Mean MAE per strategy and round over graphs, and the area under the MAE curve (lower is better)."""
    mean = curves.groupby(['strategy', 'round', 'measured'], as_index=False)['MAE'].mean()
    auc = mean.groupby('strategy')['MAE'].mean().rename('mean_MAE_over_rounds').reset_index()
    return mean, auc.sort_values('mean_MAE_over_rounds')
