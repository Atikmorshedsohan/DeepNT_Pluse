"""Non-graph baselines (spec section 16): median, hop-count regression, Linear Regression, Random Forest, MLP
on structural path features, and the classical convex tomography solution. All are scored on the same hidden
paths (observed == 0, not an outlier) as the neural models."""
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .train import eval_episode


def path_features(g, monitors):
    """Structural features only: nothing here depends on any RTT."""
    feats = g.x.numpy()
    rows = []
    for p in range(g.n_paths):
        nodes = g.node_pad[p][g.node_mask[p]].numpy()
        nf = feats[nodes]
        ef = g.edge_feat[torch.nonzero(g.A[p]).squeeze(-1), 0].numpy()
        mon = np.array([g.monitor[p] == m for m in monitors], float)
        rows.append(np.concatenate([[g.hops[p], np.log1p(g.hops[p])], nf.mean(0), nf.max(0), nf.min(0),
                                    feats[nodes[0]], feats[nodes[-1]], [ef.sum(), ef.mean(), ef.max()], mon]))
    return np.array(rows)


def _frame(g, idx, pred, name):
    return pd.DataFrame({'graph_id': g.gid, 'path_id': [g.path_ids[i] for i in idx],
                         'source': [g.sources[i] for i in idx], 'destination': [g.destinations[i] for i in idx],
                         'observed': 0, 'split': g.split, 'y_true': g.y[idx].numpy(), 'y_pred': np.asarray(pred, float),
                         'mu_log': np.log1p(np.clip(pred, 0, None)), 'sigma_log': np.nan, 'hops': g.hops[idx].numpy(),
                         'model': name})


def run_structural_baselines(graphs, seed=0):
    monitors = sorted({m for g in graphs.values() for m in g.monitor})
    X = {}
    for k, g in graphs.items():
        X[k] = path_features(g, monitors)
    train = [g for g in graphs.values() if g.split == 'train']
    Xtr = np.concatenate([X[g.gid][g.ok.numpy()] for g in train])
    ytr = np.concatenate([g.y[g.ok].numpy() for g in train])
    scaler = StandardScaler().fit(Xtr)
    hop_col = 0

    lo, hi = np.log1p(ytr.min()), np.log1p(ytr.max())                # keep predictions inside the training range
    log_models = {                                                     # name -> (constructor(h), hyper-parameter grid)
        'Linear regression': (lambda h: Ridge(alpha=h), [0.1, 1, 10, 100, 1000]),
        'Random forest': (lambda h: RandomForestRegressor(n_estimators=300, min_samples_leaf=h, random_state=seed,
                                                          n_jobs=-1), [1, 3, 8]),
        'MLP': (lambda h: MLPRegressor(hidden_layer_sizes=(64, 32), alpha=h, max_iter=800, early_stopping=True,
                                       random_state=seed), [1e-3, 1e-1, 1, 10]),
    }

    def val_mae(predict):
        errs = []
        for g in graphs.values():
            if g.split == 'val':
                idx = torch.nonzero(~g.observed & g.ok).squeeze(-1)
                errs.append(np.abs(predict(X[g.gid][idx.numpy()]) - g.y[idx].numpy()))
        return float(np.concatenate(errs).mean())

    predictors = {'Median': (lambda Xs, med=float(np.median(ytr)): np.full(len(Xs), med), None)}
    hop = Ridge(alpha=1.0).fit(Xtr[:, [hop_col]], ytr)
    predictors['Hop-count linear'] = (lambda Xs: np.clip(hop.predict(Xs[:, [hop_col]]), 0, None), None)
    for name, (make, grid) in log_models.items():                      # tuned on the validation graphs only
        best = None
        for h in grid:
            est = make(h).fit(scaler.transform(Xtr), np.log1p(ytr))
            predict = lambda Xs, est=est: np.expm1(np.clip(est.predict(scaler.transform(Xs)), lo, hi))
            mae = val_mae(predict)
            if best is None or mae < best[0]:
                best = (mae, predict, h)
        predictors[name] = (best[1], best[2])

    frames = []
    for name, (predict, hyper) in predictors.items():
        for g in graphs.values():
            if g.split == 'train':
                continue
            idx = torch.nonzero(~g.observed & g.ok).squeeze(-1)
            frames.append(_frame(g, idx.tolist(), predict(X[g.gid][idx.numpy()]), name).assign(hyperparameter=hyper))
    return pd.concat(frames, ignore_index=True)


def convex_tomography(g, alpha, edge_ms, intercept=0.0):
    """Classical tomography: min ||A x - y||^2 + alpha ||x - x0||^2, x >= 0 (x0 = constant per-edge prior),
    fitted on the observed paths of this graph and applied to the hidden ones."""
    ctx, tgt = eval_episode(g)
    A = g.A.numpy()
    Ac = np.hstack([A[ctx.numpy()], np.ones((len(ctx), 1))])
    x0 = np.concatenate([np.full(A.shape[1], edge_ms), [intercept]])
    resid = g.y[ctx].numpy() - Ac @ x0
    x = np.clip(x0 + Ac.T @ np.linalg.solve(Ac @ Ac.T + alpha * np.eye(len(ctx)), resid), 0, None)
    return tgt.tolist(), np.hstack([A[tgt.numpy()], np.ones((len(tgt), 1))]) @ x


def run_convex_baseline(graphs, norm, alphas=(0.1, 1, 10, 100, 1000)):
    def frames_for(split, alpha):
        return [_frame(g, *convex_tomography(g, alpha, norm['prior_edge_ms']), 'Convex tomography (classical)')
                for g in graphs.values() if g.split == split]
    val_mae = {a: np.mean(np.abs(np.concatenate([(f['y_pred'] - f['y_true']).to_numpy() for f in frames_for('val', a)])))
               for a in alphas}
    best = min(val_mae, key=val_mae.get)
    out = pd.concat(frames_for('val', best) + frames_for('test', best), ignore_index=True)
    out['alpha'] = best
    return out
