"""Episodic training, causal evaluation and prediction export.

Protocol (spec sections 7, 8, 17):
- Graph-level split: train graphs fit parameters, validation graphs select the checkpoint, test graphs are scored once.
- In every episode a set of *context* paths shows its RTT to the model and the remaining *target* paths are
  predicted. Train episodes draw random masks; validation/test use the fixed observed/unobserved masks.
- Causal mode (needed by the temporal branch): targets are cut into time blocks and each block only sees
  context measurements that are strictly earlier than the whole block.
- Outlier-flagged paths (quality_flag rtt_outlier*) are excluded from context, training loss and metrics.
"""
import copy
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .metrics import regression_metrics, calibration_metrics


@dataclass
class TrainConfig:
    epochs: int = 150
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 20
    ctx_frac: tuple = (0.25, 0.55)           # Centered around the 0.45 evaluation observation rate
    train_label_scope: str = 'all_paths'     # all_paths | observed_only
    causal: bool = False
    n_blocks: int = 6
    loss_type: str = 'laplace'               # laplace (robust, MAE-aligned) | gaussian
    mse_weight: float = 0.5
    tomo_weight: float = 0.5
    loss_weight: float = 0.5
    grad_clip: float = 1.0
    mc_samples: int = 20
    use_cosine_lr: bool = True
    eta_min: float = 1e-5
    residual_reg_weight: float = 0.05
    episodes_per_graph: int = 8              # Multiple episodes per graph per epoch (solves gradient step deficiency)
    val_masks: int = 4                       # Multi-mask smoothed validation to stabilize checkpoint selection

    def to_dict(self):
        return asdict(self)


def causal_blocks(g, ctx, tgt, n_blocks):
    order = tgt[torch.argsort(g.t[tgt])]
    blocks = []
    for chunk in torch.tensor_split(order, n_blocks):
        if chunk.numel():
            blocks.append((ctx[g.t[ctx] < g.t[chunk].min()], chunk))
    return blocks


def make_blocks(g, ctx, tgt, cfg):
    return causal_blocks(g, ctx, tgt, cfg.n_blocks) if cfg.causal else [(ctx, tgt)]


def sample_episode(g, rng, cfg):
    pool = g.ok & g.observed if cfg.train_label_scope == 'observed_only' else g.ok.clone()
    idx = np.flatnonzero(pool.cpu().numpy() if isinstance(pool, torch.Tensor) else pool.numpy())
    rng.shuffle(idx)
    n_ctx = max(2, int(round(rng.uniform(*cfg.ctx_frac) * len(idx))))
    if len(idx) - n_ctx < 1:
        return None
    dev = g.x.device
    return torch.tensor(np.sort(idx[:n_ctx]), device=dev), torch.tensor(np.sort(idx[n_ctx:]), device=dev)


def eval_episode(g):
    ctx = torch.nonzero(g.observed & g.ok).squeeze(-1)
    tgt = torch.nonzero(~g.observed & g.ok).squeeze(-1)
    return ctx, tgt


def compute_loss(model, g, ctx, tgt, cfg):
    out = model(g, ctx, tgt, g.y[ctx])                 # only context RTTs enter the model
    z, mu = model.z_of(g.y[tgt]), out['mu_z']          # target RTT is used here, in the loss, and nowhere else
    use_laplace = getattr(cfg, 'loss_type', 'laplace') == 'laplace'

    if out['logvar'] is not None:
        lv = out['logvar']
        if use_laplace:
            # Laplace NLL: e^{-s} * |z - mu| + s (where lv = s is the log-scale parameter)
            loss = (torch.exp(-lv) * torch.abs(z - mu) + lv).mean() + cfg.mse_weight * torch.abs(z - mu).mean()
        else:
            loss = (0.5 * (lv + (z - mu) ** 2 * torch.exp(-lv))).mean() + cfg.mse_weight * ((z - mu) ** 2).mean()
    else:
        if use_laplace:
            loss = torch.abs(z - mu).mean()
        else:
            loss = ((z - mu) ** 2).mean()

    if model.cfg.tomography and ctx.numel():
        if use_laplace:
            loss = loss + cfg.tomo_weight * torch.abs(model.z_of(out['y_add_ctx']) - model.z_of(g.y[ctx])).mean()
        else:
            loss = loss + cfg.tomo_weight * ((model.z_of(out['y_add_ctx']) - model.z_of(g.y[ctx])) ** 2).mean()

    # Residual regularization penalty: keep neural refinement bounded
    if getattr(cfg, 'residual_reg_weight', 0.0) > 0 and 'y_convex' in out and 'y_hat' in out:
        loss = loss + cfg.residual_reg_weight * torch.abs(model.z_of(out['y_hat']) - model.z_of(out['y_convex'])).mean()

    if out['loss_logit'] is not None:
        lab = g.loss[tgt]
        keep = ~torch.isnan(lab)
        if keep.any():
            loss = loss + cfg.loss_weight * F.binary_cross_entropy_with_logits(out['loss_logit'][keep], lab[keep])
    return loss


def predict_frame(model, g, cfg, ctx=None, tgt=None, mc=0):
    """Predictions for the hidden paths of one graph (or an explicit ctx/tgt pair)."""
    if ctx is None:
        ctx, tgt = eval_episode(g)
    rows = []
    use_laplace = getattr(cfg, 'loss_type', 'laplace') == 'laplace'
    for c_idx, t_idx in make_blocks(g, ctx, tgt, cfg):
        model.eval()
        with torch.no_grad():
            out = model(g, c_idx, t_idx, g.y[c_idx])
            var_epi = torch.zeros(t_idx.numel())
            if out['logvar'] is not None:
                # In Laplace NLL, lv = s = log(b). Var(Y) = 2*b^2 = 2*exp(2*lv)
                # In Gaussian NLL, lv = log(sigma^2). Var(Y) = exp(lv)
                var_alea = 2.0 * torch.exp(2.0 * out['logvar']) if use_laplace else torch.exp(out['logvar'])
            else:
                var_alea = torch.full((t_idx.numel(),), float('nan'))
            if mc > 0:
                model.train()
                passes = [model(g, c_idx, t_idx, g.y[c_idx]) for _ in range(mc)]
                model.eval()
                var_epi = torch.stack([p['mu_z'] for p in passes]).var(0)
                if passes[0]['logvar'] is not None:
                    if use_laplace:
                        var_alea = torch.stack([2.0 * torch.exp(2.0 * p['logvar']) for p in passes]).mean(0)
                    else:
                        var_alea = torch.stack([torch.exp(p['logvar']) for p in passes]).mean(0)
        total = (var_epi + var_alea) if out['logvar'] is not None else var_epi
        sigma = torch.sqrt(total) * model.log_rtt_std if (mc > 0 or out['logvar'] is not None) else torch.full_like(total, float('nan'))
        for j, p in enumerate(t_idx.tolist()):
            rows.append({'graph_id': g.gid, 'path_id': g.path_ids[p], 'source': g.sources[p],
                         'destination': g.destinations[p], 'observed': int(g.observed[p]), 'split': g.split,
                         'y_true': float(g.y[p]), 'y_pred': float(out['y_hat'][j]),
                         'mu_log': float(out['mu_z'][j] * model.log_rtt_std + model.log_rtt_mean),
                         'sigma_log': float(sigma[j]), 'var_epistemic_z': float(var_epi[j]),
                         'var_aleatoric_z': float(var_alea[j]), 'convex_weight': float(out['lam'][j]),
                         'edge_coverage': float(out['cov'][j]), 'n_context': int(c_idx.numel()),
                         'hops': float(g.hops[p])})
    return pd.DataFrame(rows)


def evaluate(model, graphs, split, cfg, mc=0):
    frames = [predict_frame(model, g, cfg, mc=mc) for g in graphs.values() if g.split == split]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score(frame, dist='laplace'):
    """Regression metrics, plus calibration when the model produced an uncertainty."""
    out = regression_metrics(frame['y_true'], frame['y_pred'])
    if frame['sigma_log'].notna().all():
        out.update(calibration_metrics(frame['y_true'], frame['mu_log'], frame['sigma_log'], dist=dist))
    return out


def train_model(model, graphs, cfg, seed=0, log=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if getattr(cfg, 'use_cosine_lr', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=getattr(cfg, 'eta_min', 1e-5)
        )
    train_ids = [k for k, g in graphs.items() if g.split == 'train']
    val_graphs = [g for g in graphs.values() if g.split == 'val']
    episodes_per_graph = getattr(cfg, 'episodes_per_graph', 8)
    n_val_masks = getattr(cfg, 'val_masks', 4)
    best, best_state, stale, history = float('inf'), None, 0, []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = []
        for _ in range(episodes_per_graph):
            for k in rng.permutation(train_ids):
                g = graphs[k]
                episode = sample_episode(g, rng, cfg)
                if episode is None:
                    continue
                blocks = make_blocks(g, *episode, cfg)
                n_tgt = sum(t.numel() for _, t in blocks)
                loss = sum(compute_loss(model, g, c, t, cfg) * t.numel() / n_tgt for c, t in blocks)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                epoch_loss.append(float(loss.detach()))
        if scheduler is not None:
            scheduler.step()

        # 1. Base canonical validation evaluation
        val_frame = evaluate(model, graphs, 'val', cfg)
        val = score(val_frame, dist=getattr(cfg, 'loss_type', 'laplace')) if not val_frame.empty else {'MAE': float('nan'), 'RMSE': float('nan')}
        
        # 2. Multi-mask validation smoothing to prevent noisy checkpoint jitter
        val_maes = [val['MAE']] if not np.isnan(val['MAE']) else []
        if n_val_masks > 1 and val_graphs:
            val_rng = np.random.default_rng(10000 + epoch)
            for _ in range(n_val_masks - 1):
                for vg in val_graphs:
                    ep = sample_episode(vg, val_rng, cfg)
                    if ep is not None:
                        f_ep = predict_frame(model, vg, cfg, ctx=ep[0], tgt=ep[1])
                        if not f_ep.empty:
                            val_maes.append(float(np.mean(np.abs(f_ep['y_pred'] - f_ep['y_true']))))
        val_mae_smooth = float(np.mean(val_maes)) if val_maes else val['MAE']

        history.append({'epoch': epoch, 'train_loss': float(np.mean(epoch_loss)), 'val_MAE': val['MAE'],
                        'val_MAE_smooth': val_mae_smooth, 'val_RMSE': val['RMSE']})
        if log and (epoch % 10 == 0 or epoch == 1):
            log(f"epoch {epoch:3d} | train loss {history[-1]['train_loss']:.4f} | val MAE {val['MAE']:.2f} ms (smooth {val_mae_smooth:.2f} ms)")
        if val_mae_smooth < best - 1e-6:
            best, best_state, stale = val_mae_smooth, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history)
