"""DeepNT+ : split-weight spatial GNN + tomography layer + ridge (convex) refinement + temporal branch
(TGN memory, GRU, Transformer) + heteroscedastic uncertainty head, fused per path.

The forward pass receives `y_ctx`, the RTT of the context paths only. It never reads `g.y`, so a hidden
target RTT cannot reach the prediction (tests/test_deepnt_plus.py enforces this).
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GNN, PathEncoder
from .tgn import TemporalBranch


@dataclass
class ModelConfig:
    node_dim: int = 8
    edge_dim: int = None             # 1 (path count only) or 2 (with geographical distance); None auto-detects from norm
    d: int = 64
    gnn_kind: str = 'split'          # split | sage | gat
    gnn_layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    context_features: bool = True    # observation encoder: node-level summary of the context measurements
    tomography: bool = True          # additive path model y = A x + b with GNN-predicted edge latents x >= 0
    convex: bool = True              # ridge refinement of x from the context paths + learned fusion gate
    temporal: bool = False           # TGN memory + GRU + Transformer over the measurement history (causal only)
    uncertainty: bool = True         # heteroscedastic Gaussian head
    has_loss: bool = False           # packet-loss head (only trained where labels exist)
    history_k: int = 8
    mem_dim: int = 32

    def to_dict(self):
        return asdict(self)


class DeepNTPlus(nn.Module):
    def __init__(self, cfg: ModelConfig, norm: dict):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        for k in ('log_rtt_mean', 'log_rtt_std', 'rate_mean', 'rate_std', 'prior_edge_ms'):
            self.register_buffer(k, torch.tensor(float(norm[k])))
        self.gnn = GNN(cfg.gnn_kind, cfg.node_dim + 2, d, cfg.gnn_layers, cfg.dropout, cfg.heads)
        edge_feat_dim = cfg.edge_dim if cfg.edge_dim is not None else norm.get('edge_dim', 1)
        self.edge_mlp = nn.Sequential(nn.Linear(2 * d + edge_feat_dim + 1, d), nn.ReLU(), nn.Linear(d, 1))
        self.bias_raw = nn.Parameter(torch.zeros(()))
        self.path_enc = PathEncoder(d, cfg.heads, 1, cfg.dropout)
        self.temporal = TemporalBranch(d, cfg.mem_dim, 16, cfg.history_k, cfg.heads, cfg.dropout) if cfg.temporal else None
        self.res_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.res_head[-1].weight)
        nn.init.zeros_(self.res_head[-1].bias)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.e ** 10 - 1)))          # ridge strength, starts at 10
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)                                      # anchor on convex estimate (residual form)
        self.var_head = nn.Sequential(nn.Linear(2 * d + 1, d), nn.ReLU(), nn.Linear(d, 1)) if cfg.uncertainty else None
        self.loss_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1)) if cfg.has_loss else None

    def z_of(self, y_ms):
        return (torch.log1p(y_ms) - self.log_rtt_mean) / self.log_rtt_std

    def _rate_z(self, y_ms, hops):
        return (torch.log1p(y_ms / hops) - self.rate_mean) / self.rate_std

    def forward(self, g, ctx, tgt, y_ctx):
        cfg = self.cfg
        A, N = g.A, g.x.size(0)
        n_ctx, n_t = ctx.numel(), tgt.numel()
        dev = g.x.device

        cnt = torch.zeros(N, device=dev)
        mean_r = torch.zeros(N, device=dev)
        if cfg.context_features and n_ctx:
            Qc = g.Q[ctx]
            cnt = Qc.sum(0)
            mean_r = (Qc.T @ self._rate_z(y_ctx, g.hops[ctx])) / cnt.clamp(min=1)
        h = self.gnn(torch.cat([g.x, torch.log1p(cnt).unsqueeze(-1), mean_r.unsqueeze(-1)], -1), g.edge_index)

        cov_e = A[ctx].sum(0) if n_ctx else torch.zeros(A.size(1), device=dev)
        hu, hv = h[g.edge_uv[:, 0]], h[g.edge_uv[:, 1]]
        e_in = torch.cat([hu + hv, (hu - hv).abs(), g.edge_feat, torch.log1p(cov_e).unsqueeze(-1)], -1)
        x_e = F.softplus(self.edge_mlp(e_in).squeeze(-1)) * self.prior_edge_ms               # ms per edge
        b = F.softplus(self.bias_raw) * self.prior_edge_ms

        Z = self.path_enc(h, g.node_pad, g.node_mask)
        Zt = Z[tgt]
        c = torch.zeros(n_t, cfg.d, device=dev)
        if self.temporal is not None:
            r_ctx = self.z_of(y_ctx) if n_ctx else torch.zeros(0, device=dev)
            c = self.temporal(Z, g.node_pad, g.node_mask, A, N, ctx, tgt, g.t[ctx], g.t[tgt], r_ctx)

        y_add = A[tgt] @ x_e + b
        base = torch.log1p(y_add) if cfg.tomography else self.log_rtt_mean.expand(n_t)
        y_g = torch.expm1((base + self.log_rtt_std * self.res_head(torch.cat([Zt, c], -1)).squeeze(-1)).clamp(max=12.0))

        cov_p = ((A[tgt] * (cov_e > 0).float()).sum(1) / g.hops[tgt]) if n_ctx else torch.zeros(n_t, device=dev)
        lam = torch.zeros(n_t, device=dev)
        y_c = y_g
        if cfg.convex and cfg.tomography and n_ctx:
            Ac = torch.cat([A[ctx], torch.ones(n_ctx, 1, device=dev)], 1)
            ctx_rate = (y_ctx / g.hops[ctx].clamp(min=1.0)).median().clamp(min=0.1)
            scale = (ctx_rate / self.prior_edge_ms).clamp(0.2, 5.0)
            x0 = torch.cat([x_e * scale, b.view(1)])
            alpha = F.softplus(self.alpha_raw) + 1e-3
            resid = y_ctx - Ac @ x0
            K = Ac @ Ac.T + alpha * torch.eye(n_ctx, device=dev)
            x_ref = (x0 + Ac.T @ torch.linalg.solve(K, resid)).clamp(min=0)                # min ||Ax-y||^2 + alpha ||x-x0||^2
            y_c = torch.cat([A[tgt], torch.ones(n_t, 1, device=dev)], 1) @ x_ref
            rel = (resid.abs() / y_ctx.clamp(min=1)).mean()
            gate_in = torch.stack([cov_p, torch.full((n_t,), math.log1p(n_ctx) / 5, device=dev), rel.expand(n_t)], -1)
            g_res = torch.sigmoid(self.gate(gate_in).squeeze(-1))
            lam = 1.0 - g_res
            y_hat = y_c + g_res * (y_g - y_c)
        else:
            y_hat = y_g

        out = {'y_hat': y_hat, 'mu_z': self.z_of(y_hat.clamp(min=0)), 'y_gnn': y_g, 'y_convex': y_c, 'lam': lam,
               'cov': cov_p, 'y_add_ctx': (A[ctx] @ x_e + b) if n_ctx else torch.zeros(0, device=dev), 'logvar': None,
               'loss_logit': None}
        if self.var_head is not None:
            out['logvar'] = self.var_head(torch.cat([Zt, c, cov_p.unsqueeze(-1)], -1)).squeeze(-1).clamp(-6, 3)
        if self.loss_head is not None:
            out['loss_logit'] = self.loss_head(torch.cat([Zt, c], -1)).squeeze(-1)
        return out
