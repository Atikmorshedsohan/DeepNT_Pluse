"""Temporal branch: TGN node memory (GRU), then a GRU + Transformer over the recent measurement history.

Events are the observed (context) path measurements, in time order. Adaptation to this dataset: the
CAIDA traceroutes almost never repeat the same (source, destination) pair, so a path's own history does not
exist. The "history window" is therefore the most recent measurements of *related* paths in the same graph,
each tagged with its time gap and its edge overlap with the path being predicted.

Causality: this module assumes every context event is earlier than every target it serves. The trainer
guarantees that by building context and targets per time block (see train.causal_blocks).
"""
import torch
import torch.nn as nn

from .layers import TimeEncoder


class TemporalBranch(nn.Module):
    def __init__(self, d, d_mem=32, d_time=16, history_k=8, heads=4, dropout=0.1):
        super().__init__()
        self.d_mem, self.k = d_mem, history_k
        self.time = TimeEncoder(d_time)
        self.msg = nn.Sequential(nn.Linear(d_mem + d + d_time + 2, d_mem), nn.ReLU())
        self.mem_gru = nn.GRUCell(d_mem, d_mem)
        self.token = nn.Linear(d + 1 + d_time + 1, d)
        self.null = nn.Parameter(torch.zeros(d))
        self.hist_gru = nn.GRU(d, d, batch_first=True)
        self.query = nn.Linear(d + d_mem, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

    def forward(self, Z, node_pad, node_mask, A, n_nodes, ctx, tgt, t_ctx, t_tgt, r_ctx):
        """Z [M,d] path embeddings; ctx/tgt path indices; t_* hours; r_ctx normalised log-RTT of context paths.
        Returns [n_tgt, d]."""
        n_t = tgt.numel()
        order = torch.argsort(t_ctx)
        ctx, t_ctx, r_ctx = ctx[order], t_ctx[order], r_ctx[order]

        state = torch.zeros(n_nodes, self.d_mem, device=Z.device)
        last = torch.full((n_nodes,), float(t_ctx[0]) if ctx.numel() else 0.0, device=Z.device)
        for j in range(ctx.numel()):                                   # TGN memory update, one event at a time
            p = ctx[j]
            nodes = node_pad[p][node_mask[p]]
            pos = torch.linspace(0, 1, nodes.numel(), device=Z.device).unsqueeze(-1)
            m = self.msg(torch.cat([state[nodes], Z[p].expand(nodes.numel(), -1),
                                    self.time(t_ctx[j] - last[nodes]), r_ctx[j].expand(nodes.numel(), 1), pos], -1))
            state = state.index_put((nodes,), self.mem_gru(m, state[nodes]))
            last = last.index_put((nodes,), t_ctx[j].expand(nodes.numel()))

        m_t = node_mask[tgt].unsqueeze(-1).float()
        mem = (state[node_pad[tgt]] * m_t).sum(1) / m_t.sum(1).clamp(min=1)          # [n_t, d_mem]

        tokens = [self.null.expand(n_t, 1, -1)]
        k = min(self.k, ctx.numel())
        if k:
            rec, t_rec, r_rec = ctx[-k:], t_ctx[-k:], r_ctx[-k:]
            inter = A[tgt] @ A[rec].T
            jac = inter / (A[tgt].sum(1, keepdim=True) + A[rec].sum(1).unsqueeze(0) - inter).clamp(min=1)
            feats = torch.cat([Z[rec].unsqueeze(0).expand(n_t, -1, -1), r_rec.view(1, k, 1).expand(n_t, -1, -1),
                               self.time(t_tgt.view(n_t, 1) - t_rec.view(1, k)), jac.unsqueeze(-1)], -1)
            tokens.append(self.token(feats))
        seq, _ = self.hist_gru(torch.cat(tokens, 1))
        q = self.query(torch.cat([Z[tgt], mem], -1)).unsqueeze(1)
        return self.tf(torch.cat([seq, q], 1))[:, -1]
