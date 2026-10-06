"""Pure-PyTorch building blocks (no torch_geometric dependency)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_neighbors(h, edge_index):
    src, dst = edge_index
    out = torch.zeros_like(h).index_add_(0, dst, h[src])
    deg = torch.zeros(h.size(0), device=h.device).index_add_(0, dst, torch.ones(src.numel(), device=h.device))
    return out / deg.clamp(min=1).unsqueeze(-1)


class SplitWeightConv(nn.Module):
    """h_v' = W_s h_v + W_n mean_{u in N(v)} h_u  (separate self and neighbour weights, the DeepNT layer)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_self = nn.Linear(d_in, d_out)
        self.w_nbr = nn.Linear(d_in, d_out, bias=False)

    def forward(self, h, edge_index):
        return self.w_self(h) + self.w_nbr(mean_neighbors(h, edge_index))


class SAGEConv(nn.Module):
    """GraphSAGE mean aggregator: W [h_v || mean_N(v) h_u] (same function class as the split-weight layer)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(2 * d_in, d_out)

    def forward(self, h, edge_index):
        return self.lin(torch.cat([h, mean_neighbors(h, edge_index)], dim=-1))


class GATConv(nn.Module):
    def __init__(self, d_in, d_out, heads=4):
        super().__init__()
        assert d_out % heads == 0
        self.h, self.c = heads, d_out // heads
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.att_src = nn.Parameter(torch.randn(heads, self.c) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(heads, self.c) * 0.1)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, h, edge_index):
        n = h.size(0)
        loops = torch.arange(n, device=h.device)
        src = torch.cat([edge_index[0], loops])
        dst = torch.cat([edge_index[1], loops])
        z = self.lin(h).view(n, self.h, self.c)
        e = F.leaky_relu((z * self.att_src).sum(-1)[src] + (z * self.att_dst).sum(-1)[dst], 0.2)
        emax = torch.full((n, self.h), float('-inf'), device=h.device).scatter_reduce(
            0, dst.unsqueeze(-1).expand(-1, self.h), e, 'amax', include_self=True)
        ex = (e - emax[dst]).exp()
        denom = torch.zeros(n, self.h, device=h.device).index_add_(0, dst, ex)
        alpha = ex / denom[dst].clamp(min=1e-16)
        out = torch.zeros(n, self.h, self.c, device=h.device).index_add_(0, dst, alpha.unsqueeze(-1) * z[src])
        return out.reshape(n, -1) + self.bias


class GNN(nn.Module):
    def __init__(self, kind, d_in, d, layers, dropout, heads=4):
        super().__init__()
        make = {'split': SplitWeightConv, 'sage': SAGEConv,
                'gat': lambda a, b: GATConv(a, b, heads)}[kind]
        self.convs = nn.ModuleList([make(d_in if i == 0 else d, d) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            out = self.drop(F.relu(norm(conv(h, edge_index))))
            h = out + h if h.size(-1) == out.size(-1) else out          # residual once the width matches
        return h


class TimeEncoder(nn.Module):
    """phi(dt) = cos(w * dt + b) with learned frequencies (dt in hours), initialised from 1/h down to 1/1000h."""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(1.0 / 10 ** torch.linspace(0, 3, d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, dt):
        return torch.cos(dt.unsqueeze(-1) * self.w + self.b)


class PathEncoder(nn.Module):
    """Ordered node embeddings of a path -> one path embedding (positional encoding + self-attention + mean pool)."""

    def __init__(self, d, heads=4, layers=1, dropout=0.1, max_len=64):
        super().__init__()
        self.max_len = max_len
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, h, node_pad, node_mask):
        pos = torch.arange(node_pad.size(1), device=h.device).clamp(max=self.max_len - 1)
        x = self.enc(h[node_pad] + self.pos(pos).unsqueeze(0), src_key_padding_mask=~node_mask)
        m = node_mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1)
