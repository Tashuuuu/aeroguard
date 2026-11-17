"""ST-GNN: TCN encoder (per station) -> GAT (across stations) -> MLP quantile head."""
import torch, torch.nn as nn, torch.nn.functional as Fn
from torch_geometric.nn import GATConv


class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):
        return self.conv(Fn.pad(x, (self.pad, 0)))        # pad only on the left => no peeking at the future


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, d, p):
        super().__init__()
        self.c1, self.c2 = CausalConv1d(cin, cout, k, d), CausalConv1d(cout, cout, k, d)
        self.drop = nn.Dropout(p)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = self.drop(Fn.relu(self.c1(x)))
        y = self.drop(Fn.relu(self.c2(y)))
        return Fn.relu(y + self.res(x))


class TCNEncoder(nn.Module):
    def __init__(self, n_feat, hidden, dilations, k, p):
        super().__init__()
        self.net = nn.Sequential(*[TCNBlock(n_feat if i == 0 else hidden, hidden, k, d, p)
                                   for i, d in enumerate(dilations)])

    def forward(self, x):                                   # (B*N, F, T)
        return self.net(x)[:, :, -1]                        # embedding at the last time step


class STGNN(nn.Module):
    def __init__(self, n_feat, n_poll, horizon, n_q, edge_index, edge_weight, hidden=64, heads=4,
                 dilations=(1, 2, 4), kernel=3, dropout=0.1, use_gat=True):
        super().__init__()
        self.P, self.H, self.Q, self.use_gat = n_poll, horizon, n_q, use_gat
        self.tcn = TCNEncoder(n_feat, hidden, list(dilations), kernel, dropout)
        self.register_buffer("ei", torch.as_tensor(edge_index, dtype=torch.long))
        self.register_buffer("ew", torch.as_tensor(edge_weight, dtype=torch.float32).unsqueeze(-1))
        if use_gat:
            self.gat = GATConv(hidden, hidden // heads, heads=heads, edge_dim=1, dropout=dropout)
            self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_poll * horizon * n_q))

    def forward(self, x):                                   # x (B,T,N,F)
        B, T, N, F = x.shape
        h = self.tcn(x.permute(0, 2, 3, 1).reshape(B * N, F, T))          # (B*N, hidden)
        if self.use_gat:
            off = (torch.arange(B, device=x.device) * N).view(B, 1, 1)    # replicate graph B times
            ei = (self.ei.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1)
            ea = self.ew.repeat(B, 1)
            h = self.norm(h + self.gat(h, ei, ea))                        # residual attention over neighbours
        out = self.head(h).view(B, N, self.P, self.H, self.Q).permute(0, 3, 1, 2, 4)   # (B,H,N,P,Q)
        return torch.sort(out, dim=-1).values   