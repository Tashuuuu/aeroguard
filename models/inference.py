"""Prediction helpers, prediction files, and the Runtime class used by anomaly / counterfactual / API."""
import numpy as np, torch
from config import *
from preprocess.dataset import (load_bundle, standardize, Scaler, channel_idx, mask_own)
from models.st_gnn import STGNN


@torch.no_grad()
def predict(model, Xs, origins, window, device, sc, idx, mask_node=None, mask_global=False, post=None, batch=64):
    """Returns RAW-unit quantile forecasts (S,H,N,P,Q) (or post(...) of each batch)."""
    model.eval()
    mean = torch.tensor(sc.mean[:3], device=device).view(1, 1, 1, 3, 1)
    std = torch.tensor(sc.std[:3], device=device).view(1, 1, 1, 3, 1)
    outs = []
    for i in range(0, len(origins), batch):
        o = origins[i:i + batch]
        x = torch.stack([Xs[s - window + 1:s + 1] for s in o]).to(device)
        B, N = x.shape[0], x.shape[2]
        if mask_global:
            x = mask_own(x, torch.ones(B, N, dtype=torch.bool, device=device), idx, mask_nbr=True)
        elif mask_node is not None:
            nm = torch.zeros(B, N, dtype=torch.bool, device=device)
            nm[:, mask_node] = True
            x = mask_own(x, nm, idx)
        y = model(x) * std + mean
        outs.append((post(y) if post else y).cpu())
    return torch.cat(outs).numpy()


def save_preds(path, origins, pred_q, qlevels, leads, med_index=None):
    """pred_q (S,H,N,P,Q) raw. Stores only the evaluation leads."""
    li = [l - 1 for l in leads]
    q = pred_q[:, li]
    mi = med_index if med_index is not None else len(qlevels) // 2
    np.savez_compressed(path, origins=origins, pred=q[..., mi], q=q, qlevels=np.array(qlevels), leads=np.array(leads))


def save_point(path, origins, pred, leads):
    """Point forecasts only (persistence / XGBoost). pred (S,L,N,P) already at the evaluation leads."""
    np.savez_compressed(path, origins=origins, pred=pred, leads=np.array(leads))


class Runtime:
    """Loads trained ST-GNN + data so other modules can forecast with a few lines."""
    def __init__(self, mode, name=None, device=None):
        self.mode, self.cfg = mode, MODES[mode]
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ART / f"{name or 'stgnn_' + mode}.pt", map_location="cpu", weights_only=False)
        self.ck, self.names = ck, ck["names"]
        self.sc = Scaler(ck["mean"], ck["std"])
        self.bundle = load_bundle(mode, drop_satfire=ck["no_satfire"])
        self.idx = channel_idx(self.names)
        g = np.load(PROC / f"graph_{mode}.npz")
        self.model = STGNN(len(self.names), 3, self.cfg["horizon"], len(QUANTILES), g["edge_index"], g["edge_weight"],
                           HP["hidden"], HP["heads"], self.cfg["dilations"], HP["kernel"], HP["dropout"],
                           use_gat=not ck["no_gat"]).to(self.dev)
        self.model.load_state_dict(ck["state"])
        self.window = self.cfg["window"]
        self.Xs = torch.tensor(standardize(self.bundle.X, self.sc))

    def run(self, origins, X_raw=None, **kw):
        Xs = self.Xs if X_raw is None else torch.tensor(standardize(X_raw, self.sc))
        return predict(self.model, Xs, origins, self.window, self.dev, self.sc, self.idx, **kw)