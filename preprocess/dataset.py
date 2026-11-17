"""Tensors, scaling, chronological splits, windows, input masking."""
from dataclasses import dataclass
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from config import *


@dataclass
class Bundle:
    X: np.ndarray            # raw (T,N,F), NaN = missing
    times: pd.DatetimeIndex
    names: list
    st: pd.DataFrame


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray


def load_bundle(mode, drop_satfire=False):
    st = pd.read_csv(PROC / "stations.csv").sort_values("station_id").reset_index(drop=True)
    df = pd.read_parquet(PROC / f"features_{mode}.parquet").sort_values(["time", "station_id"])
    names = [f for f in FEATURES if not (drop_satfire and f in SATFIRE)]
    times = pd.DatetimeIndex(sorted(df["time"].unique()))
    X = df[names].to_numpy(np.float32).reshape(len(times), len(st), len(names))
    return Bundle(X, times, names, st)


def fit_scaler(X, times, names):
    a, b = pd.Timestamp(SPLITS["train"][0]), pd.Timestamp(SPLITS["train"][1]) + pd.Timedelta(days=1)
    sel = (times >= a) & (times < b)
    flat = X[sel].reshape(-1, X.shape[-1])
    mean, std = np.nanmean(flat, 0), np.nanstd(flat, 0)
    for i, n in enumerate(names):
        if n in NO_SCALE:
            mean[i], std[i] = 0.0, 1.0
    mean = np.nan_to_num(mean)
    std = np.where(~np.isfinite(std) | (std < 1e-6), 1.0, std)
    return Scaler(mean.astype(np.float32), std.astype(np.float32))


def standardize(X, sc):
    return np.nan_to_num((X - sc.mean) / sc.std, nan=0.0).astype(np.float32)


def scale_targets(X, sc):
    return ((X[:, :, :3] - sc.mean[:3]) / sc.std[:3]).astype(np.float32)   # keeps NaN (masked in loss)


def split_origins(times, window, horizon, split, stride=1, exclude_events=True):
    a, b = pd.Timestamp(SPLITS[split][0]), pd.Timestamp(SPLITS[split][1]) + pd.Timedelta(days=1)
    s = np.arange(window - 1, len(times) - horizon)
    ok = (times[s] >= a) & (times[s + horizon] < b)
    if exclude_events:
        for lo, hi in EVENTS.values():
            lo, hi = pd.Timestamp(lo), pd.Timestamp(hi) + pd.Timedelta(days=1)
            ok &= ~((times[s + horizon] >= lo) & (times[s - window + 1] < hi))
    return s[ok][::stride]


def event_origins(times, window, horizon, event):
    lo, hi = [pd.Timestamp(x) for x in EVENTS[event]]
    s = np.arange(window - 1, len(times) - horizon)
    t1 = times[s + 1]
    return s[(t1 >= lo) & (t1 < hi + pd.Timedelta(days=1))]


class Windows(Dataset):
    def __init__(self, Xs, Ys, origins, window, horizon):
        self.X, self.Y, self.o, self.w, self.h = Xs, Ys, origins, window, horizon

    def __len__(self):
        return len(self.o)

    def __getitem__(self, i):
        s = self.o[i]
        return self.X[s - self.w + 1: s + 1], self.Y[s + 1: s + 1 + self.h]    # (W,N,F), (H,N,P)


# ---------------- channel masking (used for training augmentation, anomaly layer, counterfactuals) ----------------
def channel_idx(names):
    g = lambda L: [names.index(n) for n in L if n in names]
    return dict(own_val=g(OWN_VAL), own_miss=g(OWN_MISS), nbr=g(NBR))


def mask_channels(x, node_mask, chans, fill):
    """x (B,T,N,F); node_mask (B,N) bool. Sets chosen channels of masked nodes to `fill` (all timesteps)."""
    if not chans:
        return x
    m = node_mask[:, None, :, None]
    x = x.clone()
    x[..., chans] = x[..., chans].masked_fill(m, fill)
    return x


def mask_own(x, node_mask, idx, mask_nbr=False):
    x = mask_channels(x, node_mask, idx["own_val"], 0.0)
    x = mask_channels(x, node_mask, idx["own_miss"], 1.0)
    return mask_channels(x, node_mask, idx["nbr"], 0.0) if mask_nbr else x


def sample_train_mask(B, N, device):
    r = torch.rand(B, device=device)
    glob = r < HP["p_global_mask"]
    node = (r >= HP["p_global_mask"]) & (r < HP["p_global_mask"] + HP["p_node_mask"])
    nm = ((torch.rand(B, N, device=device) < HP["node_mask_frac"]) & node[:, None]) | glob[:, None]
    return nm, glob