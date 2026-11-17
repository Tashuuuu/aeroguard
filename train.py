"""Train ST-GNN with pinball loss. Flags: --no_gat, --no_satfire for the ablations."""
import argparse, json, os, time
import numpy as np, torch
from torch.utils.data import DataLoader
from config import *
from preprocess.dataset import *
from models.losses import pinball_loss
from models.st_gnn import STGNN
from models.inference import predict, save_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hourly"); ap.add_argument("--name", default=None)
    ap.add_argument("--no_gat", action="store_true"); ap.add_argument("--no_satfire", action="store_true")
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    a = ap.parse_args()
    name, cfg = a.name or f"stgnn_{a.mode}", MODES[a.mode]
    torch.manual_seed(HP["seed"]); np.random.seed(HP["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.init(project="aeroguard", name=name, config={**HP, **cfg, **vars(a)})

    B = load_bundle(a.mode, drop_satfire=a.no_satfire)
    sc = fit_scaler(B.X, B.times, B.names)
    Xs, Ys = torch.tensor(standardize(B.X, sc)), torch.tensor(scale_targets(B.X, sc))
    W, H, N = cfg["window"], cfg["horizon"], B.X.shape[1]
    o_tr = split_origins(B.times, W, H, "train")
    o_va = split_origins(B.times, W, H, "val", stride=cfg["eval_stride"])
    o_te = split_origins(B.times, W, H, "test", stride=cfg["eval_stride"])
    print(f"origins train {len(o_tr)} | val {len(o_va)} | test {len(o_te)} | features {len(B.names)} | stations {N}")
    dl_tr = DataLoader(Windows(Xs, Ys, o_tr, W, H), batch_size=cfg["batch"], shuffle=True, num_workers=2,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(Windows(Xs, Ys, o_va, W, H), batch_size=cfg["batch"] * 2, num_workers=2)

    g = np.load(PROC / f"graph_{a.mode}.npz")
    model = STGNN(len(B.names), 3, H, len(QUANTILES), g["edge_index"], g["edge_weight"], HP["hidden"], HP["heads"],
                  cfg["dilations"], HP["kernel"], HP["dropout"], use_gat=not a.no_gat).to(dev)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    idx = channel_idx(B.names)
    best, bad = 1e9, 0

    for ep in range(a.epochs):
        model.train(); t0, run = time.time(), 0.0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            nm, gm = sample_train_mask(x.shape[0], N, dev)             # random hiding of own readings
            x = mask_own(x, nm, idx)
            x = mask_channels(x, gm[:, None].expand(-1, N), idx["nbr"], 0.0)
            m = ~torch.isnan(y)
            loss = pinball_loss(model(x), torch.nan_to_num(y), QUANTILES, m)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            run += loss.item()
        model.eval(); vl, n = 0.0, 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                vl += pinball_loss(model(x), torch.nan_to_num(y), QUANTILES, ~torch.isnan(y)).item(); n += 1
        vl /= max(n, 1); sched.step(vl)
        print(f"ep {ep:02d} train {run / len(dl_tr):.4f} val {vl:.4f} ({time.time() - t0:.0f}s)")
        if use_wandb:
            wandb.log({"epoch": ep, "train_pinball": run / len(dl_tr), "val_pinball": vl})
        if vl < best:
            best, bad = vl, 0
            torch.save(dict(state=model.state_dict(), names=B.names, mean=sc.mean, std=sc.std, mode=a.mode,
                            no_gat=a.no_gat, no_satfire=a.no_satfire), ART / f"{name}.pt")
        else:
            bad += 1
            if bad >= HP["patience"]:
                print("early stop"); break

    model.load_state_dict(torch.load(ART / f"{name}.pt", map_location=dev, weights_only=False)["state"])
    pred = predict(model, Xs, o_te, W, dev, sc, idx)
    save_preds(PRED / f"{a.mode}_{name}.npz", o_te, pred, QUANTILES, cfg["leads"])
    print("saved test predictions ->", PRED / f"{a.mode}_{name}.npz")


if __name__ == "__main__":
    main()