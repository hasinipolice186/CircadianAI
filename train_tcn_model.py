"""
train_tcn_model.py — Training script for CircadianTCN v3
======================================================
Multi-task learning across 5 heads:
  1. Sleep duration regression     (Huber loss)
  2. Insomnia binary classification (Focal BCE)
  3. Recovery days regression      (Huber loss)
  4. 7-day insomnia trajectory     (MSE per-day sequence)
  5. Adaptation strategy           (CrossEntropy, 5 classes)

Usage:
    python train_tcn_model.py

Outputs:
    tcn_model.pth   — full checkpoint
    tcn_edge.pt     — TorchScript edge model
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score, accuracy_score

from tcn_model import CircadianTCN, EdgeTCN
from preprocess import run_pipeline, NUM_FEATURES, SEQ_LEN, STRATEGY_LABELS


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "num_features":  NUM_FEATURES,   # 19
    "seq_len":       SEQ_LEN,           # 7
    "channels":      [64, 128, 128, 64],
    "kernel_size":   3,
    "dropout":       0.2,
    "lr":            3e-4,
    "weight_decay":  1e-4,
    "epochs":        100,
    "batch_size":    64,
    "patience":      15,
    "save_dir":      ".",
    "seed":          42,
    # Task loss weights
    "w_duration":    0.20,
    "w_insomnia":    0.25,
    "w_recovery":    0.20,
    "w_trajectory":  0.20,
    "w_strategy":    0.15,
}


# ── Dataset ────────────────────────────────────────────────────────────────────

class SleepDataset(Dataset):
    def __init__(self, X, y_dur, y_ins, y_rec, y_ins7d, y_strat):
        self.X      = torch.tensor(X,      dtype=torch.float32)
        self.y_dur  = torch.tensor(y_dur,  dtype=torch.float32)
        self.y_ins  = torch.tensor(y_ins,  dtype=torch.float32)
        self.y_rec  = torch.tensor(y_rec,  dtype=torch.float32)
        self.y_ins7d= torch.tensor(y_ins7d,dtype=torch.float32)
        self.y_strat= torch.tensor(y_strat,dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y_dur[idx], self.y_ins[idx],
                self.y_rec[idx], self.y_ins7d[idx], self.y_strat[idx])


def make_loader(X, y_dur, y_ins, y_rec, y_ins7d, y_strat,
                batch_size, shuffle=True, balance=False):
    ds = SleepDataset(X, y_dur, y_ins, y_rec, y_ins7d, y_strat)
    sampler = None
    if balance and shuffle:
        counts  = np.bincount(y_ins.astype(int))
        weights = 1.0 / counts[y_ins.astype(int)]
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, num_workers=0)


# ── Multi-task loss ───────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Weighted combination of all 5 task losses.
    """
    def __init__(self, cfg):
        super().__init__()
        self.w_dur   = cfg["w_duration"]
        self.w_ins   = cfg["w_insomnia"]
        self.w_rec   = cfg["w_recovery"]
        self.w_traj  = cfg["w_trajectory"]
        self.w_strat = cfg["w_strategy"]
        self.huber   = nn.HuberLoss(delta=1.0)
        self.ce      = nn.CrossEntropyLoss()

    def focal_bce(self, pred, target, gamma=2.0):
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt  = torch.exp(-bce)
        return ((1 - pt) ** gamma * bce).mean()

    def forward(self, dur_pred, ins_pred, rec_pred, traj_pred, strat_pred,
                dur_true, ins_true, rec_true, traj_true, strat_true):

        # Duration: Huber on normalised values
        l_dur  = self.huber(dur_pred / 12.0, dur_true / 12.0)
        # Insomnia: focal BCE
        l_ins  = self.focal_bce(ins_pred, ins_true)
        # Recovery: Huber on normalised values
        l_rec  = self.huber(rec_pred / 21.0, rec_true / 21.0)
        # Trajectory: MSE across 7 forecast days
        l_traj = F.mse_loss(traj_pred, traj_true)
        # Strategy: cross-entropy
        l_strat = self.ce(strat_pred, strat_true)

        total = (self.w_dur  * l_dur +
                 self.w_ins  * l_ins +
                 self.w_rec  * l_rec +
                 self.w_traj * l_traj +
                 self.w_strat * l_strat)

        return total, l_dur, l_ins, l_rec, l_traj, l_strat


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    totals = {"loss":0,"dur":0,"ins":0,"rec":0,"traj":0,"strat":0}
    for X, yd, yi, yr, yi7, ys in loader:
        X, yd, yi, yr, yi7, ys = (
            X.to(device), yd.to(device), yi.to(device),
            yr.to(device), yi7.to(device), ys.to(device)
        )
        optimizer.zero_grad()
        dur_p, ins_p, rec_p, traj_p, strat_p = model(X)
        loss, ld, li, lr_, lt, ls = criterion(
            dur_p, ins_p, rec_p, traj_p, strat_p,
            yd, yi, yr, yi7, ys
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["loss"] += loss.item()
        totals["dur"]  += ld.item()
        totals["ins"]  += li.item()
        totals["rec"]  += lr_.item()
        totals["traj"] += lt.item()
        totals["strat"]+= ls.item()
    n = len(loader)
    return {k: v/n for k,v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_dur_p, all_dur_t = [], []
    all_ins_p, all_ins_t = [], []
    all_rec_p, all_rec_t = [], []
    all_traj_p, all_traj_t = [], []
    all_strat_p, all_strat_t = [], []

    for X, yd, yi, yr, yi7, ys in loader:
        X, yd, yi, yr, yi7, ys = (
            X.to(device), yd.to(device), yi.to(device),
            yr.to(device), yi7.to(device), ys.to(device)
        )
        dur_p, ins_p, rec_p, traj_p, strat_p = model(X)
        loss, *_ = criterion(dur_p, ins_p, rec_p, traj_p, strat_p, yd, yi, yr, yi7, ys)
        total_loss += loss.item()

        all_dur_p.extend(dur_p.cpu().numpy())
        all_dur_t.extend(yd.cpu().numpy())
        all_ins_p.extend(ins_p.cpu().numpy())
        all_ins_t.extend(yi.cpu().numpy())
        all_rec_p.extend(rec_p.cpu().numpy())
        all_rec_t.extend(yr.cpu().numpy())
        all_traj_p.append(traj_p.cpu().numpy())
        all_traj_t.append(yi7.cpu().numpy())
        all_strat_p.extend(strat_p.argmax(dim=1).cpu().numpy())
        all_strat_t.extend(ys.cpu().numpy())

    dur_mae  = mean_absolute_error(all_dur_t, all_dur_p)
    rec_mae  = mean_absolute_error(all_rec_t, all_rec_p)
    try:
        ins_auc = roc_auc_score(all_ins_t, all_ins_p)
    except Exception:
        ins_auc = float("nan")

    traj_mse = float(np.mean(
        (np.vstack(all_traj_p) - np.vstack(all_traj_t)) ** 2
    ))
    strat_acc = accuracy_score(all_strat_t, all_strat_p)

    return {
        "loss":      total_loss / len(loader),
        "dur_mae":   dur_mae,
        "rec_mae":   rec_mae,
        "ins_auc":   ins_auc,
        "traj_mse":  traj_mse,
        "strat_acc": strat_acc,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg=CONFIG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Features: {cfg['num_features']}")

    save_dir    = cfg["save_dir"]
    splits_path = os.path.join(save_dir, "splits.npz")

    # Check if existing splits.npz is v3-compatible (has new labels)
    splits_valid = False
    if os.path.exists(splits_path):
        with np.load(splits_path) as d_check:
            keys = list(d_check.keys())
        if "y_rec_train" in keys and "y_ins7d_train" in keys and "y_strat_train" in keys:
            splits_valid = True
        else:
            print("Found old splits.npz (v2 format — missing simulation labels). Regenerating …")
            os.remove(splits_path)

    if splits_valid:
        print("Loading existing splits.npz …")
        d = np.load(splits_path)
        (X_tr, y_dur_tr, y_ins_tr, y_rec_tr, y_ins7d_tr, y_strat_tr) = (
            d["X_train"], d["y_dur_train"], d["y_ins_train"],
            d["y_rec_train"], d["y_ins7d_train"], d["y_strat_train"]
        )
        (X_va, y_dur_va, y_ins_va, y_rec_va, y_ins7d_va, y_strat_va) = (
            d["X_val"], d["y_dur_val"], d["y_ins_val"],
            d["y_rec_val"], d["y_ins7d_val"], d["y_strat_val"]
        )
    else:
        result = run_pipeline(save_dir=save_dir)
        (X_tr, y_dur_tr, y_ins_tr, y_rec_tr, y_ins7d_tr, y_strat_tr,
         X_va, y_dur_va, y_ins_va, y_rec_va, y_ins7d_va, y_strat_va) = result

    tr_loader = make_loader(X_tr, y_dur_tr, y_ins_tr, y_rec_tr, y_ins7d_tr,
                            y_strat_tr, cfg["batch_size"], shuffle=True, balance=True)
    va_loader = make_loader(X_va, y_dur_va, y_ins_va, y_rec_va, y_ins7d_va,
                            y_strat_va, cfg["batch_size"], shuffle=False)

    model = CircadianTCN(
        num_features  = cfg["num_features"],
        seq_len       = cfg["seq_len"],
        channels      = cfg["channels"],
        kernel_size   = cfg["kernel_size"],
        dropout       = cfg["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )
    criterion = MultiTaskLoss(cfg)

    best_val_loss  = float("inf")
    patience_count = 0

    header = (f"{'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  "
              f"{'DurMAE':>7}  {'RecMAE':>7}  {'InsAUC':>7}  "
              f"{'TrajMSE':>8}  {'StratAcc':>9}")
    print(f"\n{header}")
    print("─" * len(header))

    for epoch in range(1, cfg["epochs"] + 1):
        t0    = time.time()
        tr    = train_epoch(model, tr_loader, optimizer, criterion, device)
        va    = evaluate(model, va_loader, criterion, device)
        scheduler.step()

        print(
            f"{epoch:>4}  {tr['loss']:>8.4f}  {va['loss']:>8.4f}  "
            f"{va['dur_mae']:>7.3f}  {va['rec_mae']:>7.2f}  {va['ins_auc']:>7.3f}  "
            f"{va['traj_mse']:>8.4f}  {va['strat_acc']:>9.3f}  "
            f"({time.time()-t0:.1f}s)"
        )

        if va["loss"] < best_val_loss:
            best_val_loss  = va["loss"]
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": va["loss"],
                "val_metrics": va,
                "config": cfg,
            }, os.path.join(save_dir, "tcn_model.pth"))
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")

    # ── Export edge model ────────────────────────────────────────────────────
    print("\nExporting TorchScript edge model (tcn_edge.pt) …")
    edge = EdgeTCN(num_features=NUM_FEATURES, seq_len=SEQ_LEN).to("cpu")
    edge.eval()
    dummy = torch.randn(1, SEQ_LEN, NUM_FEATURES)
    with torch.no_grad():
        traced = torch.jit.trace(edge, dummy)
        traced(dummy)
    traced.save(os.path.join(save_dir, "tcn_edge.pt"))
    print("  Saved tcn_edge.pt")

    # ── Test evaluation ──────────────────────────────────────────────────────
    splits_path = os.path.join(save_dir, "splits.npz")
    if os.path.exists(splits_path):
        d = np.load(splits_path)
        if "X_test" in d:
            te_loader = make_loader(
                d["X_test"], d["y_dur_test"], d["y_ins_test"],
                d["y_rec_test"], d["y_ins7d_test"], d["y_strat_test"],
                cfg["batch_size"], shuffle=False
            )
            ckpt = torch.load(os.path.join(save_dir, "tcn_model.pth"), map_location=device)
            model.load_state_dict(ckpt["model_state"])
            te = evaluate(model, te_loader, criterion, device)
            print("\n── Test Results ────────────────────────────")
            print(f"  Sleep Duration MAE    : {te['dur_mae']:.3f} hrs")
            print(f"  Recovery Days MAE     : {te['rec_mae']:.2f} days")
            print(f"  Insomnia AUC          : {te['ins_auc']:.3f}")
            print(f"  Trajectory MSE        : {te['traj_mse']:.4f}")
            print(f"  Strategy Accuracy     : {te['strat_acc']:.3f}")

    print("\nTraining complete. Files saved:")
    for f in ["tcn_model.pth", "tcn_edge.pt", "final_tcn_dataset.csv", "splits.npz"]:
        p = os.path.join(save_dir, f)
        if os.path.exists(p):
            print(f"  ✓ {f}")


if __name__ == "__main__":
    train()