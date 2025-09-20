import os
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# =============================
# Columns & device
# =============================
FEATURES = ["time"] + [chr(c) for c in range(ord("A"), ord("N") + 1)]  # time + A..N (15 total)
STATE_COLS = [c for c in FEATURES if c != "time"]  # A..N (14 dims)
TARGETS = ["Y1", "Y2"]  # optional supervised head (kept for future use)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================
# Data loading
# =============================
def load_train_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Make sure required cols exist
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")
    # Coerce numeric
    for c in FEATURES + TARGETS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows with NaNs in features, sort by time
    df = df.dropna(subset=FEATURES).sort_values("time").reset_index(drop=True)
    return df


# =============================
# Training dataset (pairwise)
# =============================
class PairDataset(Dataset):
    """
    Builds (x_t, t_t, x_{t+1}, t_{t+1}, dt) pairs for training reconstruction.
    Uses a provided (mean,std) scaler for states. If None, fits it on x0.
    """
    def __init__(self, df: pd.DataFrame, scaler_state: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        t = df["time"].values.astype(np.float32)
        x = df[STATE_COLS].values.astype(np.float32)

        dt = np.diff(t)
        valid = dt > 0
        idx = np.where(valid)[0]  # pairs i -> i+1

        self.t0 = t[idx]
        self.t1 = t[idx + 1]
        self.dt = dt[idx]
        self.x0_raw = x[idx]
        self.x1_raw = x[idx + 1]

        if scaler_state is None:
            mean = self.x0_raw.mean(axis=0, keepdims=True)
            std = self.x0_raw.std(axis=0, keepdims=True) + 1e-8
        else:
            mean, std = scaler_state

        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.x0 = (self.x0_raw - self.mean) / self.std
        self.x1 = (self.x1_raw - self.mean) / self.std

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x0[i]),
            torch.tensor(self.t0[i]),
            torch.from_numpy(self.x1[i]),
            torch.tensor(self.t1[i]),
            torch.tensor(self.dt[i]),
        )


# =============================
# Validation dataset (rollout)
# =============================
class RolloutDataset(Dataset):
    """
    For validation: start states and ground-truth future windows to evaluate multi-step prediction.
    Given a horizon H, returns (x_start, t_start, dt_seq[H], x_future_seq[H]).
    All states are scaled with (mean,std) from training.
    """
    def __init__(self, df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, horizon: int = 20):
        t = df["time"].values.astype(np.float32)
        x = df[STATE_COLS].values.astype(np.float32)

        # indices such that i+horizon exists
        max_start = len(t) - (horizon + 1)
        self.starts = np.arange(max(0, max_start))
        self.horizon = horizon

        self.t = t
        self.x = (x - mean) / std
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        i = self.starts[k]
        j = i + 1
        h = self.horizon

        x_start = torch.from_numpy(self.x[i:i+1])          # [1, D]
        t_start = torch.tensor(self.t[i])                  # scalar
        # dt sequence length h: dt_m = t_{i+m+1}-t_{i+m}
        dt_seq = torch.from_numpy(np.diff(self.t[i:i+h+1]).astype(np.float32))  # [h]
        x_future = torch.from_numpy(self.x[j:j+h+1])       # states from i+1 ... i+h  -> shape [h, D]

        return x_start.squeeze(0), t_start, dt_seq, x_future   # [D], scalar, [h], [h, D]


# =============================
# Dynamics model f(x,t) = dx/dt
# =============================
class DynamicsNet(nn.Module):
    def __init__(self, x_dim: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = x_dim + 1  # concat time
        layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, D]; t: [B] or scalar
        if t.dim() == 0:
            t = t.expand(x.size(0))
        t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=1))


# =============================
# RK4 integrator
# =============================
def rk4_step(x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, f: DynamicsNet) -> torch.Tensor:
    if dt.dim() == 0:
        dt_b = dt.view(1, 1).expand(x.size(0), 1)
    else:
        dt_b = dt.view(-1, 1)
    if t.dim() == 0:
        t_b = t.view(1).expand(x.size(0))
    else:
        t_b = t

    k1 = f(x, t_b)
    k2 = f(x + 0.5 * dt_b * k1, t_b + 0.5 * dt_b.squeeze(1))
    k3 = f(x + 0.5 * dt_b * k2, t_b + 0.5 * dt_b.squeeze(1))
    k4 = f(x + dt_b * k3,       t_b + dt_b.squeeze(1))
    return x + (dt_b / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# =============================
# Training loop
# =============================
def train_loop(
    model_f: DynamicsNet,
    train_ds: PairDataset,
    val_ds: RolloutDataset,
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    w_forward: float = 1.0,
    w_backward: float = 1.0,
    w_cycle: float = 0.5,
    patience: int = 30,
):
    model_f.to(device)

    opt = torch.optim.AdamW(model_f.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6, verbose=True)

    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False)  # per-trajectory rollout

    best_val = float("inf")
    patience_ct = 0

    for ep in range(1, epochs + 1):
        # -------- Train (reconstruction-based) --------
        model_f.train()
        tr_sum = 0.0
        n_tr = 0

        for x0, t0, x1, t1, dt in dl_tr:
            x0, x1 = x0.to(device), x1.to(device)
            t0, t1, dt = t0.to(device), t1.to(device), dt.to(device)

            opt.zero_grad()

            # forward one step
            x1_hat = rk4_step(x0, t0, dt, model_f)
            # backward one step
            x0_hat = rk4_step(x1, t1, -dt, model_f)
            # cycle (forward then backward)
            with torch.no_grad():
                x1_hat_det = x1_hat.detach()
            x0_cyc = rk4_step(x1_hat_det, t1, -dt, model_f)

            l_fwd = F.mse_loss(x1_hat, x1)
            l_bwd = F.mse_loss(x0_hat, x0)
            l_cyc = F.mse_loss(x0_cyc, x0)

            loss = w_forward * l_fwd + w_backward * l_bwd + w_cycle * l_cyc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_f.parameters(), 1.0)
            opt.step()

            bs = x0.size(0)
            tr_sum += loss.item() * bs
            n_tr += bs

        tr_loss = tr_sum / max(1, n_tr)

        # -------- Validation (future prediction ONLY) --------
        model_f.eval()
        va_sum = 0.0
        n_va = 0
        with torch.no_grad():
            for x_start, t_start, dt_seq, x_future in dl_va:
                # Shapes: x_start [D]; dt_seq [H]; x_future [H, D]
                x = x_start.to(device).unsqueeze(0)  # [1, D]
                t = t_start.to(device)
                dt_seq = dt_seq.squeeze(0).to(device)  # [H]
                gt = x_future.squeeze(0).to(device)    # [H, D]

                preds = []
                for h in range(dt_seq.shape[0]):
                    dt_h = dt_seq[h]
                    x = rk4_step(x, t, dt_h, model_f)  # 1 step
                    t = t + dt_h
                    preds.append(x.squeeze(0))
                preds = torch.stack(preds, dim=0)  # [H, D]

                va_sum += F.mse_loss(preds, gt).item() * gt.shape[0]
                n_va += gt.shape[0]

        va_loss = va_sum / max(1, n_va)
        sched.step(va_loss)

        print(f"Epoch {ep:03d} | train={tr_loss:.6f} | val_future_pred={va_loss:.6f}")

        # Early stopping based on future prediction error
        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"[info] Early stopping at epoch {ep}. Best future-pred val={best_val:.6f}")
                break


# =============================
# Main (no artifacts saved)
# =============================
def main():
    # ---- edit these ----
    csv_path    = "../research/data/train.csv"   # single CSV; we'll split internally
    seed        = 42
    val_frac    = 0.15
    horizon     = 20   # steps ahead to evaluate during validation

    hidden      = 128
    depth       = 4

    epochs      = 300
    batch_size  = 128
    lr          = 1e-3
    weight_decay= 1e-4

    w_forward   = 1.0
    w_backward  = 1.0
    w_cycle     = 0.5
    patience    = 30
    # --------------------

    set_seed(seed)
    df = load_train_csv(csv_path)

    # Build a global scaler from the *pair* x0 states
    full_pairs = PairDataset(df)  # this fits mean/std on x0
    mean, std = full_pairs.mean, full_pairs.std

    # Split pairs for training/validation indices
    N = len(full_pairs)
    n_val = max(1, int(val_frac * N))
    perm = np.random.permutation(N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # Build train pair dataset (using same scaler stats)
    train_ds = PairDataset(df, scaler_state=(mean, std))
    # Subsample to train split
    def subset_pairs(ds: PairDataset, idxs: np.ndarray) -> PairDataset:
        sub = PairDataset.__new__(PairDataset)
        sub.t0 = ds.t0[idxs]; sub.t1 = ds.t1[idxs]; sub.dt = ds.dt[idxs]
        sub.x0 = ds.x0[idxs]; sub.x1 = ds.x1[idxs]
        sub.x0_raw = ds.x0_raw[idxs]; sub.x1_raw = ds.x1_raw[idxs]
        sub.mean = ds.mean; sub.std = ds.std
        return sub
    train_ds = subset_pairs(train_ds, tr_idx)

    # Build validation rollout dataset from the SAME sequence using same scaler
    val_ds = RolloutDataset(df, mean=mean, std=std, horizon=horizon)

    # Model
    x_dim = len(STATE_COLS)  # 14
    model_f = DynamicsNet(x_dim=x_dim, hidden=hidden, depth=depth)

    # Train
    train_loop(
        model_f,
        train_ds,
        val_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        w_forward=w_forward,
        w_backward=w_backward,
        w_cycle=w_cycle,
        patience=patience,
    )

if __name__ == "__main__":
    main()
