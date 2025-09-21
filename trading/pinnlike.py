# time_pinn_trainer.py
# ------------------------------------------------------------
# - Expects a single CSV with columns: time, A..N, Y1, Y2
# - Sorts by time and forms (x_t, t_t) -> (x_{t+1}, t_{t+1}) pairs
# - Trains a dynamics net f(x,t) using RK4 with reconstruction losses:
#     forward:    x_{t+1} ≈ RK4(x_t, t_t,  +dt)
#     backward:   x_t     ≈ RK4(x_{t+1}, t_{t+1}, -dt)
#     cycle:      x_t     ≈ RK4(RK4(x_t, t_t, +dt), t_{t+1}, -dt)
# - Validation ONLY checks multi-step future prediction error (no reconstruction)
# - No artifacts (models/plots) are saved
# ------------------------------------------------------------

import os
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm






# =============================
# Columns & device
# =============================
FEATURES = ["time"] + [chr(c) for c in range(ord("A"), ord("N") + 1)]  # time + A..N (15 total)
STATE_COLS = [c for c in FEATURES if c != "time"]                       # A..N (14 dims)
TARGETS = ["Y1", "Y2"]  # present in CSV but not required for training here

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

    # Ensure required columns exist (Y columns may be present but are not used)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}\nFound: {list(df.columns)}")

    # Coerce numeric for features we use
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaNs in features and sort by time
    df = df.dropna(subset=FEATURES).sort_values("time").reset_index(drop=True)
    return df


# =============================
# Training dataset (pairwise)
# =============================
class PairDataset(Dataset):
    """
    Builds (x_t, t_t, x_{t+1}, t_{t+1}, dt) pairs for training reconstruction.
    Scales states with (mean,std) computed on x0 unless provided.
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

        # Build/Apply scaler on x0
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
            torch.from_numpy(self.x0[i]),            # [D]
            torch.tensor(self.t0[i]),                # scalar
            torch.from_numpy(self.x1[i]),            # [D]
            torch.tensor(self.t1[i]),                # scalar
            torch.tensor(self.dt[i]),               # scalar
        )


# =============================
# Validation dataset (rollout)
# =============================
class RolloutDataset(Dataset):
    """
    For validation: start states and ground-truth future windows to evaluate multi-step prediction.
    Returns (x_start, t_start, dt_seq[H], x_future_seq[H]) where H=horizon.
    All states are scaled with (mean,std) from training.
    """
    def __init__(self, df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, horizon: int = 20):
        t = df["time"].values.astype(np.float32)
        x = df[STATE_COLS].values.astype(np.float32)

        self.horizon = horizon
        self.t = t
        self.x = (x - mean) / std
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        # valid start indices s.t. s+horizon exists (need indices 0..horizon inclusive)
        max_start = len(t) - (horizon + 1)
        self.starts = np.arange(max(0, max_start))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        i = self.starts[k]
        j = i + 1
        h = self.horizon

        # Start state/time
        x_start = torch.from_numpy(self.x[i]).to(torch.float32)        # [D]
        t_start = torch.tensor(self.t[i], dtype=torch.float32)         # scalar

        # dt sequence: length h  (t_{i+1}-t_i, ..., t_{i+h}-t_{i+h-1})
        dt_seq = torch.from_numpy(
            np.diff(self.t[i:i + h + 1]).astype(np.float32)
        )  # [h]

        # Future states: EXACTLY h steps: x_{i+1} ... x_{i+h}
        x_future = torch.from_numpy(self.x[j:j + h]).to(torch.float32)  # [h, D]

        return x_start, t_start, dt_seq, x_future


# =============================
# Dynamics model f(x,t) = dx/dt
# =============================
class DynamicsNet(nn.Module):
    def __init__(self, x_dim: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = x_dim + 1  # concat time scalar
        layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B,D]
        t: scalar or [B] or [B,1] or [B,1,1] -> normalize to [B,1]
        """
        if t.dim() == 0:
            t = t.expand(x.size(0))
        elif t.dim() > 1:
            t = t.view(t.size(0), -1).squeeze(1)
        t = t.view(-1, 1)  # [B,1]

        return self.net(torch.cat([x, t], dim=1))


# =============================
# RK4 integrator (robust shapes)
# =============================
def rk4_step(x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, f: DynamicsNet) -> torch.Tensor:
    """
    RK4 with strict shape handling:
      x  : [B, D]
      t  : scalar / [B] / [B,1] / [B,1,1] (normalized to [B] for f, which turns it into [B,1])
      dt : scalar / [B] / [B,1] -> normalized to [B,1]
    """
    B = x.size(0)

    # normalize dt -> [B,1]
    if dt.dim() == 0:
        dt_b = dt.view(1, 1).expand(B, 1)
    elif dt.dim() == 1:
        dt_b = dt.view(-1, 1)
    else:
        dt_b = dt.view(B, -1)[:, :1]

    # normalize t -> [B] here; DynamicsNet will convert to [B,1]
    if t.dim() == 0:
        t_b = t.expand(B)
    elif t.dim() == 1:
        t_b = t
    else:
        t_b = t.view(B, -1)[:, 0]

    k1 = f(x, t_b)
    k2 = f(x + 0.5 * dt_b * k1, t_b + 0.5 * dt_b.squeeze(1))
    k3 = f(x + 0.5 * dt_b * k2, t_b + 0.5 * dt_b.squeeze(1))
    k4 = f(x + dt_b * k3,       t_b +        dt_b.squeeze(1))
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
 
    """
    Train with reconstruction losses on consecutive pairs.
    Validate ONLY by multi-step future prediction (rollouts).
    """
    model_f.to(device)

    opt = torch.optim.AdamW(model_f.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)

    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    # batch_size=1 for per-trajectory rollout; keep shapes consistent
    dl_va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    best_val = float("inf")
    patience_ct = 0

    # =============================
    # Progress Bar
    # =============================
    tr_sum, n_tr = 0.0, 0
    pbar = tqdm(dl_tr, desc="Training", unit="batch")

    for ep in range(1, epochs + 1):
        # ---------------- Train (reconstruction) ----------------
        model_f.train()
        tr_sum, n_tr = 0.0, 0

        for x0, t0, x1, t1, dt in pbar:
            x0, x1 = x0.to(device), x1.to(device)
            t0, t1, dt = t0.to(device), t1.to(device), dt.to(device)

            opt.zero_grad()

            # forward/backward RK4 steps
            x1_hat = rk4_step(x0, t0, dt, model_f)       # predict next
            x0_hat = rk4_step(x1, t1, -dt, model_f)      # reconstruct prev
            with torch.no_grad():
                x1_hat_det = x1_hat.detach()
            x0_cyc = rk4_step(x1_hat_det, t1, -dt, model_f)  # cycle: forward then back

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
            pbar.set_postfix(loss=f"{tr_loss:.4f}")
        pbar.close()

        print("training done, starting validation...")

        # ---------------- Validation (future prediction ONLY) ----------------
        model_f.eval()
        va_sum, n_va = 0.0, 0
        with torch.no_grad():
            pbar = tqdm(dl_va, desc="Validating", unit="batch")
            for x_start, t_start, dt_seq, x_future in pbar:
                # Expected shapes:
                # x_start : [B, D]
                # t_start : [B]
                # dt_seq  : [B, H]
                # x_future: [B, H, D]
                B, H, D = x_future.shape

                x = x_start.to(device)                 # [B, D]
                t = t_start.to(device).view(B, 1)      # [B, 1]
                dt_seq = dt_seq.to(device)             # [B, H]
                gt = x_future.to(device)               # [B, H, D]

                preds = []
                for h in range(H):
                    dt_h = dt_seq[:, h].view(B, 1)          # [B, 1]
                    x = rk4_step(x, t, dt_h, model_f)       # [B, D]
                    t = t + dt_h                            # [B, 1]
                    preds.append(x.unsqueeze(1))            # [B, 1, D]

                preds = torch.cat(preds, dim=1)             # [B, H, D]

                # MSE over all elements (sum reduced by H*B for average later)
                batch_loss = F.mse_loss(preds, gt, reduction="sum")
                va_sum += batch_loss.item()
                n_va += B * H

                va_loss = va_sum / max(1, n_va)
                pbar.set_postfix(loss=f"{va_loss:.4f}")
        pbar.close()

        va_loss = va_sum / max(1, n_va)
        sched.step(va_loss)

        print(f"Epoch {ep:03d} | train={tr_loss:.6f} | val_future_pred={va_loss:.6f}")

        # Early stopping on future-pred error
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
    csv_path    = "../research/data/train.csv"  # single CSV with time,A..N,Y1,Y2 columns
    seed        = 42
    val_frac    = 0.15
    horizon     = 20   # steps ahead to evaluate during validation

    hidden      = 128
    depth       = 4

    epochs      = 300
    batch_size  = 528
    lr          = 1e-3
    weight_decay= 1e-4

    w_forward   = 1.0
    w_backward  = 1.0
    w_cycle     = 0.5
    patience    = 30
    # --------------------

    set_seed(seed)
    df = load_train_csv(csv_path)

    # Build pair dataset once to extract scaler stats (mean/std on x0)
    full_pairs = PairDataset(df)
    mean, std = full_pairs.mean, full_pairs.std

    # Split pair indices for train/val
    N = len(full_pairs)
    n_val = max(1, int(val_frac * N))
    perm = np.random.permutation(N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # Build train split using same scaler stats
    train_ds_all = PairDataset(df, scaler_state=(mean, std))

    def subset_pairs(ds: PairDataset, idxs: np.ndarray) -> PairDataset:
        sub = PairDataset.__new__(PairDataset)
        sub.t0 = ds.t0[idxs]; sub.t1 = ds.t1[idxs]; sub.dt = ds.dt[idxs]
        sub.x0 = ds.x0[idxs]; sub.x1 = ds.x1[idxs]
        sub.x0_raw = ds.x0_raw[idxs]; sub.x1_raw = ds.x1_raw[idxs]
        sub.mean = ds.mean; sub.std = ds.std
        return sub

    train_ds = subset_pairs(train_ds_all, tr_idx)

    # Validation dataset for future prediction windows
    val_ds = RolloutDataset(df, mean=mean, std=std, horizon=horizon)

    # Model
    x_dim = len(STATE_COLS)  # 14
    model_f = DynamicsNet(x_dim=x_dim, hidden=hidden, depth=depth)

    print("training start")

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
