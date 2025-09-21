# time_pinn_trainer.py
# ------------------------------------------------------------
# Purpose
#   Learn a continuous-time dynamics model f(x, t) for states A..N and
#   a readout head h(x, t) to predict Y1, Y2.
#
# Data
#   Single CSV with columns: time, A..N, Y1, Y2
#   We normalize X (A..N) and Y (Y1, Y2) by train-set statistics.
#
# Training (pairwise reconstruction + supervised Y)
#   Given consecutive pairs (x_t, t_t) -> (x_{t+1}, t_{t+1}), dt > 0:
#   - Forward:    x_{t+1} ≈ RK4(x_t, t_t,  +dt)
#   - Backward:   x_t     ≈ RK4(x_{t+1}, t_{t+1}, -dt)
#   - Cycle:      x_t     ≈ RK4(RK4(x_t, t_t, +dt), t_{t+1}, -dt)
#   - Y same-time:         y_t      ≈ h(x_t, t_t)
#   - Y next (teacher):    y_{t+1}  ≈ h(x_{t+1}, t_{t+1})
#   - Y next (pred via f): y_{t+1}  ≈ h(RK4(x_t, t_t, +dt), t_{t+1})
#
# Validation (multi-step rollout on Y ONLY)
#   Starting from (x_start, t_start), unroll H steps with RK4, predict Y at each step,
#   and compute MSE vs ground-truth Y window.
#
# Notes
#   - No artifacts (models/plots) are saved.
#   - Batch shape handling is robust for B > 1.
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
TARGETS = ["Y1", "Y2"]  # used for supervision

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

    # Ensure required columns exist (Y columns are used)
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    # Coerce numeric for features we use
    for c in FEATURES + TARGETS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaNs in features/targets and sort by time
    df = df.dropna(subset=FEATURES + TARGETS).sort_values("time").reset_index(drop=True)
    return df


# =============================
# Training dataset (pairwise)
# =============================
class PairDataset(Dataset):
    """
    Builds (x_t, t_t, x_{t+1}, t_{t+1}, dt, y_t, y_{t+1}) pairs for training.
    Scales x with (mean,std) and y with (mean,std) computed on t unless provided.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        scaler_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        scaler_target: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        t = df["time"].values.astype(np.float32)
        x = df[STATE_COLS].values.astype(np.float32)
        y = df[TARGETS].values.astype(np.float32)

        dt = np.diff(t)
        valid = dt > 0
        idx = np.where(valid)[0]  # pairs i -> i+1

        self.t0 = t[idx]
        self.t1 = t[idx + 1]
        self.dt = dt[idx]
        self.x0_raw = x[idx]
        self.x1_raw = x[idx + 1]
        self.y0_raw = y[idx]
        self.y1_raw = y[idx + 1]

        # X scaler
        if scaler_state is None:
            x_mean = self.x0_raw.mean(axis=0, keepdims=True)
            x_std  = self.x0_raw.std(axis=0, keepdims=True) + 1e-8
        else:
            x_mean, x_std = scaler_state

        # Y scaler
        if scaler_target is None:
            y_mean = self.y0_raw.mean(axis=0, keepdims=True)
            y_std  = self.y0_raw.std(axis=0, keepdims=True) + 1e-8
        else:
            y_mean, y_std = scaler_target

        self.x_mean = x_mean.astype(np.float32)
        self.x_std  = x_std.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std  = y_std.astype(np.float32)

        self.x0 = (self.x0_raw - self.x_mean) / self.x_std
        self.x1 = (self.x1_raw - self.x_mean) / self.x_std
        self.y0 = (self.y0_raw - self.y_mean) / self.y_std
        self.y1 = (self.y1_raw - self.y_mean) / self.y_std

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x0[i]),            # [D]
            torch.tensor(self.t0[i]),                # scalar
            torch.from_numpy(self.x1[i]),            # [D]
            torch.tensor(self.t1[i]),                # scalar
            torch.tensor(self.dt[i]),                # scalar
            torch.from_numpy(self.y0[i]),            # [2]
            torch.from_numpy(self.y1[i]),            # [2]
        )


# =============================
# Validation dataset (rollout)
# =============================
class RolloutDataset(Dataset):
    """
    For validation: start states and ground-truth future windows to evaluate multi-step Y prediction.
    Returns (x_start, t_start, dt_seq[H], x_future_seq[H], y_future_seq[H]) where H=horizon.
    All states and targets are scaled with training (mean,std).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        mean_x: np.ndarray,
        std_x: np.ndarray,
        mean_y: np.ndarray,
        std_y: np.ndarray,
        horizon: int = 20
    ):
        t = df["time"].values.astype(np.float32)
        x = df[STATE_COLS].values.astype(np.float32)
        y = df[TARGETS].values.astype(np.float32)

        self.horizon = horizon
        self.t = t
        self.x = (x - mean_x) / std_x
        self.y = (y - mean_y) / std_y
        self.x_mean = mean_x.astype(np.float32)
        self.x_std  = std_x.astype(np.float32)
        self.y_mean = mean_y.astype(np.float32)
        self.y_std  = std_y.astype(np.float32)

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
        dt_seq = torch.from_numpy(np.diff(self.t[i:i + h + 1]).astype(np.float32))  # [h]

        # Future states and targets: EXACTLY h steps: indices j..j+h-1
        x_future = torch.from_numpy(self.x[j:j + h]).to(torch.float32)  # [h, D]
        y_future = torch.from_numpy(self.y[j:j + h]).to(torch.float32)  # [h, 2]

        return x_start, t_start, dt_seq, x_future, y_future


# =============================
# Models
# =============================
class DynamicsNet(nn.Module):
    """ f(x,t) -> dx/dt """
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


class ReadoutNet(nn.Module):
    """ h(x,t) -> y_hat (Y1, Y2) """
    def __init__(self, x_dim: int, y_dim: int = 2, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = x_dim + 1
        layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, y_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.size(0))
        elif t.dim() > 1:
            t = t.view(t.size(0), -1).squeeze(1)
        t = t.view(-1, 1)
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
    readout: ReadoutNet,
    train_ds: PairDataset,
    val_ds: RolloutDataset,
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    # reconstruction weights
    w_forward: float = 1.0,
    w_backward: float = 1.0,
    w_cycle: float = 0.5,
    # supervised Y weights
    w_y_now: float = 1.0,
    w_y_next: float = 1.0,
    w_y_tf: float = 0.5,
    patience: int = 30,
):
    """
    Train with reconstruction on X and supervised losses on Y.
    Validate ONLY by multi-step Y rollout.
    """
    model_f.to(device)
    readout.to(device)

    opt = torch.optim.AdamW(
        list(model_f.parameters()) + list(readout.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)

    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    best_val = float("inf")
    patience_ct = 0

    for ep in range(1, epochs + 1):
        # ---------------- Train ----------------
        model_f.train()
        readout.train()
        tr_sum, n_tr = 0.0, 0

        pbar_train = tqdm(dl_tr, desc=f"Training [ep {ep}/{epochs}]", unit="batch", leave=False)
        for x0, t0, x1, t1, dt, y0, y1 in pbar_train:
            x0, x1, y0, y1 = x0.to(device), x1.to(device), y0.to(device), y1.to(device)
            t0, t1, dt = t0.to(device), t1.to(device), dt.to(device)

            opt.zero_grad()

            # RK4 reconstruction on X
            x1_hat = rk4_step(x0, t0, dt, model_f)       # predict next
            x0_hat = rk4_step(x1, t1, -dt, model_f)      # reconstruct prev
            with torch.no_grad():
                x1_hat_det = x1_hat.detach()
            x0_cyc = rk4_step(x1_hat_det, t1, -dt, model_f)  # cycle: forward then back

            l_fwd = F.mse_loss(x1_hat, x1)
            l_bwd = F.mse_loss(x0_hat, x0)
            l_cyc = F.mse_loss(x0_cyc, x0)

            # Y supervision
            y0_hat      = readout(x0, t0)          # same-time
            y1_hat_tf   = readout(x1, t1)          # teacher-forced next-time
            y1_hat_pred = readout(x1_hat, t1)      # predicted-next via dynamics

            ly_now  = F.mse_loss(y0_hat, y0)
            ly_tf   = F.mse_loss(y1_hat_tf, y1)
            ly_next = F.mse_loss(y1_hat_pred, y1)

            loss = (
                w_forward * l_fwd + w_backward * l_bwd + w_cycle * l_cyc
                + w_y_now * ly_now + w_y_next * ly_next + w_y_tf * ly_tf
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model_f.parameters()) + list(readout.parameters()), 1.0)
            opt.step()

            bs = x0.size(0)
            tr_sum += loss.item() * bs
            n_tr += bs
            pbar_train.set_postfix(loss=f"{tr_sum / max(1, n_tr):.4f}")
        pbar_train.close()
        tr_loss = tr_sum / max(1, n_tr)

        # ---------------- Validation (Y rollout) ----------------
        model_f.eval()
        readout.eval()
        va_sum, n_va = 0.0, 0

        pbar_val = tqdm(dl_va, desc="Validating", unit="batch", leave=False)
        with torch.no_grad():
            for x_start, t_start, dt_seq, x_future, y_future in pbar_val:
                # Expected shapes:
                # x_start : [B, D]
                # t_start : [B]
                # dt_seq  : [B, H]
                # x_future: [B, H, D]
                # y_future: [B, H, 2]
                B, H, D = x_future.shape

                x = x_start.to(device)                 # [B, D]
                t = t_start.to(device).view(B, 1)      # [B, 1]
                dt_seq = dt_seq.to(device)             # [B, H]
                y_gt = y_future.to(device)             # [B, H, 2]

                y_preds = []
                for h in range(H):
                    dt_h = dt_seq[:, h].view(B, 1)          # [B, 1]
                    x = rk4_step(x, t, dt_h, model_f)       # [B, D]
                    t = t + dt_h                            # [B, 1]
                    y_hat_h = readout(x, t.view(B))         # [B, 2]
                    y_preds.append(y_hat_h.unsqueeze(1))    # [B, 1, 2]

                y_preds = torch.cat(y_preds, dim=1)         # [B, H, 2]

                batch_loss = F.mse_loss(y_preds, y_gt, reduction="sum")
                va_sum += batch_loss.item()
                n_va += B * H

                pbar_val.set_postfix(loss=f"{va_sum / max(1, n_va):.4f}")
        pbar_val.close()

        va_loss = va_sum / max(1, n_va)
        sched.step(va_loss)

        print(f"Epoch {ep:03d} | train_total={tr_loss:.6f} | val_y_rollout={va_loss:.6f}")

        # Early stopping on Y-rollout error
        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"[info] Early stopping at epoch {ep}. Best val Y-rollout={best_val:.6f}")
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
    depth_f     = 4     # dynamics depth
    depth_h     = 2     # readout depth

    epochs      = 300
    batch_size  = 528
    lr          = 1e-3
    weight_decay= 1e-4

    # reconstruction weights
    w_forward   = 1.0
    w_backward  = 1.0
    w_cycle     = 0.5

    # Y supervision weights
    w_y_now     = 1.0
    w_y_next    = 1.0
    w_y_tf      = 0.5

    patience    = 30
    # --------------------

    set_seed(seed)
    df = load_train_csv(csv_path)

    # Build pair dataset once to extract scalers (x_mean/std, y_mean/std)
    full_pairs = PairDataset(df)
    x_mean, x_std = full_pairs.x_mean, full_pairs.x_std
    y_mean, y_std = full_pairs.y_mean, full_pairs.y_std

    # Split pair indices for train/val (for training batches)
    N = len(full_pairs)
    n_val = max(1, int(val_frac * N))
    perm = np.random.permutation(N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # Build train split using same scalers
    train_ds_all = PairDataset(df, scaler_state=(x_mean, x_std), scaler_target=(y_mean, y_std))

    def subset_pairs(ds: PairDataset, idxs: np.ndarray) -> PairDataset:
        sub = PairDataset.__new__(PairDataset)
        sub.t0 = ds.t0[idxs]; sub.t1 = ds.t1[idxs]; sub.dt = ds.dt[idxs]
        sub.x0 = ds.x0[idxs]; sub.x1 = ds.x1[idxs]
        sub.x0_raw = ds.x0_raw[idxs]; sub.x1_raw = ds.x1_raw[idxs]
        sub.y0 = ds.y0[idxs]; sub.y1 = ds.y1[idxs]
        sub.y0_raw = ds.y0_raw[idxs]; sub.y1_raw = ds.y1_raw[idxs]
        sub.x_mean = ds.x_mean; sub.x_std = ds.x_std
        sub.y_mean = ds.y_mean; sub.y_std = ds.y_std
        return sub

    train_ds = subset_pairs(train_ds_all, tr_idx)

    # Validation dataset for Y rollout windows (uses same scalers)
    val_ds = RolloutDataset(
        df, mean_x=x_mean, std_x=x_std, mean_y=y_mean, std_y=y_std, horizon=horizon
    )

    # Models
    x_dim = len(STATE_COLS)  # 14
    y_dim = len(TARGETS)     # 2
    model_f = DynamicsNet(x_dim=x_dim, hidden=hidden, depth=depth_f)
    readout = ReadoutNet(x_dim=x_dim, y_dim=y_dim, hidden=max(64, hidden // 2), depth=depth_h)

    print("[info] Training start")
    train_loop(
        model_f=model_f,
        readout=readout,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        w_forward=w_forward,
        w_backward=w_backward,
        w_cycle=w_cycle,
        w_y_now=w_y_now,
        w_y_next=w_y_next,
        w_y_tf=w_y_tf,
        patience=patience,
    )


if __name__ == "__main__":
    main()
