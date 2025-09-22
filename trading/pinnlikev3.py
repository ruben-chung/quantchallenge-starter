# time_pinn_trainer.py
# ------------------------------------------------------------
# Purpose
#   Learn:
#     - f(x,t): continuous-time dynamics via RK4 (forward/back/cycle)
#     - g(x,t,dt): direct next-step predictor for X_{t+1}
#     - h(x,t): readout for Y=(Y1,Y2) at SAME time step
#
# Data
#   Single CSV with columns: time, A..N, Y1, Y2
#   We normalize X (A..N) and Y (Y1, Y2) by train-set statistics.
#
# Training losses
#   Given pairs (x_t, t_t) -> (x_{t+1}, t_{t+1}), dt > 0:
#   - RK4 forward/back/cycle on X using f:
#       * forward:    x_{t+1} ≈ RK4(x_t, t_t, +dt)
#       * backward:   x_t     ≈ RK4(x_{t+1}, t_{t+1}, -dt)
#       * cycle:      x_t     ≈ RK4(RK4(x_t, t_t, +dt), t_{t+1}, -dt)
#   - Direct next-step X via g:
#       * x_{t+1} ≈ g(x_t, t_t, dt)
#   - Y at SAME time via h:
#       * y_t ≈ h(x_t, t_t)
#     (optional anchors you can keep/tune)
#       * y_{t+1} ≈ h(RK4(x_t, t_t, +dt), t_{t+1})
#       * y_{t+1} ≈ h(g(x_t, t_t, dt),      t_{t+1})
#
# Validation metric
#   ONLY direct same-time:  y_hat = h(x_true, t_true) vs y_true
#   (No rollout, no next-step in validation.)
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
# Model save
# =============================
def save_checkpoint(save_path: str, model_f, model_g, readout, extra: dict = None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "model_f": model_f.state_dict(),
        "model_g": model_g.state_dict(),
        "readout": readout.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, save_path)

# =============================
# Data loading
# =============================
def load_train_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    # Ensure required columns exist
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    # Coerce numeric
    for c in FEATURES + TARGETS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaNs and sort by time
    df = df.dropna(subset=FEATURES + TARGETS).sort_values("time").reset_index(drop=True)
    return df


# =============================
# Training dataset (pairwise)
# =============================
class PairDataset(Dataset):
    """
    Returns (x_t, t_t, x_{t+1}, t_{t+1}, dt, y_t, y_{t+1}),
    with X and Y standardized using stats from x_t and y_t unless provided.
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
# Validation dataset (same-time Y only)
# =============================
class ValDatasetSameTimeY(Dataset):
    """
    For validation: same-time mapping (x,t) -> y.
    Returns batches of windows, but we only use the future X and times to compute Y at SAME steps.
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

        max_start = len(t) - (horizon + 1)
        self.starts = np.arange(max(0, max_start))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        i = self.starts[k]
        j = i + 1
        h = self.horizon

        # Start (for time base)
        x_start = torch.from_numpy(self.x[i]).float()     # [D]
        t_start = torch.tensor(self.t[i]).float()         # scalar

        # dt sequence, future X/Y windows
        dt_seq   = torch.from_numpy(np.diff(self.t[i:i+h+1]).astype(np.float32))  # [h]
        x_future = torch.from_numpy(self.x[j:j+h]).float()                        # [h, D]
        y_future = torch.from_numpy(self.y[j:j+h]).float()                        # [h, 2]

        return x_start, t_start, dt_seq, x_future, y_future


# =============================
# Models
# =============================
class DynamicsNet(nn.Module):
    """ f(x,t) -> dx/dt """
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
        if t.dim() == 0:
            t = t.expand(x.size(0))
        elif t.dim() > 1:
            t = t.view(t.size(0), -1).squeeze(1)
        t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=1))


class NextStepNet(nn.Module):
    """ g(x,t,dt) -> x_next (direct one-step transition) """
    def __init__(self, x_dim: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = x_dim + 2  # x + [t, dt]
        layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if t.dim() == 0: t = t.expand(B)
        elif t.dim() > 1: t = t.view(B, -1).squeeze(1)
        t = t.view(B, 1)

        if dt.dim() == 0: dt = dt.view(1).expand(B)
        elif dt.dim() > 1: dt = dt.view(B, -1).squeeze(1)
        dt = dt.view(B, 1)

        return self.net(torch.cat([x, t, dt], dim=1))


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
# RK4 integrator
# =============================
def rk4_step(x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, f: DynamicsNet) -> torch.Tensor:
    B = x.size(0)

    # dt -> [B,1]
    if dt.dim() == 0:
        dt_b = dt.view(1, 1).expand(B, 1)
    elif dt.dim() == 1:
        dt_b = dt.view(-1, 1)
    else:
        dt_b = dt.view(B, -1)[:, :1]

    # t -> [B]
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
    model_g: NextStepNet,
    readout: ReadoutNet,
    train_ds: PairDataset,
    val_ds: ValDatasetSameTimeY,
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    w_forward: float = 1.0,
    w_backward: float = 1.0,
    w_cycle: float = 0.5,
    w_xnext_direct: float = 1.0,
    w_y_now: float = 1.0,
    w_y_next_via_rk4: float = 0.25,
    w_y_next_via_g: float = 0.25,
    patience: int = 30,
    save_path: Optional[str] = None,           # <— NEW
    save_extra: Optional[dict] = None,         # <— NEW (scalers, metadata, etc.)
):
    """
    Train with:
      - RK4 reconstruction on X via f
      - Direct next-step X via g
      - SAME-TIME Y via h (required)
      - OPTIONAL next-time Y anchors via RK4 and/or g
    Validate ONLY by SAME-TIME Y from ground-truth X,t.
    """
    model_f.to(device)
    model_g.to(device)
    readout.to(device)

    opt = torch.optim.AdamW(
        list(model_f.parameters()) + list(model_g.parameters()) + list(readout.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)

    dl_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    best_val = float("inf")
    patience_ct = 0

    for ep in range(1, epochs + 1):
        # ---------------- Train ----------------
        model_f.train(); model_g.train(); readout.train()
        tr_sum, n_tr = 0.0, 0

        pbar_train = tqdm(dl_tr, desc=f"Training [ep {ep}/{epochs}]", unit="batch", leave=False)
        for x0, t0, x1, t1, dt, y0, y1 in pbar_train:
            x0, x1, y0, y1 = x0.to(device), x1.to(device), y0.to(device), y1.to(device)
            t0, t1, dt = t0.to(device), t1.to(device), dt.to(device)

            opt.zero_grad()

            # RK4 reconstruction via f
            x1_hat_rk4 = rk4_step(x0, t0, dt, model_f)        # predict next
            x0_hat_rk4 = rk4_step(x1, t1, -dt, model_f)       # reconstruct prev
            with torch.no_grad():
                x1_hat_det = x1_hat_rk4.detach()
            x0_cyc_rk4 = rk4_step(x1_hat_det, t1, -dt, model_f)

            l_fwd = F.mse_loss(x1_hat_rk4, x1)
            l_bwd = F.mse_loss(x0_hat_rk4, x0)
            l_cyc = F.mse_loss(x0_cyc_rk4, x0)

            # Direct next-step via g
            x1_hat_g = model_g(x0, t0, dt)
            l_xnext_direct = F.mse_loss(x1_hat_g, x1)

            # Y supervision
            y0_hat = readout(x0, t0)                   # same-time (required)
            ly_now = F.mse_loss(y0_hat, y0)

            # Optional: next-time anchors to stabilize h and coupling
            ly_next_rk4 = torch.tensor(0.0, device=device)
            if w_y_next_via_rk4 > 0:
                y1_hat_rk4 = readout(x1_hat_rk4, t1)
                ly_next_rk4 = F.mse_loss(y1_hat_rk4, y1)

            ly_next_g = torch.tensor(0.0, device=device)
            if w_y_next_via_g > 0:
                y1_hat_g = readout(x1_hat_g, t1)
                ly_next_g = F.mse_loss(y1_hat_g, y1)

            loss = (
                w_forward * l_fwd + w_backward * l_bwd + w_cycle * l_cyc
                + w_xnext_direct * l_xnext_direct
                + w_y_now * ly_now
                + w_y_next_via_rk4 * ly_next_rk4
                + w_y_next_via_g * ly_next_g
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model_f.parameters()) + list(model_g.parameters()) + list(readout.parameters()),
                1.0
            )
            opt.step()

            bs = x0.size(0)
            tr_sum += loss.item() * bs
            n_tr += bs
            pbar_train.set_postfix(loss=f"{tr_sum / max(1, n_tr):.4f}")
        pbar_train.close()
        tr_loss = tr_sum / max(1, n_tr)

        # ---------------- Validation (SAME-TIME Y only) ----------------
        model_f.eval(); model_g.eval(); readout.eval()
        va_sum, n_va = 0.0, 0

        pbar_val = tqdm(dl_va, desc="Validating", unit="batch", leave=False)
        with torch.no_grad():
            for x_start, t_start, dt_seq, x_future, y_future in pbar_val:
                # Shapes:
                # x_start : [B, D] (unused except for time base)
                # t_start : [B]
                # dt_seq  : [B, H]
                # x_future: [B, H, D]  (ground-truth X at each step)
                # y_future: [B, H, 2]  (ground-truth Y at each step)
                B, H, D = x_future.shape
                x_future = x_future.to(device)
                y_gt     = y_future.to(device)
                t0       = t_start.to(device).view(B, 1)
                dt_seq   = dt_seq.to(device)

                # Absolute times per step: t_h = t0 + cumsum(dt_seq)
                t_abs = t0 + torch.cumsum(dt_seq, dim=1)   # [B, H]

                # Predict Y directly from ground-truth X at the SAME times
                y_preds = []
                for h in range(H):
                    x_h = x_future[:, h, :]             # [B, D]
                    t_h = t_abs[:, h]                   # [B]
                    y_hat_h = readout(x_h, t_h)         # [B, 2]
                    y_preds.append(y_hat_h.unsqueeze(1))
                y_preds = torch.cat(y_preds, dim=1)     # [B, H, 2]

                batch_loss = F.mse_loss(y_preds, y_gt, reduction="sum")
                va_sum += batch_loss.item()
                n_va   += B * H
                pbar_val.set_postfix(loss=f"{va_sum / max(1, n_va):.4f}")
        pbar_val.close()

        va_loss = va_sum / max(1, n_va)
        sched.step(va_loss)

        print(f"Epoch {ep:03d} | train_total={tr_loss:.6f} | val_y_same_time={va_loss:.6f}")

        # Early stopping on validation metric
        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            patience_ct = 0

            # --- SAVE BEST ---
            if save_path is not None:
                save_checkpoint(save_path, model_f, model_g, readout, save_extra)
                print("saved")  # exactly as requested
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"[info] Early stopping at epoch {ep}. Best val={best_val:.6f}")
                break


# =============================
# Main (no artifacts saved)
# =============================
def main():
    # ---- edit these ----
    csv_path    = "../research/data/train.csv"  # expects time,A..N,Y1,Y2 columns
    seed        = 42
    val_frac    = 0.15
    horizon     = 20

    hidden      = 128
    depth_f     = 4     # dynamics depth
    depth_g     = 3     # next-step predictor depth
    depth_h     = 2     # readout depth

    epochs      = 300
    batch_size  = 528
    lr          = 1e-3
    weight_decay= 1e-4

    # reconstruction via f
    w_forward   = 1.0
    w_backward  = 1.0
    w_cycle     = 0.5

    # direct next-step X via g
    w_xnext_direct = 1.0

    # Y supervision (same-time required; next-time anchors optional)
    w_y_now         = 1.0
    w_y_next_via_rk4= 0.25
    w_y_next_via_g  = 0.25

    patience    = 30
    # --------------------

    set_seed(seed)
    df = load_train_csv(csv_path)

    # Build pair dataset to extract scalers
    full_pairs = PairDataset(df)
    x_mean, x_std = full_pairs.x_mean, full_pairs.x_std
    y_mean, y_std = full_pairs.y_mean, full_pairs.y_std

    # Split pairs for training batches
    N = len(full_pairs)
    n_val = max(1, int(val_frac * N))
    perm = np.random.permutation(N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # Train split using same scalers
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

    # Validation set (same-time Y only) with same scalers
    val_ds = ValDatasetSameTimeY(
        df, mean_x=x_mean, std_x=x_std, mean_y=y_mean, std_y=y_std, horizon=horizon
    )

    # Models
    x_dim = len(STATE_COLS)  # 14
    y_dim = len(TARGETS)     # 2
    model_f = DynamicsNet(x_dim=x_dim, hidden=hidden, depth=depth_f)
    model_g = NextStepNet(x_dim=x_dim, hidden=max(64, hidden), depth=depth_g)
    readout = ReadoutNet(x_dim=x_dim, y_dim=y_dim, hidden=max(64, hidden // 2), depth=depth_h)

    # Saving parameters
    out_dir = "./artifacts"
    save_path = os.path.join(out_dir, "best_model.pt")

    save_extra = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "meta": {
            "csv_path": csv_path,
            "horizon": horizon,
            "hidden": hidden,
            "depth_f": depth_f,
            "depth_g": depth_g,
            "depth_h": depth_h,
            "seed": seed,
        }
    }


    print("[info] Training start")
    train_loop(
        model_f=model_f,
        model_g=model_g,
        readout=readout,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        w_forward=w_forward, w_backward=w_backward, w_cycle=w_cycle,
        w_xnext_direct=w_xnext_direct,
        w_y_now=w_y_now, w_y_next_via_rk4=w_y_next_via_rk4, w_y_next_via_g=w_y_next_via_g,
        patience=patience,
        save_path=save_path,          # <— NEW
        save_extra=save_extra,        # <— NEW
    )



if __name__ == "__main__":
    main()
