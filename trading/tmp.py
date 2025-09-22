# time_pinn_trainer.py
# ------------------------------------------------------------
# Train on train.csv, validate on held-out windows within train.csv,
# then run the best (in-memory) model on test.csv and write predictions
# to a CSV with columns: id, Y1, Y2.
#
# Requirements:
#   train.csv: columns time, A..N, Y1, Y2
#   test.csv : columns time, A..N (+ optional id)
#
# Key design:
#   - No normalization/standardization (raw scale).
#   - Loss = SUM of MSEs for:
#       * RK4 forward/back/cycle on X via f(x,t)
#       * Direct next-step X via g(x,t,dt)
#       * Same-time Y via h(x,t)
#   - Validation metric: same-time Y MSE (direct h(x,t)), averaged per element.
#   - After training, run inference on test.csv and save id,Y1,Y2
# ------------------------------------------------------------

import os
import argparse
from typing import List, Optional, Tuple

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
TARGETS = ["Y1", "Y2"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================
# Data loading (no scaling)
# =============================
def require_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}\nFound: {list(df.columns)}")


def load_train_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    require_cols(df, FEATURES + TARGETS, "train.csv")

    for c in FEATURES + TARGETS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + TARGETS).sort_values("time").reset_index(drop=True)
    return df


def load_test_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    require_cols(df, FEATURES, "test.csv")  # Y not required
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES).sort_values("time").reset_index(drop=True)
    return df


def find_id_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer exact 'id', else case-insensitive match.
    if "id" in df.columns:
        return "id"
    for c in df.columns:
        if c.lower() == "id":
            return c
    return None


# =============================
# Datasets (no scaling)
# =============================
class PairDataset(Dataset):
    """
    Training pairs: (x_t, t_t, x_{t+1}, t_{t+1}, dt, y_t, y_{t+1})
    """
    def __init__(self, df: pd.DataFrame):
        t = df["time"].values.astype(np.float32)
        x = df[STATE_COLS].values.astype(np.float32)
        y = df[TARGETS].values.astype(np.float32)

        dt = np.diff(t)
        valid = dt > 0
        idx = np.where(valid)[0]  # i -> i+1

        self.t0 = t[idx]
        self.t1 = t[idx + 1]
        self.dt = dt[idx]
        self.x0 = x[idx]
        self.x1 = x[idx + 1]
        self.y0 = y[idx]
        self.y1 = y[idx + 1]

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x0[i]),            # [D]
            torch.tensor(self.t0[i]),                # []
            torch.from_numpy(self.x1[i]),            # [D]
            torch.tensor(self.t1[i]),                # []
            torch.tensor(self.dt[i]),                # []
            torch.from_numpy(self.y0[i]),            # [2]
            torch.from_numpy(self.y1[i]),            # [2]
        )


class ValDatasetSameTimeY(Dataset):
    """
    Validation windows for same-time Y: (x_start, t_start, dt_seq[H], x_future[H,D], y_future[H,2])
    """
    def __init__(self, df: pd.DataFrame, horizon: int = 20):
        t = df["time"].values.astype(np.float32)
        x = df[STATE_COLS].values.astype(np.float32)
        y = df[TARGETS].values.astype(np.float32)

        self.horizon = horizon
        self.t = t
        self.x = x
        self.y = y

        max_start = len(t) - (horizon + 1)
        self.starts = np.arange(max(0, max_start))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        i = self.starts[k]
        j = i + 1
        h = self.horizon

        x_start = torch.from_numpy(self.x[i]).float()     # [D]
        t_start = torch.tensor(self.t[i]).float()         # []

        dt_seq   = torch.from_numpy(np.diff(self.t[i:i+h+1]).astype(np.float32))  # [h]
        x_future = torch.from_numpy(self.x[j:j+h]).float()                        # [h, D]
        y_future = torch.from_numpy(self.y[j:j+h]).float()                        # [h, 2]

        return x_start, t_start, dt_seq, x_future, y_future


class TestDataset(Dataset):
    """
    Test rows for inference: (id, x, t)
    If no id column, id = row index [0..N-1].
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.ids_col = find_id_column(df)
        self.ids = df[self.ids_col].values if self.ids_col is not None else np.arange(len(df))
        self.t = df["time"].values.astype(np.float32)
        self.x = df[STATE_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return (
            int(self.ids[i]),
            torch.from_numpy(self.x[i]).float(),  # [D]
            torch.tensor(self.t[i]).float(),      # []
        )


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
        if t.dim() == 0: t = t.expand(x.size(0))
        elif t.dim() > 1: t = t.view(t.size(0), -1).squeeze(1)
        t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=1))


class NextStepNet(nn.Module):
    """ g(x,t,dt) -> x_next """
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
    """ h(x,t) -> (Y1, Y2) """
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
        if t.dim() == 0: t = t.expand(x.size(0))
        elif t.dim() > 1: t = t.view(t.size(0), -1).squeeze(1)
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
    patience: int = 30,
):
    """
    Loss per batch = SUM of MSEs:
      l_fwd  + l_bwd + l_cyc     (X via RK4)
      + l_xnext_direct           (X via g)
      + ly_now                   (same-time Y)
    Validation metric = mean MSE over Y predictions at same time (h(x,t)).
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
    best_state = None
    patience_ct = 0

    for ep in range(1, epochs + 1):
        # ---------------- Train ----------------
        model_f.train(); model_g.train(); readout.train()
        tr_loss_sum = 0.0
        tr_count = 0

        pbar_train = tqdm(dl_tr, desc=f"Training [ep {ep}/{epochs}]", unit="batch", leave=False)
        for x0, t0, x1, t1, dt, y0, y1 in pbar_train:
            x0, x1, y0, y1 = x0.to(device), x1.to(device), y0.to(device), y1.to(device)
            t0, t1, dt = t0.to(device), t1.to(device), dt.to(device)

            opt.zero_grad()

            # RK4 reconstruction via f
            x1_hat_rk4 = rk4_step(x0, t0, dt, model_f)
            x0_hat_rk4 = rk4_step(x1, t1, -dt, model_f)
            with torch.no_grad():
                x1_hat_det = x1_hat_rk4.detach()
            x0_cyc_rk4 = rk4_step(x1_hat_det, t1, -dt, model_f)

            # SUM MSE components (reduction='sum')
            l_fwd = F.mse_loss(x1_hat_rk4, x1, reduction="sum")
            l_bwd = F.mse_loss(x0_hat_rk4, x0, reduction="sum")
            l_cyc = F.mse_loss(x0_cyc_rk4, x0, reduction="sum")

            # Direct next-step via g
            x1_hat_g = model_g(x0, t0, dt)
            l_xnext_direct = F.mse_loss(x1_hat_g, x1, reduction="sum")

            # SAME-TIME Y via h
            y0_hat = readout(x0, t0)
            ly_now = F.mse_loss(y0_hat, y0, reduction="sum")

            loss = l_fwd + l_bwd + l_cyc + l_xnext_direct + ly_now
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model_f.parameters()) + list(model_g.parameters()) + list(readout.parameters()),
                1.0
            )
            opt.step()

            tr_loss_sum += loss.item()
            tr_count += 1
            pbar_train.set_postfix(loss_sum=f"{tr_loss_sum / max(1,tr_count):.6e}")
        pbar_train.close()

        # ---------------- Validation (SAME-TIME Y only) ----------------
        model_f.eval(); model_g.eval(); readout.eval()
        va_sum, n_va = 0.0, 0

        pbar_val = tqdm(dl_va, desc="Validating", unit="batch", leave=False)
        with torch.no_grad():
            for x_start, t_start, dt_seq, x_future, y_future in pbar_val:
                # x_future: [B, H, D], y_future: [B, H, 2]
                B, H, D = x_future.shape
                x_future = x_future.to(device)
                y_gt     = y_future.to(device)
                t0       = t_start.to(device).view(B, 1)
                dt_seq   = dt_seq.to(device)

                # Absolute times per step: t_h = t0 + cumsum(dt_seq)
                t_abs = t0 + torch.cumsum(dt_seq, dim=1)   # [B, H]

                # Predict Y directly from ground-truth X at SAME times
                y_preds = []
                for h in range(H):
                    x_h = x_future[:, h, :]         # [B, D]
                    t_h = t_abs[:, h]               # [B]
                    y_hat_h = readout(x_h, t_h)     # [B, 2]
                    y_preds.append(y_hat_h.unsqueeze(1))
                y_preds = torch.cat(y_preds, dim=1) # [B, H, 2]

                # Mean MSE per element as validation metric (stable wrt batch/horizon)
                batch_loss = F.mse_loss(y_preds, y_gt, reduction="sum")
                va_sum += batch_loss.item()
                n_va   += B * H * 2  # elements
                pbar_val.set_postfix(val_mse=f"{(va_sum/max(1,n_va)):.6e}")
        pbar_val.close()

        va_mse = va_sum / max(1, n_va)
        sched.step(va_mse)
        print(f"Epoch {ep:03d} | val_same_time_Y_mse={va_mse:.6e}")

        # Early stopping on validation metric; keep best state in memory
        if va_mse + 1e-12 < best_val:
            best_val = va_mse
            patience_ct = 0
            best_state = {
                "f": model_f.state_dict(),
                "g": model_g.state_dict(),
                "h": readout.state_dict(),
            }
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"[info] Early stopping at epoch {ep}. Best val mse={best_val:.6e}")
                break

    # Load best state before returning
    if best_state is not None:
        model_f.load_state_dict(best_state["f"])
        model_g.load_state_dict(best_state["g"])
        readout.load_state_dict(best_state["h"])

    return best_val


# =============================
# Inference on test + CSV write
# =============================
@torch.no_grad()
def predict_test_and_write_csv(
    readout: ReadoutNet, test_df: pd.DataFrame, out_csv: str, batch_size: int = 1024
):
    readout.to(device).eval()
    td = TestDataset(test_df)
    dl = DataLoader(td, batch_size=batch_size, shuffle=False, drop_last=False)

    ids_all: List[int] = []
    y1_all: List[float] = []
    y2_all: List[float] = []

    for batch in tqdm(dl, desc="Predicting test", unit="batch", leave=False):
        ids, x, t = batch
        x = x.to(device)
        t = t.to(device)

        y_hat = readout(x, t)  # [B, 2]
        y1_all.extend(y_hat[:, 0].cpu().numpy().tolist())
        y2_all.extend(y_hat[:, 1].cpu().numpy().tolist())
        ids_all.extend([int(i) for i in ids])

    out_df = pd.DataFrame({"id": ids_all, "Y1": y1_all, "Y2": y2_all})
    out_df.to_csv(out_csv, index=False)
    print(f"[info] Wrote predictions to {out_csv} ({len(out_df)} rows)")


# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser(description="Train PINN-like model and predict on test.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train.csv (time,A..N,Y1,Y2).")
    parser.add_argument("--test_csv",  type=str, required=True, help="Path to test.csv (time,A..N,[id]).")
    parser.add_argument("--out_csv",   type=str, default="predictions.csv", help="Output CSV path (id,Y1,Y2).")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--epochs",    type=int, default=300)
    parser.add_argument("--batch_size",type=int, default=512)
    parser.add_argument("--horizon",   type=int, default=20, help="Validation window length.")
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience",  type=int, default=30)
    parser.add_argument("--hidden",    type=int, default=128)
    parser.add_argument("--depth_f",   type=int, default=4)
    parser.add_argument("--depth_g",   type=int, default=3)
    parser.add_argument("--depth_h",   type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load data
    train_df = load_train_csv(args.train_csv)
    test_df  = load_test_csv(args.test_csv)

    # Build datasets
    full_pairs = PairDataset(train_df)

    # Validation windows from the same train_df (same-time Y only)
    val_ds = ValDatasetSameTimeY(train_df, horizon=args.horizon)

    # Simple random split of pair indices for training (pairs only affect reconstruction terms)
    N = len(full_pairs)
    if N < 2:
        raise ValueError("Not enough consecutive pairs in train.csv (need at least 2 rows with strictly increasing time).")
    perm = np.random.permutation(N)
    # Use all pairs for training since validation metric is computed from ValDatasetSameTimeY
    train_idx = perm

    # Subset the training pairs
    def subset_pairs(ds: PairDataset, idxs: np.ndarray) -> PairDataset:
        sub = PairDataset.__new__(PairDataset)
        sub.t0 = ds.t0[idxs]; sub.t1 = ds.t1[idxs]; sub.dt = ds.dt[idxs]
        sub.x0 = ds.x0[idxs]; sub.x1 = ds.x1[idxs]
        sub.y0 = ds.y0[idxs]; sub.y1 = ds.y1[idxs]
        return sub

    train_ds = subset_pairs(full_pairs, train_idx)

    # Models
    x_dim = len(STATE_COLS)
    y_dim = len(TARGETS)
    model_f = DynamicsNet(x_dim=x_dim, hidden=args.hidden, depth=args.depth_f)
    model_g = NextStepNet(x_dim=x_dim, hidden=max(64, args.hidden), depth=args.depth_g)
    readout = ReadoutNet(x_dim=x_dim, y_dim=y_dim, hidden=max(64, args.hidden // 2), depth=args.depth_h)

    print("[info] Training start")
    _ = train_loop(
        model_f=model_f,
        model_g=model_g,
        readout=readout,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    # Inference on test + CSV write (using best in-memory weights)
    predict_test_and_write_csv(readout, test_df, args.out_csv, batch_size=max(512, args.batch_size))


if __name__ == "__main__":
    main()
