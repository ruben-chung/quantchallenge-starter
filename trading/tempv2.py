# time_pinn_trainer.py
# ------------------------------------------------------------
# Train on train.csv (time, A..N, Y1, Y2) with summed-MSE losses,
# evaluate validation (same-time Y only) using **summed MSE**,
# and EVERY TIME validation summed MSE improves:
#   -> run inference on test.csv (time, A..N)  [id is NOT used even if present]
#   -> write predictions:
#        - --out_csv (overwritten each improvement)
#        - snapshot: <stem>.best_epoch{ep}_val{val:.6e}.csv
#
# Architecture:
#   Copied from the provided script:
#     - DynamicsNet f(x,t)  (SiLU MLP, depth=4 by default)
#     - NextStepNet g(x,t,dt) (SiLU MLP, depth=3)
#     - ReadoutNet h(x,t)   (SiLU MLP, depth=2)
#
# Notes:
#   - No normalization/standardization (raw scale).
#   - We never save model weights; only CSVs are written on improvements.
#   - Validation uses ONLY ground-truth X and time to predict Y at the SAME step.
#   - Test output CSV has columns: id, Y1, Y2  where id is 0..N-1 (time-sorted).
# ------------------------------------------------------------

import os
import argparse
from typing import List

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
    # We explicitly ignore any 'id' column if present.
    require_cols(df, FEATURES, "test.csv")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES).sort_values("time").reset_index(drop=True)
    return df


# =============================
# Datasets (no scaling)
# =============================
class PairDataset(Dataset):
    """Training pairs: (x_t, t_t, x_{t+1}, t_{t+1}, dt, y_t, y_{t+1})"""
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
    Validation windows for same-time Y:
      (x_start, t_start, dt_seq[H], x_future[H,D], y_future[H,2])
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
    """Test rows for inference: (gen_id, x, t) — id is generated as row index 0..N-1 (time-sorted)."""
    def __init__(self, df: pd.DataFrame):
        self.t = df["time"].values.astype(np.float32)
        self.x = df[STATE_COLS].values.astype(np.float32)
        self.gen_ids = np.arange(len(df), dtype=np.int64)  # ignore any id column

    def __len__(self):
        return len(self.gen_ids)

    def __getitem__(self, i):
        return (
            int(self.gen_ids[i]),
            torch.from_numpy(self.x[i]).float(),  # [D]
            torch.tensor(self.t[i]).float(),      # []
        )


# =============================
# Models (architecture copied)
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
# Helpers for CSV writing on improvement
# =============================
def make_snapshot_path(out_csv: str, epoch: int, val_sum_mse: float) -> str:
    base, ext = os.path.splitext(out_csv)
    return f"{base}.best_epoch{epoch}_val{val_sum_mse:.6e}{ext}"


@torch.no_grad()
def predict_test_and_write_csv(
    readout: ReadoutNet, test_df: pd.DataFrame, out_csv: str, batch_size: int = 1024
):
    """Predict on test (ignores any id column) and write id,Y1,Y2 where id=0..N-1."""
    readout.to(device).eval()
    td = TestDataset(test_df)
    dl = DataLoader(td, batch_size=batch_size, shuffle=False, drop_last=False)

    ids_all: List[int] = []
    y1_all: List[float] = []
    y2_all: List[float] = []

    for ids, x, t in tqdm(dl, desc="Predicting test", unit="batch", leave=False):
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
# Training loop (write CSV on improvement)
# =============================
def train_loop(
    model_f: DynamicsNet,
    model_g: NextStepNet,
    readout: ReadoutNet,
    train_ds: PairDataset,
    val_ds: ValDatasetSameTimeY,
    test_df: pd.DataFrame,
    out_csv: str,
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
):
    """
    Loss per batch = SUM of MSEs (reduction='sum'):
      l_fwd + l_bwd + l_cyc  (X via RK4)
      + l_xnext_direct       (X via g)
      + ly_now               (same-time Y)
    Validation metric = **summed** MSE over all Y elements (same-time).
    On every improvement:
      -> write --out_csv and a snapshot CSV with epoch & val in the filename.
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

    best_val_sum = float("inf")
    patience_ct = 0

    for ep in range(1, epochs + 1):
        # ---------------- Train ----------------
        model_f.train(); model_g.train(); readout.train()
        tr_sum = 0.0

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

            # Summed MSEs
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

            tr_sum += loss.item()
            pbar_train.set_postfix(train_sum_mse=f"{tr_sum:.6e}")
        pbar_train.close()

        # ---------------- Validation (SAME-TIME Y only, summed MSE) ----------------
        model_f.eval(); model_g.eval(); readout.eval()
        va_sum = 0.0

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

                # **Summed** MSE over all elements in the batch window
                batch_sum = F.mse_loss(y_preds, y_gt, reduction="sum")
                va_sum += batch_sum.item()
                pbar_val.set_postfix(val_sum_mse=f"{va_sum:.6e}")
        pbar_val.close()

        sched.step(va_sum)
        print(f"Epoch {ep:03d} | val_sum_mse={va_sum:.6e}")

        # If improved, write predictions CSV(s) immediately
        if va_sum + 1e-12 < best_val_sum:
            best_val_sum = va_sum
            patience_ct = 0

            # Main CSV (overwrite) + snapshot CSV
            predict_test_and_write_csv(readout, test_df, out_csv, batch_size=max(512, batch_size))
            snap_path = make_snapshot_path(out_csv, ep, best_val_sum)
            predict_test_and_write_csv(readout, test_df, snap_path, batch_size=max(512, batch_size))
            print(f"[info] New best at epoch {ep}: val_sum_mse={best_val_sum:.6e}")
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"[info] Early stopping at epoch {ep}. Best val_sum_mse={best_val_sum:.6e}")
                break


# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser(
        description="Train PINN-like model; write CSV on every validation improvement (id ignored)."
    )
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train.csv (time,A..N,Y1,Y2).")
    parser.add_argument("--test_csv",  type=str, required=True, help="Path to test.csv (time,A..N).")
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
    val_ds = ValDatasetSameTimeY(train_df, horizon=args.horizon)

    # Use all pairs for training (validation is separate same-time windows)
    def subset_pairs(ds: PairDataset, idxs: np.ndarray) -> PairDataset:
        sub = PairDataset.__new__(PairDataset)
        sub.t0 = ds.t0[idxs]; sub.t1 = ds.t1[idxs]; sub.dt = ds.dt[idxs]
        sub.x0 = ds.x0[idxs]; sub.x1 = ds.x1[idxs]
        sub.y0 = ds.y0[idxs]; sub.y1 = ds.y1[idxs]
        return sub

    N = len(full_pairs)
    if N < 2:
        raise ValueError("Not enough consecutive pairs in train.csv (need at least 2 rows with strictly increasing time).")
    train_idx = np.arange(N)
    train_ds = subset_pairs(full_pairs, train_idx)

    # Models (architecture copied exactly from your working script)
    x_dim = len(STATE_COLS)
    y_dim = len(TARGETS)
    model_f = DynamicsNet(x_dim=x_dim, hidden=args.hidden, depth=args.depth_f)
    model_g = NextStepNet(x_dim=x_dim, hidden=max(64, args.hidden), depth=args.depth_g)
    readout = ReadoutNet(x_dim=x_dim, y_dim=y_dim, hidden=max(64, args.hidden // 2), depth=args.depth_h)

    print("[info] Training start")
    train_loop(
        model_f=model_f,
        model_g=model_g,
        readout=readout,
        train_ds=train_ds,
        val_ds=val_ds,
        test_df=test_df,
        out_csv=args.out_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
