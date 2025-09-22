# time_flow_trainer.py
# ------------------------------------------------------------
# Train NICE-flow + linear head on train.csv (time, A..N, Y1, Y2),
# validate with SAME-TIME Y summed-MSE over time windows,
# and EVERY TIME validation improves:
#   -> predict on test.csv (time, A..N) [id is ignored]
#   -> write:
#        - --out_csv (overwrite each improvement)
#        - snapshot: <stem>.best_epoch{ep}_val{val:.6e}.csv
#
# No normalization/standardization. Raw scale end-to-end.
# ------------------------------------------------------------

import os
import argparse
from typing import List
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm


# =============================
# Columns / device
# =============================
FEATURES = ["time"] + [chr(c) for c in range(ord("A"), ord("N") + 1)]  # time + A..N
STATE_COLS = [c for c in FEATURES if c != "time"]                       # A..N (14)
TARGETS = ["Y1", "Y2"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================
# Data loading (NO scaling)
# =============================
def require_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}\nFound: {list(df.columns)}")

def load_train_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_csv(path)
    require_cols(df, FEATURES + TARGETS, "train.csv")
    for c in FEATURES + TARGETS: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + TARGETS).sort_values("time").reset_index(drop=True)
    return df

def load_test_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # We IGNORE any id column; only require time + A..N
    require_cols(df, FEATURES, "test.csv")
    for c in FEATURES: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES).sort_values("time").reset_index(drop=True)
    return df


# =============================
# Datasets
# =============================
class XYDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]

class ValDatasetWindows(Dataset):
    """
    Windowed validation for SAME-TIME Y:
      returns (x_future[H,D], y_future[H,2]) for each start index
    """
    def __init__(self, df: pd.DataFrame, horizon: int = 20):
        x = df[STATE_COLS].values.astype(np.float32)
        y = df[TARGETS].values.astype(np.float32)
        t = df["time"].values.astype(np.float32)
        self.h = horizon
        self.x = x; self.y = y; self.t = t
        max_start = len(t) - (horizon + 1)
        self.starts = np.arange(max(0, max_start))

    def __len__(self): return len(self.starts)

    def __getitem__(self, k):
        i = self.starts[k]
        j = i + 1
        h = self.h
        x_future = torch.from_numpy(self.x[j:j+h]).float()   # [H, D]
        y_future = torch.from_numpy(self.y[j:j+h]).float()   # [H, 2]
        return x_future, y_future


class TestDataset(Dataset):
    """
    Test rows for inference: (gen_id, x).
    id is GENERATED as 0..N-1 in time order (we ignore any id column).
    """
    def __init__(self, df: pd.DataFrame):
        self.x = df[STATE_COLS].values.astype(np.float32)
        self.gen_ids = np.arange(len(df), dtype=np.int64)

    def __len__(self): return len(self.gen_ids)

    def __getitem__(self, i):
        return int(self.gen_ids[i]), torch.from_numpy(self.x[i]).float()


# =============================
# NICE flow + linear head (copied arch)
# =============================
class NICECouplingLayer(nn.Module):
    """
    14 -> 14 additive coupling (split 7/7).
    y1 = x1
    y2 = x2 + m(x1)
    inverse:
    x1 = y1
    x2 = y2 - m(y1)
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        assert dim % 2 == 0, f"Coupling expects even dim, got {dim}"
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, half),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        y1 = x1
        y2 = x2 + self.net(x1)
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=1)
        x1 = y1
        x2 = y2 - self.net(y1)
        return torch.cat([x1, x2], dim=1)


class NICEFlow(nn.Module):
    """
    Stack multiple coupling layers with swaps to mix dims.
    Shape: 14 -> 14
    """
    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 128):
        super().__init__()
        assert dim % 2 == 0
        self.layers = nn.ModuleList([NICECouplingLayer(dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % 2 == 0:  # swap halves after every other layer
                x1, x2 = x.chunk(2, dim=1)
                x = torch.cat([x2, x1], dim=1)
        return x

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers) - 1, -1, -1):
            if i % 2 == 0:
                z1, z2 = z.chunk(2, dim=1)
                z = torch.cat([z2, z1], dim=1)
            z = self.layers[i].inverse(z)
        return z


class FlowThenLinearHead(nn.Module):
    """
    14 -> NICEFlow -> 14 -> Linear(14->2)
    """
    def __init__(self, in_dim: int = 14, flow_layers: int = 4, flow_hidden: int = 128):
        super().__init__()
        assert in_dim == 14, f"Expected in_dim=14 (A..N), got {in_dim}"
        self.flow = NICEFlow(in_dim, num_layers=flow_layers, hidden_dim=flow_hidden)
        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.flow(x)
        y = self.head(z)
        return y

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.flow.inverse(z)


# =============================
# CSV writing helpers
# =============================
def make_snapshot_path(out_csv: str, epoch: int, val_sum_mse: float) -> str:
    base, ext = os.path.splitext(out_csv)
    return f"{base}.best_epoch{epoch}_val{val_sum_mse:.6e}{ext}"

@torch.no_grad()
def predict_test_and_write_csv(
    model: FlowThenLinearHead, test_df: pd.DataFrame, out_csv: str, batch_size: int = 1024
):
    model.to(device).eval()
    dl = DataLoader(TestDataset(test_df), batch_size=batch_size, shuffle=False, drop_last=False)
    ids_all: List[int] = []
    y1_all: List[float] = []
    y2_all: List[float] = []
    for ids, xb in tqdm(dl, desc="Predicting test", unit="batch", leave=False):
        xb = xb.to(device)
        pred = model(xb)  # [B,2]
        ids_all.extend([int(i) for i in ids])
        y1_all.extend(pred[:, 0].cpu().numpy().tolist())
        y2_all.extend(pred[:, 1].cpu().numpy().tolist())
    out_df = pd.DataFrame({"id": ids_all, "Y1": y1_all, "Y2": y2_all})
    out_df.to_csv(out_csv, index=False)
    print(f"[info] Wrote predictions to {out_csv} ({len(out_df)} rows)")


# =============================
# Training (summed MSE + write on improvement)
# =============================
def train_loop(
    model: FlowThenLinearHead,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_csv: str,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 25,
    recon_w: float = 0.0,
    flow_layers: int = 4,
    flow_hidden: int = 128,
    horizon: int = 20,
):
    # Build loaders
    x_tr = train_df[STATE_COLS].values.astype(np.float32)
    y_tr = train_df[TARGETS].values.astype(np.float32)
    dl_tr = DataLoader(XYDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)

    val_windows = ValDatasetWindows(val_df, horizon=horizon)
    dl_va = DataLoader(val_windows, batch_size=batch_size, shuffle=False, drop_last=False)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(10, epochs))

    best_val_sum = float("inf")
    patience_ct = 0

    for ep in range(1, epochs + 1):
        # ---------- Train (summed MSE on Y, optional recon) ----------
        model.train()
        train_sum = 0.0

        pbar_tr = tqdm(dl_tr, desc=f"Training [ep {ep}/{epochs}]", unit="batch", leave=False)
        for xb, yb in pbar_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            pred = model(xb)                     # [B,2]
            loss_y = F.mse_loss(pred, yb, reduction="sum")

            # Optional invertibility reconstruction on X via flow
            if recon_w > 0:
                z = model.encode(xb)
                xr = model.decode(z)
                loss_recon = F.mse_loss(xr, xb, reduction="sum")
                loss = loss_y + recon_w * loss_recon
            else:
                loss = loss_y

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_sum += loss.item()
            pbar_tr.set_postfix(train_sum_mse=f"{train_sum:.6e}")
        pbar_tr.close()

        # ---------- Validation (SAME-TIME Y, windowed, SUMMED MSE) ----------
        model.eval()
        val_sum = 0.0
        pbar_va = tqdm(dl_va, desc="Validating", unit="batch", leave=False)
        with torch.no_grad():
            for x_future, y_future in pbar_va:
                # x_future: [B, H, D], y_future: [B, H, 2]
                B, H, D = x_future.shape
                x_future = x_future.to(device)
                y_gt = y_future.to(device)

                # Predict per step (no time used in this model)
                y_preds = []
                for h in range(H):
                    y_hat_h = model(x_future[:, h, :])  # [B,2]
                    y_preds.append(y_hat_h.unsqueeze(1))
                y_pred = torch.cat(y_preds, dim=1)       # [B,H,2]

                batch_sum = F.mse_loss(y_pred, y_gt, reduction="sum")
                val_sum += batch_sum.item()
                pbar_va.set_postfix(val_sum_mse=f"{val_sum:.6e}")
        pbar_va.close()
        sched.step()

        print(f"Epoch {ep:03d} | val_sum_mse={val_sum:.6e}")

        # ---------- Write CSV on improvement ----------
        if val_sum + 1e-12 < best_val_sum:
            best_val_sum = val_sum
            patience_ct = 0
            # main csv
            predict_test_and_write_csv(model, test_df, out_csv, batch_size=max(512, batch_size))
            # snapshot csv
            snap_path = make_snapshot_path(out_csv, ep, best_val_sum)
            #predict_test_and_write_csv(model, test_df, snap_path, batch_size=max(512, batch_size))
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
        description="Train NICE-flow + linear head; write CSV on each val improvement (id ignored)."
    )
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train.csv (time,A..N,Y1,Y2).")
    parser.add_argument("--test_csv",  type=str, required=True, help="Path to test.csv (time,A..N).")
    parser.add_argument("--out_csv",   type=str, default="predictions.csv", help="Output CSV path (id,Y1,Y2).")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--epochs",    type=int, default=150)
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience",  type=int, default=25)
    parser.add_argument("--horizon",   type=int, default=20, help="Validation window length.")
    parser.add_argument("--recon_w",   type=float, default=0.0, help="Optional X reconstruction weight via flow.")
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_hidden", type=int, default=128)
    parser.add_argument("--val_frac",  type=float, default=0.15, help="Fraction of train used as validation tail.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load data
    train_df = load_train_csv(args.train_csv)
    test_df  = load_test_csv(args.test_csv)

    # Build a time-based split for validation: use tail fraction as val to prevent leakage
    n = len(train_df)
    n_val = max(1, int(args.val_frac * n))
    tr_df = train_df.iloc[: n - n_val].copy()
    va_df = train_df.iloc[n - n_val :].copy()

    # Model
    model = FlowThenLinearHead(in_dim=14, flow_layers=args.flow_layers, flow_hidden=args.flow_hidden)

    print("[info] Training start")
    train_loop(
        model=model,
        train_df=tr_df,
        val_df=va_df,
        test_df=test_df,
        out_csv=args.out_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        recon_w=args.recon_w,
        flow_layers=args.flow_layers,
        flow_hidden=args.flow_hidden,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
