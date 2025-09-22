import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


EXPECTED_FEATURES = [chr(c) for c in range(ord("A"), ord("N") + 1)]  # A..N
TARGETS = ["Y1", "Y2"]
OPTIONAL_COLS = ["time"]


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Loads the dataset, validates columns, coerces to numeric, drops NaNs.
    Returns X, y, and optional time Series aligned to the cleaned rows.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # basic validation
    missing = [c for c in EXPECTED_FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGETS].copy()
    t = df["time"].copy() if "time" in df.columns else pd.Series(np.arange(len(df)), name="time")

    # coerce to numeric and drop NaNs
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in y.columns:
        y[c] = pd.to_numeric(y[c], errors="coerce")
    t = pd.to_numeric(t, errors="coerce")

    mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1) | t.isna())
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"[info] Dropping {dropped} rows with NaNs or non-numeric values.", file=sys.stderr)

    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    t = t[mask].reset_index(drop=True)

    return X, y, t


class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [64, 64, 32], out_dim: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainConfig:
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    val_split: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    num_workers: int = 0
    dropout: float = 0.1


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)


def _check_time_continuity(time_values: np.ndarray):
    """
    Warns if time is not strictly increasing or if step sizes are inconsistent.
    """
    diffs = np.diff(time_values)
    if not np.all(diffs > 0):
        print("[warn] Validation 'time' is not strictly increasing before sort; sorting by time.", file=sys.stderr)
    # After sort, re-check
    diffs_sorted = np.diff(np.sort(time_values))
    if not np.all(diffs_sorted > 0):
        print("[warn] Duplicate or non-increasing 'time' values exist in validation set.", file=sys.stderr)
    # continuity check (uniform step)
    if len(diffs_sorted) > 0:
        # most common step
        steps, counts = np.unique(diffs_sorted, return_counts=True)
        mode_step = steps[np.argmax(counts)]
        if not np.allclose(diffs_sorted, mode_step):
            print("[warn] Validation 'time' steps are not uniform; continuity may be broken.", file=sys.stderr)


def main():
    # Fixed hyperparameters and file path
    csv_path = "../research/data/train.csv"      # <-- replace with your CSV file
    cfg = TrainConfig(
        epochs=200,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        patience=20,
        val_split=0.15,
        test_size=0.15,
        seed=42,
        dropout=0.1,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # Load dataset (includes 'time' if present; otherwise uses 0..N-1)
    X_df, y_df, t_sr = load_dataset(csv_path)

    # First split: train/test (keep time aligned by passing it to train_test_split)
    X_train_raw, X_temp_raw, y_train, y_temp, t_train, t_temp = train_test_split(
        X_df.values, y_df.values, t_sr.values,
        test_size=cfg.test_size, random_state=cfg.seed
    )

    # Second split: train/val within the training portion
    n_train_total = X_train_raw.shape[0]
    n_val = int(np.floor(cfg.val_split * n_train_total))
    perm = np.random.RandomState(cfg.seed).permutation(n_train_total)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr_raw, y_tr = X_train_raw[tr_idx], y_train[tr_idx]
    X_val_raw, y_val = X_train_raw[val_idx], y_train[val_idx]
    t_val = t_train[val_idx]  # <-- times for validation set

    # Fit scaler ONLY on training data to avoid leakage
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_val = scaler.transform(X_val_raw)
    X_te = scaler.transform(X_temp_raw)

    ds_tr  = XYDataset(X_tr, y_tr)
    ds_val = XYDataset(X_val, y_val)
    ds_te  = XYDataset(X_te, y_temp)

    dl_tr  = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_te  = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = MLP(in_dim=X_tr.shape[1], hidden=[64, 64, 32], out_dim=2, dropout=cfg.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, cfg.epochs))

    # Training loop with early stopping
    best_val, best_state, patience_counter = float("inf"), None, 0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, criterion, optimizer, device)
        val_loss = eval_epoch(model, dl_val, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"[info] Early stopping at epoch {epoch}. Best val_loss={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== Final TEST evaluation (unchanged) =====
    preds_te, gts_te = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds_te.append(out)
            gts_te.append(yb.numpy())
    preds_te, gts_te = np.vstack(preds_te), np.vstack(gts_te)

    mae = mean_absolute_error(gts_te, preds_te, multioutput="raw_values")
    mse = mean_squared_error(gts_te, preds_te, multioutput="raw_values")
    print("\n=== Test Metrics ===")
    for i, t in enumerate(TARGETS):
        print(f"{t}: MAE={mae[i]:.4f}, MSE={mse[i]:.4f}, RMSE={np.sqrt(mse[i]):.4f}")

    # ===== Final VALIDATION CSV with Y1/Y2 predictions, time ascending =====
    # Gather validation predictions in-order of ds_val (matches X_val_raw order)
    val_preds = []
    val_gts = []
    with torch.no_grad():
        model.eval()
        for xb, yb in dl_val:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            val_preds.append(pred)
            val_gts.append(yb.numpy())
    val_preds = np.vstack(val_preds)  # [N_val, 2]
    val_gts = np.vstack(val_gts)      # [N_val, 2]

    # Build DataFrame with time + truths + preds
    val_df = pd.DataFrame({
        "time": t_val,
        "Y1_true": val_gts[:, 0],
        "Y2_true": val_gts[:, 1],
        "Y1_pred": val_preds[:, 0],
        "Y2_pred": val_preds[:, 1],
    })

    # Ensure time ascending; check continuity and duplicates
    _check_time_continuity(val_df["time"].to_numpy())
    val_df = val_df.sort_values("time", kind="mergesort").reset_index(drop=True)

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")
    joblib.dump(scaler, "artifacts/scaler.joblib")
    val_csv_path = "artifacts/val_predictions.csv"
    val_df.to_csv(val_csv_path, index=False)

    print("Saved model and scaler to ./artifacts/")
    print(f"Saved validation predictions to {val_csv_path}")
    print(f"[info] Validation rows: {len(val_df)} | time range: [{val_df['time'].iloc[0]} .. {val_df['time'].iloc[-1]}]")


if __name__ == "__main__":
    main()
