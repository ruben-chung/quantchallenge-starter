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


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # basic validation
    missing = [c for c in EXPECTED_FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGETS].copy()

    # coerce to numeric and drop NaNs
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in y.columns:
        y[c] = pd.to_numeric(y[c], errors="coerce")

    mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"[info] Dropping {dropped} rows with NaNs or non-numeric values.", file=sys.stderr)

    X = X[mask]
    y = y[mask]

    return X, y


def load_features_only(csv_path: str) -> pd.DataFrame:
    """
    Load ONLY A..N for inference (no targets required).
    Drops rows with non-numeric / NaNs in the features.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    missing_feats = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing required feature columns: {missing_feats}\nFound: {list(df.columns)}")

    X = df[EXPECTED_FEATURES].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    mask = ~X.isna().any(axis=1)
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"[info] Dropping {dropped} test rows with NaNs or non-numeric feature values.", file=sys.stderr)
    return X[mask].reset_index(drop=True)


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


def predict_from_csv(
    test_csv: str,
    model_path: str = "artifacts/model.pt",
    scaler_path: str = "artifacts/scaler.joblib",
    out_csv: str = "artifacts/test_predictions.csv",
    batch_size: int = 512,
    device: Optional[torch.device] = None,
):
    """
    Load A..N from `test_csv`, apply saved scaler+model, and write a CSV with:
        Y1_pred, Y2_pred
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load features
    X_df = load_features_only(test_csv)

    # Load scaler + model
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    scaler: StandardScaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_df.values)

    model = MLP(in_dim=X_scaled.shape[1], hidden=[64, 64, 32], out_dim=2, dropout=0.0).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Predict in batches
    preds = []
    with torch.no_grad():
        for i in range(0, X_scaled.shape[0], batch_size):
            xb = torch.from_numpy(X_scaled[i:i+batch_size].astype(np.float32)).to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
    preds = np.vstack(preds) if preds else np.zeros((0, 2), dtype=np.float32)

    # Save CSV with predictions only
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame({"Y1_pred": preds[:, 0], "Y2_pred": preds[:, 1]}).to_csv(out_csv, index=False)
    print(f"[info] Wrote predictions: {out_csv} (rows={len(preds)})")


def main():
    # Fixed hyperparameters and file path
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="../research/data/train.csv",
                        help="Path to labeled training CSV (must have A..N and Y1,Y2).")
    parser.add_argument("--predict", type=str, default=None,
                        help="If set, run inference on this test CSV (must have A..N) and exit.")
    parser.add_argument("--out_csv", type=str, default="artifacts/test_predictions.csv",
                        help="Where to write predictions CSV if --predict is set.")
    args = parser.parse_args()

    # If only predicting, do that and exit
    if args.predict is not None:
        predict_from_csv(
            test_csv=args.predict,
            out_csv=args.out_csv,
        )
        return

    # Otherwise: train
    epochs = 200
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-4
    patience = 20
    val_split = 0.15
    test_size = 0.15
    seed = 42
    dropout = 0.1

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # Load dataset
    X_df, y_df = load_dataset(args.train_csv)

    # Split train/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_df.values, y_df.values, test_size=test_size, random_state=seed
    )

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_temp_scaled  = scaler.transform(X_temp)

    # Train/val split
    n_train = X_train_scaled.shape[0]
    n_val = int(np.floor(val_split * n_train))
    perm = np.random.permutation(n_train)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr, y_tr = X_train_scaled[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]

    ds_tr  = XYDataset(X_tr, y_tr)
    ds_val = XYDataset(X_val, y_val)
    ds_te  = XYDataset(X_temp_scaled, y_temp)

    dl_tr  = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_te  = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    # Model
    model = MLP(in_dim=X_tr.shape[1], hidden=[512, 256, 128], out_dim=2, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, epochs))

    # Training loop with early stopping
    best_val, best_state, patience_counter = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
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
            if patience_counter >= patience:
                print(f"[info] Early stopping at epoch {epoch}. Best val_loss={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            gts.append(yb.numpy())
    preds, gts = np.vstack(preds), np.vstack(gts)

    mae = mean_absolute_error(gts, preds, multioutput="raw_values")
    mse = mean_squared_error(gts, preds, multioutput="raw_values")
    print("\n=== Test Metrics ===")
    for i, t in enumerate(TARGETS):
        print(f"{t}: MAE={mae[i]:.4f}, MSE={mse[i]:.4f}, RMSE={np.sqrt(mse[i]):.4f}")

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")
    joblib.dump(scaler, "artifacts/scaler.joblib")
    print("Saved model and scaler to ./artifacts/")


if __name__ == "__main__":
    main()
