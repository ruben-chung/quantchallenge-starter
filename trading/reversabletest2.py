import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ======================================
# Fixed columns: EXACTLY A..N (14 dims)
# ======================================
FEATURES = [chr(c) for c in range(ord("A"), ord("N") + 1)]  # ['A', ..., 'N'] -> 14
TARGETS  = ["Y1", "Y2"]


# -----------------
# Utilities
# -----------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGETS].apply(pd.to_numeric, errors="coerce")
    mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    bad = (~mask).sum()
    if bad:
        print(f"[info] Dropping {bad} bad rows", file=sys.stderr)
    return X[mask].values, y[mask].values


class XYDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


# -----------------
# Simple NICE flow
# -----------------
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
        self.dim = dim
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
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
    Stack multiple coupling layers with channel swaps to mix dimensions.
    Keeps shape 14 -> 14 exactly.
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


# -----------------
# Model
# -----------------
class FlowThenLinearHead(nn.Module):
    """
    14 -> (NICE flow) -> 14 -> Linear(14->2)
    """
    def __init__(self, in_dim: int = 14, flow_layers: int = 4, flow_hidden: int = 128):
        super().__init__()
        assert in_dim == 14, f"Expected in_dim=14 (A..N), got {in_dim}"
        self.flow = NICEFlow(in_dim, num_layers=flow_layers, hidden_dim=flow_hidden)
        self.head = nn.Linear(in_dim, 2)  # single layer 14->2 as requested

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.flow(x)       # 14 -> 14
        y = self.head(z)       # 14 -> 2
        return y

    # Optional helpers if you want to peek at/invert the representation
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow(x)
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.flow.inverse(z)


# -----------------
# Training script
# -----------------
def main():
    # ===== Edit these values as needed =====
    csv_path   = "../research/data/train.csv"   # single CSV; we'll split internally
    epochs     = 150
    batch_size = 64
    lr         = 1e-3
    weight_decay = 1e-4
    seed       = 42
    val_frac   = 0.15
    test_frac  = 0.15
    recon_w    = 0.0   # optional: ||x - inverse(encode(x))||^2 (on scaled X)
    flow_layers = 4
    flow_hidden = 128

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # Load data (A..N only -> 14 dims)
    X_all, y_all = load_csv(csv_path)
    if X_all.shape[1] != 14:
        raise ValueError(f"Expected exactly 14 features (A..N). Got {X_all.shape[1]}.")

    # Split train/test, then carve out validation from train
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_frac, random_state=seed
    )

    # Scale features with train statistics only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Build validation split
    n_train = X_train_s.shape[0]
    n_val = max(1, int(val_frac * n_train))
    perm = np.random.permutation(n_train)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr, y_tr = X_train_s[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train_s[val_idx], y_train[val_idx]

    # DataLoaders
    dl_tr  = DataLoader(XYDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(XYDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    dl_te  = DataLoader(XYDataset(X_test_s, y_test), batch_size=batch_size, shuffle=False)

    # Model/opt
    model = FlowThenLinearHead(in_dim=14, flow_layers=flow_layers, flow_hidden=flow_hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(10, epochs))
    mse   = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience, patience_ct = 25, 0

    # Train
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            if recon_w > 0:
                with torch.no_grad():
                    z = model.encode(xb)
                    xr = model.decode(z)
                loss = loss + recon_w * ((xr - xb) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        tr_loss = total / len(dl_tr.dataset)

        # Validate
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                l = mse(out, yb)
                if recon_w > 0:
                    z = model.encode(xb)
                    xr = model.decode(z)
                    l = l + recon_w * ((xr - xb) ** 2).mean()
                vloss += l.item() * xb.size(0)
        vloss /= len(dl_val.dataset)
        sched.step()

        print(f"Epoch {ep:03d} | train={tr_loss:.5f} | val={vloss:.5f}")

        if vloss < best_val - 1e-8:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"[info] Early stopping at epoch {ep}. Best val={best_val:.6f}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            gts.append(yb.numpy())
    preds, gts = np.vstack(preds), np.vstack(gts)
    mae = mean_absolute_error(gts, preds, multioutput="raw_values")
    mse_vals = mean_squared_error(gts, preds, multioutput="raw_values")
    print("\n=== Test Metrics ===")
    for i, t in enumerate(TARGETS):
        print(f"{t}: MAE={mae[i]:.5f}, MSE={mse_vals[i]:.5f}, RMSE={np.sqrt(mse_vals[i]):.5f}")

    # Save
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model_14to14_then2.pt")
    joblib.dump(scaler, "artifacts/scaler.joblib")
    print("Saved model & scaler to ./artifacts")


if __name__ == "__main__":
    main()
