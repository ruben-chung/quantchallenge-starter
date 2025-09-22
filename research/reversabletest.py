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

# =======================
# Config
# =======================
FEATURES = ["time"] + [chr(c) for c in range(ord("A"), ord("N") + 1)]
TARGETS = ["Y1", "Y2"]


# =======================
# Utilities
# =======================
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
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGETS].apply(pd.to_numeric, errors="coerce")

    mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    dropped = (~mask).sum()
    if dropped:
        print(f"[info] Dropping {dropped} bad rows", file=sys.stderr)
    return X[mask], y[mask]


class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =======================
# NICE flow
# =======================
class AdditiveCoupling(nn.Module):
    """
    NICE additive coupling with true indexing (not elementwise zero mask).
    We split x into x1 (active indices) and x2 (complement) using boolean mask.
    s(x1) maps R^{|x2|} from R^{|x1|}. We then merge back in original order.
    """
    def __init__(self, dim: int, hidden: List[int], mask: torch.Tensor):
        super().__init__()
        # mask is float tensor of shape [D] with 0/1 entries
        self.register_buffer("mask", mask)
        idx1 = (mask > 0.5).nonzero(as_tuple=False).squeeze(1)
        idx0 = (mask <= 0.5).nonzero(as_tuple=False).squeeze(1)
        self.register_buffer("idx1", idx1)  # active part
        self.register_buffer("idx0", idx0)  # complementary part

        in_dim  = idx1.numel()
        out_dim = idx0.numel()

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.s = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        # Split by indices (actual sub-vectors, not masked full vectors)
        x1 = x[:, self.idx1]  # shape [B, in_dim]
        x2 = x[:, self.idx0]  # shape [B, out_dim]

        shift = self.s(x1)    # shape [B, out_dim]
        if reverse:
            y2 = x2 - shift
            y1 = x1
        else:
            y2 = x2 + shift
            y1 = x1

        # Merge back to original ordering
        y = x.clone()
        y[:, self.idx1] = y1
        y[:, self.idx0] = y2
        return y


class Permute(nn.Module):
    def __init__(self, dim: int, perm: torch.Tensor):
        super().__init__()
        self.register_buffer("perm", perm)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(dim)
        self.register_buffer("invperm", inv)

    def forward(self, x, reverse: bool = False):
        return x[:, self.invperm] if reverse else x[:, self.perm]


class NICEFlow(nn.Module):
    def __init__(self, dim: int, num_blocks: int = 4, hidden: List[int] = [128, 128]):
        super().__init__()
        layers = []
        base_mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(dim)], dtype=torch.float32)
        perms = [torch.randperm(dim) for _ in range(num_blocks)]
        for b in range(num_blocks):
            mask = base_mask if b % 2 == 0 else 1 - base_mask
            layers.append(AdditiveCoupling(dim, hidden, mask))
            layers.append(Permute(dim, perms[b]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, reverse: bool = False):
        if reverse:
            for layer in reversed(self.layers):
                x = layer(x, reverse=True) if isinstance(layer, (AdditiveCoupling, Permute)) else layer(x)
            return x
        else:
            for layer in self.layers:
                x = layer(x, reverse=False) if isinstance(layer, (AdditiveCoupling, Permute)) else layer(x)
            return x


# =======================
# Model
# =======================
class FlowRegressor(nn.Module):
    def __init__(self, in_dim: int, head_hidden=[64, 64], num_blocks=4, flow_hidden=[128, 128]):
        super().__init__()
        self.pad = in_dim % 2 != 0
        self.flow_dim = in_dim + (1 if self.pad else 0)
        self.flow = NICEFlow(self.flow_dim, num_blocks=num_blocks, hidden=flow_hidden)
        layers = []
        prev = self.flow_dim
        for h in head_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 2)]
        self.head = nn.Sequential(*layers)

    def _pad(self, x):
        if not self.pad:
            return x
        pad = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    def _unpad(self, x):
        return x if not self.pad else x[:, :-1]

    def forward(self, x):
        z = self.flow(self._pad(x))
        return self.head(z)

    def encode(self, x):
        return self.flow(self._pad(x))

    def decode(self, z):
        return self._unpad(self.flow(z, reverse=True))


# =======================
# Training (fixed values)
# =======================
def main():
    csv_path = "../research/data/train.csv"     # <-- change as needed
    epochs = 200
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-4
    val_split = 0.15
    test_size = 0.15
    patience = 20
    seed = 42
    recon_w = 1e-3            # 0 disables reconstruction loss

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")

    # ----- data -----
    X_df, y_df = load_dataset(csv_path)
    X_all, y_all = X_df.values, y_df.values
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_tmp_s = scaler.transform(X_tmp)

    n_train = X_train_s.shape[0]
    n_val = max(1, int(val_split * n_train))
    perm = np.random.permutation(n_train)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr, y_tr = X_train_s[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train_s[val_idx], y_train[val_idx]

    dl_tr = DataLoader(XYDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(XYDataset(X_val, y_val), batch_size=batch_size)
    dl_te = DataLoader(XYDataset(X_tmp_s, y_tmp), batch_size=batch_size)

    # ----- model -----
    model = FlowRegressor(in_dim=X_tr.shape[1]).to(device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(10, epochs))

    best_val, best_state, patience_ct = float("inf"), None, 0

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            if recon_w > 0:
                rec = model.decode(model.encode(xb))
                loss += recon_w * ((rec - xb) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        tr_loss = total / len(dl_tr.dataset)

        # val
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                l = mse(out, yb)
                if recon_w > 0:
                    rec = model.decode(model.encode(xb))
                    l += recon_w * ((rec - xb) ** 2).mean()
                vloss += l.item() * xb.size(0)
        vloss /= len(dl_val.dataset)
        sched.step()

        print(f"Epoch {ep:03d} | train={tr_loss:.5f} | val={vloss:.5f}")

        if vloss < best_val - 1e-8:
            best_val, best_state = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"[info] Early stop @ {ep}, best val={best_val:.5f}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ----- test -----
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
        print(f"{t}: MAE={mae[i]:.4f}, MSE={mse_vals[i]:.4f}, RMSE={np.sqrt(mse_vals[i]):.4f}")

    # ----- save -----
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model_nice.pt")
    joblib.dump(scaler, "artifacts/scaler.joblib")
    print("Saved model + scaler to ./artifacts/")


if __name__ == "__main__":
    main()
