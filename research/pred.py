# pred.py
import os
import numpy as np
import pandas as pd
import torch

from pinntest import DynamicsNet, NextStepNet, ReadoutNet, STATE_COLS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_from_csv(
    model_path: str,
    csv_in: str,
    csv_out: str,
):
    """
    Load the best model checkpoint and predict Y1,Y2 for new data.

    Input CSV must have columns: id, time, A..N
    Output CSV will have columns: id, Y1, Y2
    """
    # ---------- Load checkpoint (PyTorch 2.6: force weights_only=False) ----------
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # scalers (saved during training)
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std  = np.asarray(ckpt["x_std"],  dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std  = np.asarray(ckpt["y_std"],  dtype=np.float32)

    # model dims / hyperparams (fallbacks if meta missing)
    meta = ckpt.get("meta", {})
    hidden   = int(meta.get("hidden", 128))
    depth_f  = int(meta.get("depth_f", 4))
    depth_g  = int(meta.get("depth_g", 3))
    depth_h  = int(meta.get("depth_h", 2))

    x_dim = len(STATE_COLS)  # 14
    y_dim = 2

    # ---------- Rebuild models exactly as trained ----------
    model_f = DynamicsNet(x_dim=x_dim, hidden=hidden, depth=depth_f).to(device)
    model_g = NextStepNet(x_dim=x_dim, hidden=max(64, hidden), depth=depth_g).to(device)
    readout = ReadoutNet(x_dim=x_dim, y_dim=y_dim, hidden=max(64, hidden // 2), depth=depth_h).to(device)

    model_f.load_state_dict(ckpt["model_f"])
    model_g.load_state_dict(ckpt["model_g"])
    readout.load_state_dict(ckpt["readout"])

    model_f.eval(); model_g.eval(); readout.eval()

    # ---------- Load input CSV ----------
    df = pd.read_csv(csv_in)
    required = ["id", "time"] + STATE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    # normalize X (A..N) with training scalers
    x_raw = df[STATE_COLS].to_numpy(dtype=np.float32)
    x_norm = (x_raw - x_mean) / (x_std + 1e-8)

    t = df["time"].to_numpy(dtype=np.float32)

    x_tensor = torch.from_numpy(x_norm).to(device)
    t_tensor = torch.from_numpy(t).to(device)

    # ---------- Predict same-time Y ----------
    with torch.no_grad():
        y_hat_norm = readout(x_tensor, t_tensor)          # normalized Y
        y_hat = y_hat_norm.cpu().numpy() * y_std + y_mean # de-normalize

    out = pd.DataFrame({
        "id": df["id"],
        "Y1": y_hat[:, 0],
        "Y2": y_hat[:, 1],
    })
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    out.to_csv(csv_out, index=False)
    print(f"[info] Wrote predictions -> {csv_out}")



predict_from_csv(
    model_path="./artifacts/best_model.pt",
    csv_in="../research/data/test.csv",
    csv_out="./artifacts/predictions.csv")