#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

# ======================
# CONFIG: file locations
# ======================
DATA_DIR  = "../research/data"  # change if needed
TRAIN_OLD = f"{DATA_DIR}/train.csv"
TRAIN_NEW = f"{DATA_DIR}/train_new.csv"
TEST_OLD  = f"{DATA_DIR}/test.csv"
TEST_NEW  = f"{DATA_DIR}/test_new.csv"

# ====== Helpers ======
EXPECTED_FEATURES = [chr(c) for c in range(ord("A"), ord("N") + 1)]  # A..N
TARGETS = ["Y1", "Y2"]
OPTIONAL = {"time"}  # ignored if present

def set_seed(seed: int = 42):
    np.random.seed(seed)

def pick_features(train_df: pd.DataFrame) -> list[str]:
    if all(c in train_df.columns for c in EXPECTED_FEATURES):
        return EXPECTED_FEATURES
    # fallback: all non-target numeric columns (excluding optional)
    return [c for c in train_df.select_dtypes(include=[np.number]).columns
            if c not in TARGETS and c not in OPTIONAL]

def to_numeric_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def mean_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((r2_score(y_true[:, 0], y_pred[:, 0]) +
                  r2_score(y_true[:, 1], y_pred[:, 1])) / 2.0)

def kfold_cv_mean_r2(estimator, X: pd.DataFrame, y: pd.DataFrame, folds: int = 5, seed: int = 42) -> float:
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx].values, y.iloc[va_idx].values
        estimator.fit(X_tr, y_tr)
        y_hat = estimator.predict(X_va)
        scores.append(mean_r2(y_va, y_hat))
    return float(np.mean(scores))

def train_and_predict(train_path: str, test_path: str, out_dir: str, folds: int, seed: int):
    # load
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # checks
    for t in TARGETS:
        if t not in train.columns:
            raise ValueError(f"Missing target column in train: {t}")

    feature_cols = pick_features(train)

    # numeric conversion
    X = to_numeric_df(train, feature_cols)
    y = to_numeric_df(train, TARGETS)
    X_test = to_numeric_df(test, feature_cols)

    # drop bad rows in train
    mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[info] Dropping {dropped} train rows with NaNs or non-numeric values.")
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # fill test NaNs with train medians
    X_test = X_test.fillna(X.median(numeric_only=True))

    # small but solid param grid
    param_grid = [
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,  "max_depth": -1},
        {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,  "max_depth": -1},
        {"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 127, "max_depth": -1},
        {"n_estimators": 500, "learning_rate": 0.10, "num_leaves": 31,  "max_depth": -1},
    ]

    best_score = -np.inf
    best_est = None
    for i, p in enumerate(param_grid, 1):
        base = LGBMRegressor(
            random_state=seed,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            **p
        )
        model = MultiOutputRegressor(base)
        cv_score = kfold_cv_mean_r2(model, X, y, folds=folds, seed=seed)
        print(f"[cv] {i}/{len(param_grid)} params={p} | mean R^2={cv_score:.4f}")
        if cv_score > best_score:
            best_score = cv_score
            best_est = model

    if best_est is None:
        raise RuntimeError("No model selected.")

    print(f"[info] Best CV mean R^2 = {best_score:.4f}. Refitting on full train...")
    best_est.fit(X, y.values)

    # predict test
    y_hat = best_est.predict(X_test)

    # ids
    ids = test["id"].values if "id" in test.columns else np.arange(len(test))

    # write predictions
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.csv")
    pred_df = pd.DataFrame({"id": ids, "y1": y_hat[:, 0], "y2": y_hat[:, 1]})
    pred_df.to_csv(out_path, index=False)
    print(f"[info] Wrote predictions to {out_path}")

    # optional train-fit R² sanity
    y_fit = best_est.predict(X)
    r2_y1 = r2_score(y.values[:, 0], y_fit[:, 0])
    r2_y2 = r2_score(y.values[:, 1], y_fit[:, 1])
    print(f"[train fit] R^2 y1={r2_y1:.4f} y2={r2_y2:.4f} mean={(r2_y1+r2_y2)/2:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train LGBM multi-output regressor and write predictions CSV (id,y1,y2).")
    parser.add_argument("--which", choices=["new", "old", "both"], default="new",
                        help="Choose dataset pair: new=train_new/test_new, old=train/test, both=runs twice.")
    parser.add_argument("--folds", type=int, default=5, help="CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-root", type=str, default="artifacts", help="Root output directory")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.which in ("new", "both"):
        out_dir_new = os.path.join(args.out_root, "new")
        print(f"[run] Using NEW paths: train={TRAIN_NEW}  test={TEST_NEW}")
        train_and_predict(TRAIN_NEW, TEST_NEW, out_dir_new, args.folds, args.seed)

    if args.which in ("old", "both"):
        out_dir_old = os.path.join(args.out_root, "old")
        print(f"[run] Using OLD paths: train={TRAIN_OLD}  test={TEST_OLD}")
        train_and_predict(TRAIN_OLD, TEST_OLD, out_dir_old, args.folds, args.seed)

if __name__ == "__main__":
    main()
