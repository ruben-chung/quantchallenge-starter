#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# base models
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb
import lightgbm as lgb

# ----------------------------------------------------
# config
# ----------------------------------------------------
TRAIN_CSV = Path("/Users/rubenchung/Desktop/GitHUB/quantchallenge-starter/research/data/train.csv")
TEST_CSV  = Path("/Users/rubenchung/Desktop/GitHUB/quantchallenge-starter/research/data/test.csv")

FEATURE_COLS = list("ABCDEFGHIJKLMN")
TARGET_COLS  = ["Y1", "Y2"]

# choose lags (in rows). tune as needed.
LAGS = [1, 2, 3]
ADD_DELTAS = True       # create delta features: X - X.shift(1)
ADD_PCT    = False      # set True if you want pct_change() features too
ROLL_MEAN_WINDOWS: List[int] = []  # e.g., [3, 5] if you want rolling means


# ----------------------------------------------------
# feature engineering: lags, deltas, rolling, pct
# ----------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # ensure a time column exists and is datetime, then sort
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
        except Exception:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        # fallback: synthetic integer time if absent
        df["time"] = np.arange(len(df))
    return df.sort_values("time").reset_index(drop=True)

def make_ts_features(df: pd.DataFrame,
                     feature_cols: List[str],
                     lags: List[int],
                     add_deltas: bool = True,
                     add_pct: bool = False,
                     roll_windows: Optional[List[int]] = None) -> pd.DataFrame:
    roll_windows = roll_windows or []
    out = df.copy()

    # lags
    for c in feature_cols:
        for k in lags:
            out[f"{c}_lag{k}"] = out[c].shift(k)

    # deltas (against 1-step lag)
    if add_deltas:
        for c in feature_cols:
            out[f"{c}_delta1"] = out[c] - out[c].shift(1)

    # pct change (1-step)
    if add_pct:
        for c in feature_cols:
            out[f"{c}_pct1"] = out[c].pct_change(1).replace([np.inf, -np.inf], np.nan)

    # rolling means
    for w in roll_windows:
        for c in feature_cols:
            out[f"{c}_rollmean{w}"] = out[c].rolling(w).mean()

    return out

def build_train_test_with_lags(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        lags: List[int],
        add_deltas: bool = True,
        add_pct: bool = False,
        roll_windows: Optional[List[int]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # tag and concat so lags don’t leak across the boundary incorrectly
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["__is_train__"] = 1
    test_df["__is_train__"]  = 0

    # combine by time, engineer features once, then split back
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = add_time_features(combined)

    combined_fe = make_ts_features(
        combined, feature_cols, lags,
        add_deltas=add_deltas, add_pct=add_pct, roll_windows=roll_windows
    )

    # final split
    train_fe = combined_fe[combined_fe["__is_train__"] == 1].copy()
    test_fe  = combined_fe[combined_fe["__is_train__"] == 0].copy()

    # drop rows at the start of training where lag features are NaN
    # to avoid leakage, only drop in train (test may still align because we used combined)
    min_lag = max(lags) if lags else 0
    if min_lag > 0:
        # keep only rows from min_lag onwards in train
        train_fe = train_fe.iloc[min_lag:].copy()

    # assemble final feature column list
    extra_cols = []
    for c in feature_cols:
        for k in lags:
            extra_cols.append(f"{c}_lag{k}")
        if add_deltas:
            extra_cols.append(f"{c}_delta1")
        if add_pct:
            extra_cols.append(f"{c}_pct1")
        for w in (roll_windows or []):
            extra_cols.append(f"{c}_rollmean{w}")

    final_features = feature_cols + extra_cols

    # drop helper cols
    for dcol in ["__is_train__"]:
        if dcol in train_fe.columns: train_fe = train_fe.drop(columns=[dcol])
        if dcol in test_fe.columns:  test_fe  = test_fe.drop(columns=[dcol])

    return train_fe, test_fe, final_features


# ----------------------------------------------------
# stacking: LGB + XGB -> Ridge(meta)
# time-series aware out-of-fold predictions for meta-train
# ----------------------------------------------------
def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

def fit_base_models_get_oof_and_test(
    X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame, n_splits: int = 5, random_state: int = 42
):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # base models
    xgb_base = MultiOutputRegressor(
        xgb.XGBRegressor(
            random_state=random_state,
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1
        )
    )
    lgb_base = MultiOutputRegressor(
        lgb.LGBMRegressor(
            random_state=random_state,
            n_estimators=500, learning_rate=0.05, max_depth=-1,
            num_leaves=64, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1
        )
    )

    # out-of-fold containers for meta-model training
    oof_xgb = np.zeros((len(X), y.shape[1]))
    oof_lgb = np.zeros((len(X), y.shape[1]))

    # test preds per fold to average
    test_preds_xgb = []
    test_preds_lgb = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # fit
        xgb_base.fit(X_tr, y_tr)
        lgb_base.fit(X_tr, y_tr)

        # oof preds
        oof_xgb[va_idx, :] = xgb_base.predict(X_va)
        oof_lgb[va_idx, :] = lgb_base.predict(X_va)

        # test preds
        test_preds_xgb.append(xgb_base.predict(X_test))
        test_preds_lgb.append(lgb_base.predict(X_test))

        # quick fold metrics
        y_va_pred = lgb_base.predict(X_va)
        print(f"[fold {fold}] LGB RMSE Y1={rmse(y_va['Y1'], y_va_pred[:,0]):.4f}  Y2={rmse(y_va['Y2'], y_va_pred[:,1]):.4f}")

    # average test preds across folds
    test_xgb = np.mean(test_preds_xgb, axis=0)
    test_lgb = np.mean(test_preds_lgb, axis=0)

    # return oof matrices and averaged test preds
    return oof_xgb, oof_lgb, test_xgb, test_lgb


def stack_and_predict(
    X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get oof and test preds from base models
    oof_xgb, oof_lgb, test_xgb, test_lgb = fit_base_models_get_oof_and_test(X, y, X_test)

    # meta-features = concat the base predictions
    oof_meta  = np.column_stack([oof_xgb, oof_lgb])   # shape (n, 4) for 2 targets -> actually (n, 4)
    test_meta = np.column_stack([test_xgb, test_lgb])

    # train a simple Ridge meta-model per target (or use MultiOutputRegressor)
    meta = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
    meta.fit(oof_meta, y.values)

    # final predictions
    y_pred_train = meta.predict(oof_meta)     # meta fit on OOF (for reference metrics)
    y_pred_test  = meta.predict(test_meta)

    return y_pred_train, y_pred_test, test_meta


# ----------------------------------------------------
# main
# ----------------------------------------------------
def main():
    print("loading data…")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    # build lag/delta features
    print("building lag/delta features…")
    train_fe, test_fe, final_features = build_train_test_with_lags(
        train_df, test_df, FEATURE_COLS, TARGET_COLS,
        lags=LAGS, add_deltas=ADD_DELTAS, add_pct=ADD_PCT, roll_windows=ROLL_MEAN_WINDOWS
    )

    # split X/y
    X_train = train_fe[final_features].copy()
    y_train = train_fe[TARGET_COLS].copy()
    X_test  = test_fe[final_features].copy()

    print(f"train shape: X={X_train.shape}, y={y_train.shape} | test X={X_test.shape}")

    # stack
    print("training stacking ensemble (xgb + lgb -> ridge)…")
    y_pred_train, y_pred_test, _ = stack_and_predict(X_train, y_train, X_test)

    # metrics on last OOF (just for a quick sense)
    rmse_y1 = rmse(y_train["Y1"].values, y_pred_train[:, 0])
    rmse_y2 = rmse(y_train["Y2"].values, y_pred_train[:, 1])
    r2_y1   = r2_score(y_train["Y1"].values, y_pred_train[:, 0])
    r2_y2   = r2_score(y_train["Y2"].values, y_pred_train[:, 1])

    print(f"stack meta OOF metrics: RMSE_Y1={rmse_y1:.4f} R2_Y1={r2_y1:.4f} | RMSE_Y2={rmse_y2:.4f} R2_Y2={r2_y2:.4f}")

    # save predictions
    out = pd.DataFrame({
        "id": test_df["id"],
        "Y1": y_pred_test[:, 0],
        "Y2": y_pred_test[:, 1]
    })
    out_path = Path("predictions_stacked_lags.csv")
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}  shape={out.shape}")

if __name__ == "__main__":
    main()
