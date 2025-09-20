# Let's load the dataset, generate a few helpful charts, and also save a reusable
# VSCode-friendly script you can run locally (visualize_market_data.py).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("/Users/rubenchung/Desktop/GitHUB/quantchallenge-starter/research/data/train.csv")

# Load
df = pd.read_csv(csv_path)

# Basic parsing & cleaning
if "time" in df.columns:
    try:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
    except Exception:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
else:
    # If there's no explicit time column, create a synthetic time index (integer order)
    df["time"] = np.arange(len(df))

# Identify columns
feature_cols = [c for c in df.columns if c in list("ABCDEFGHIJKLMN")]
target_cols = [c for c in df.columns if c in ["Y1", "Y2"]]

# Sort by time for time-series plots
df = df.sort_values("time").reset_index(drop=True)

# --- 1) Plot Y1 and Y2 over time (each on its own plot) ---
for tcol in target_cols:
    plt.figure()
    plt.plot(df["time"], df[tcol])
    plt.title(f"{tcol} over time")
    plt.xlabel("Time")
    plt.ylabel(tcol)
    plt.tight_layout()
    plt.show()

# --- 2) Rank features by correlation with Y1 (if present) and show top-k ---
corr_table = pd.DataFrame()
if "Y1" in target_cols:
    corr_vals = {}
    for c in feature_cols:
        try:
            corr_vals[c] = df[[c, "Y1"]].dropna().corr().iloc[0, 1]
        except Exception:
            corr_vals[c] = np.nan
    corr_series = pd.Series(corr_vals, name="corr_with_Y1").sort_values(key=lambda s: s.abs(), ascending=False)
    corr_table = corr_series.to_frame()
else:
    # Fall back: correlations among features and Y2 if Y1 missing
    if "Y2" in target_cols:
        corr_vals = {}
        for c in feature_cols:
            try:
                corr_vals[c] = df[[c, "Y2"]].dropna().corr().iloc[0, 1]
            except Exception:
                corr_vals[c] = np.nan
        corr_series = pd.Series(corr_vals, name="corr_with_Y2").sort_values(key=lambda s: s.abs(), ascending=False)
        corr_table = corr_series.to_frame()

# Display correlation table if we computed one
try:
    from caas_jupyter_tools import display_dataframe_to_user
    if not corr_table.empty:
        display_dataframe_to_user("Feature correlations (top to bottom by |corr|)", corr_table.reset_index(names="feature"))
except Exception:
    pass

# Choose top features to visualize
top_feats = []
if not corr_table.empty:
    top_feats = list(corr_table.dropna().index[:4])
else:
    # If no correlations available, choose up to 4 alphabetical features by default
    top_feats = feature_cols[:4]

# --- 3) Time-series of top features (each on its own plot) ---
for c in top_feats:
    plt.figure()
    plt.plot(df["time"], df[c])
    plt.title(f"{c} over time")
    plt.xlabel("Time")
    plt.ylabel(c)
    plt.tight_layout()
    plt.show()

# --- 4) Scatter: top features vs targets (one plot per (feature,target)) ---
for tcol in target_cols:
    for c in top_feats:
        plt.figure()
        plt.scatter(df[c], df[tcol], s=10)
        plt.title(f"{tcol} vs {c}")
        plt.xlabel(c)
        plt.ylabel(tcol)
        plt.tight_layout()
        plt.show()

# --- 5) Small correlation "heatmap" between all features and Y1/Y2 (single plot) ---
if target_cols and feature_cols:
    sub = df[feature_cols + target_cols].astype(float)
    corr = sub.corr()
    # Keep rows = features, cols = targets only (if available)
    corr_focus = corr.loc[feature_cols, [c for c in ["Y1", "Y2"] if c in corr.columns]]
    if not corr_focus.empty:
        plt.figure()
        plt.imshow(corr_focus.values, aspect="auto", interpolation="nearest")
        plt.colorbar()
        plt.xticks(ticks=np.arange(corr_focus.shape[1]), labels=corr_focus.columns, rotation=0)
        plt.yticks(ticks=np.arange(corr_focus.shape[0]), labels=corr_focus.index)
        plt.title("Correlation of features with targets")
        plt.tight_layout()
        plt.show()

