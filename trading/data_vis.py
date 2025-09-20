#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main(csv_path: Path, out_dir: Optional[Path], img_format: str, dpi: int, show_plots: bool):
    df = pd.read_csv(csv_path)

    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
        except Exception:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["time"] = np.arange(len(df))

    feature_cols = [c for c in df.columns if c in list("ABCDEFGHIJKLMN")]
    target_cols = [c for c in df.columns if c in ["Y1", "Y2"]]

    df = df.sort_values("time").reset_index(drop=True)

    def savefig(name: str):
        if out_dir is None:
            return
        ensure_dir(out_dir)
        path = out_dir / f"{name}.{img_format}"
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    # 1) Y targets over time
    for tcol in target_cols:
        plt.figure()
        plt.plot(df["time"], df[tcol])
        plt.title(f"{tcol} over time")
        plt.xlabel("Time")
        plt.ylabel(tcol)
        savefig(f"time_{tcol}")
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 2) Pick top correlated features
    top_feats = []
    if "Y1" in target_cols:
        corr_vals = {}
        for c in feature_cols:
            try:
                corr_vals[c] = df[[c, "Y1"]].dropna().corr().iloc[0, 1]
            except Exception:
                corr_vals[c] = np.nan
        top_feats = list(pd.Series(corr_vals).dropna().abs().sort_values(ascending=False).index[:4])
    elif "Y2" in target_cols:
        corr_vals = {}
        for c in feature_cols:
            try:
                corr_vals[c] = df[[c, "Y2"]].dropna().corr().iloc[0, 1]
            except Exception:
                corr_vals[c] = np.nan
        top_feats = list(pd.Series(corr_vals).dropna().abs().sort_values(ascending=False).index[:4])
    else:
        top_feats = feature_cols[:4]

    # 3) Top features over time
    for c in top_feats:
        plt.figure()
        plt.plot(df["time"], df[c])
        plt.title(f"{c} over time")
        plt.xlabel("Time")
        plt.ylabel(c)
        savefig(f"time_{c}")
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 4) Scatter vs targets
    for tcol in target_cols:
        for c in top_feats:
            plt.figure()
            plt.scatter(df[c], df[tcol], s=10)
            plt.title(f"{tcol} vs {c}")
            plt.xlabel(c)
            plt.ylabel(tcol)
            savefig(f"scatter_{tcol}_vs_{c}")
            if show_plots:
                plt.show()
            else:
                plt.close()

    # 5) Correlation heatmap features vs targets
    if target_cols and feature_cols:
        sub = df[feature_cols + target_cols].astype(float)
        corr = sub.corr()
        corr_focus = corr.loc[feature_cols, [c for c in ["Y1", "Y2"] if c in corr.columns]]
        if not corr_focus.empty:
            plt.figure()
            plt.imshow(corr_focus.values, aspect="auto", interpolation="nearest")
            plt.colorbar()
            plt.xticks(ticks=np.arange(corr_focus.shape[1]), labels=corr_focus.columns, rotation=0)
            plt.yticks(ticks=np.arange(corr_focus.shape[0]), labels=corr_focus.index)
            plt.title("Correlation of features with targets")
            savefig("corr_heatmap_features_vs_targets")
            if show_plots:
                plt.show()
            else:
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize market dataset (time, A–N, Y1, Y2).")
    default_csv = Path("/Users/rubenchung/Desktop/GitHUB/quantchallenge-starter/research/data/train.csv")
    parser.add_argument("--csv", type=Path, default=default_csv,
                        help="Path to train.csv (defaults to your dataset path).")
    parser.add_argument("--out", type=Path, default=Path("graphs_out"), help="Directory to save images.")
    parser.add_argument("--format", type=str, default="png", help="Image format (png, jpg, svg, etc).")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (resolution).")
    parser.add_argument("--show", action="store_true", help="Also show plots interactively.")
    args = parser.parse_args()
    main(args.csv, args.out, args.format, args.dpi, args.show)
