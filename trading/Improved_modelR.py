# Retry with a lighter, faster benchmark to avoid timeouts:
# - 3-fold CV
# - Fewer models
# - Smaller forest size

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer, r2_score

from pathlib import Path
from caas_jupyter_tools import display_dataframe_to_user

train_path = Path("/mnt/data/train.csv")
test_path  = Path("/mnt/data/test.csv")

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

expected_features = [chr(c) for c in range(ord("A"), ord("N") + 1)]
targets = ["Y1", "Y2"]

missing_targets = [t for t in targets if t not in train_df.columns]
if missing_targets:
    raise ValueError(f"Missing target columns in train.csv: {missing_targets}")

if all(col in train_df.columns for col in expected_features):
    feature_cols = expected_features
else:
    feature_cols = [c for c in train_df.columns if c not in targets]

X = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
y = train_df[targets].apply(pd.to_numeric, errors="coerce")
mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
X = X.loc[mask]
y = y.loc[mask]

X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
X_test = X_test.fillna(X.median())

def mean_r2(y_true, y_pred):
    r2s = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return float(np.mean(r2s))

mean_r2_scorer = make_scorer(mean_r2, greater_is_better=True)

candidates = {
    "LinearRegression": Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
    "RidgeCV": Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(RidgeCV(alphas=np.logspace(-3,3,15))))]),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
}

kf = KFold(n_splits=3, shuffle=True, random_state=42)
leaderboard = []
for name, model in candidates.items():
    try:
        scores = cross_val_score(model, X, y, cv=kf, scoring=mean_r2_scorer)
        leaderboard.append({"model": name, "mean_cv_meanR2": float(np.mean(scores)), "std": float(np.std(scores))})
    except Exception as e:
        leaderboard.append({"model": name, "mean_cv_meanR2": -np.inf, "std": np.nan, "error": str(e)})

lb_df = pd.DataFrame(leaderboard).sort_values(by="mean_cv_meanR2", ascending=False).reset_index(drop=True)
display_dataframe_to_user("Model Leaderboard (fast 3-fold mean R^2)", lb_df)

best_name = lb_df.iloc[0]["model"]
best_model = candidates[best_name]
best_model.fit(X, y)
y_pred = best_model.predict(X_test)

ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
pred_df = pd.DataFrame({"id": ids, "y1": y_pred[:, 0], "y2": y_pred[:, 1]})
out_path = Path("/mnt/data/predictions.csv")
pred_df.to_csv(out_path, index=False)

out_path, best_name
