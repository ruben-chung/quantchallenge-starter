import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import uniform, randint
import time

# ======================
# CONFIG: file locations
# ======================
DATA_DIR = "../research/data"  # change if needed
TRAIN_OLD = f"{DATA_DIR}/train.csv"
TRAIN_NEW = f"{DATA_DIR}/train_new.csv"
TEST_OLD  = f"{DATA_DIR}/test.csv"
TEST_NEW  = f"{DATA_DIR}/test_new.csv"

feature_cols = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
target_cols  = ['Y1','Y2']

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# robust multi-output RMSE scorer (forces numpy arrays)
def multi_output_rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse_y1 = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_y2 = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    return -(rmse_y1 + rmse_y2) / 2  # negative because sklearn maximizes

multi_rmse_scorer = make_scorer(multi_output_rmse, greater_is_better=False)

# ======================
# LOAD + COMBINE CSVs
# ======================
print("Loading and combining data...")
train_old = pd.read_csv(TRAIN_OLD)
train_new = pd.read_csv(TRAIN_NEW)
test_old  = pd.read_csv(TEST_OLD)
test_new  = pd.read_csv(TEST_NEW)

train_df = pd.concat([train_old, train_new], ignore_index=True)
test_df  = pd.concat([test_old,  test_new],  ignore_index=True)

# Optional: drop duplicate ids if present (keep latest)
if "id" in train_df.columns:
    train_df = train_df.drop_duplicates(subset=["id"], keep="last")
if "id" in test_df.columns:
    test_df = test_df.drop_duplicates(subset=["id"], keep="last")

# Ensure numeric features/targets; drop NaN rows in train; fill NaNs in test with train medians
X_all = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
y_all = train_df[target_cols].apply(pd.to_numeric, errors="coerce")

mask = ~(X_all.isna().any(axis=1) | y_all.isna().any(axis=1))
if (~mask).sum() > 0:
    print(f"[info] Dropping {(~mask).sum()} training rows with NaNs.")
X_all = X_all.loc[mask]
y_all = y_all.loc[mask]

X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
X_test = X_test.fillna(X_all.median())

print(f"Training set: {X_all.shape}")
print(f"Test set:     {X_test.shape}")

# Train/validation split (no stratify for regression)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

# ======================
# HYPERPARAMETER TUNING
# ======================
print("\n=== Starting Hyperparameter Tuning ===")

# XGBoost distributions
xgb_param_dist = {
    'estimator__n_estimators': randint(100, 500),
    'estimator__learning_rate': uniform(0.01, 0.2),
    'estimator__max_depth': randint(3, 10),
    'estimator__min_child_weight': randint(1, 10),
    'estimator__subsample': uniform(0.6, 0.4),
    'estimator__colsample_bytree': uniform(0.6, 0.4),
    'estimator__reg_alpha': uniform(0, 1),
    'estimator__reg_lambda': uniform(0, 2)
}

# LightGBM distributions
lgb_param_dist = {
    'estimator__n_estimators': randint(100, 500),
    'estimator__learning_rate': uniform(0.01, 0.2),
    'estimator__max_depth': randint(3, 10),
    'estimator__min_child_samples': randint(10, 100),
    'estimator__subsample': uniform(0.6, 0.4),
    'estimator__colsample_bytree': uniform(0.6, 0.4),
    'estimator__reg_alpha': uniform(0, 1),
    'estimator__reg_lambda': uniform(0, 2)
}

# Tune XGBoost
print("Tuning XGBoost...")
start_time = time.time()

xgb_base = MultiOutputRegressor(xgb.XGBRegressor(
    random_state=42, n_jobs=-1, tree_method="hist", verbosity=0
))
xgb_random = RandomizedSearchCV(
    xgb_base,
    xgb_param_dist,
    n_iter=50,
    cv=3,
    scoring=multi_rmse_scorer,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
xgb_random.fit(X_train_split, y_train_split)
xgb_tune_time = time.time() - start_time

print(f"XGBoost tuning completed in {xgb_tune_time:.2f} seconds")
print(f"Best XGBoost score (avg RMSE): {-xgb_random.best_score_:.4f}")
print(f"Best XGBoost params: {xgb_random.best_params_}")

# Tune LightGBM
print("\nTuning LightGBM...")
start_time = time.time()

lgb_base = MultiOutputRegressor(lgb.LGBMRegressor(
    random_state=42, verbose=-1, n_jobs=-1
))
lgb_random = RandomizedSearchCV(
    lgb_base,
    lgb_param_dist,
    n_iter=50,
    cv=3,
    scoring=multi_rmse_scorer,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
lgb_random.fit(X_train_split, y_train_split)
lgb_tune_time = time.time() - start_time

print(f"LightGBM tuning completed in {lgb_tune_time:.2f} seconds")
print(f"Best LightGBM score (avg RMSE): {-lgb_random.best_score_:.4f}")
print(f"Best LightGBM params: {lgb_random.best_params_}")

# ======================
# MODEL EVALUATION
# ======================
print("\n=== Evaluating Tuned Models ===")

best_xgb = xgb_random.best_estimator_
best_lgb = lgb_random.best_estimator_

models = {
    'XGBoost_Tuned': best_xgb,
    'LightGBM_Tuned': best_lgb
}

results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_pred_val = model.predict(X_val_split)

    rmse_y1 = np.sqrt(mean_squared_error(y_val_split['Y1'], y_pred_val[:, 0]))
    rmse_y2 = np.sqrt(mean_squared_error(y_val_split['Y2'], y_pred_val[:, 1]))
    r2_y1 = r2_score(y_val_split['Y1'], y_pred_val[:, 0])
    r2_y2 = r2_score(y_val_split['Y2'], y_pred_val[:, 1])

    results[name] = {
        'RMSE_Y1': rmse_y1,
        'RMSE_Y2': rmse_y2,
        'R2_Y1': r2_y1,
        'R2_Y2': r2_y2,
        'Avg_RMSE': (rmse_y1 + rmse_y2) / 2
    }

    print(f"Y1 - RMSE: {rmse_y1:.4f}, R2: {r2_y1:.4f}")
    print(f"Y2 - RMSE: {rmse_y2:.4f}, R2: {r2_y2:.4f}")
    print(f"Average RMSE: {results[name]['Avg_RMSE']:.4f}")

best_model_name = min(results.keys(), key=lambda x: results[x]['Avg_RMSE'])
best_model = models[best_model_name]

print(f"\n=== Best Model: {best_model_name} ===")
print(f"Average RMSE: {results[best_model_name]['Avg_RMSE']:.4f}")

# ======================
# FINAL PREDICTIONS
# ======================
print(f"\nTraining {best_model_name} on full combined training set...")
best_model.fit(X_all, y_all)

print("Generating predictions for combined test set...")
test_predictions = best_model.predict(X_test)

# Use existing test id if present; otherwise fallback to row index
test_ids = test_df['id'] if 'id' in test_df.columns else pd.Series(np.arange(len(test_df)))

submission_df = pd.DataFrame({
    'id': test_ids.values,
    'Y1': test_predictions[:, 0],
    'Y2': test_predictions[:, 1]
})

# ======================
# ANALYSIS AND INSIGHTS
# ======================
print("\n=== Prediction Analysis ===")
print(f"Test predictions shape: {test_predictions.shape}")
print(f"\nY1 Statistics: mean={submission_df['Y1'].mean():.4f}, std={submission_df['Y1'].std():.4f}, "
      f"min={submission_df['Y1'].min():.4f}, max={submission_df['Y1'].max():.4f}")
print(f"Y2 Statistics: mean={submission_df['Y2'].mean():.4f}, std={submission_df['Y2'].std():.4f}, "
      f"min={submission_df['Y2'].min():.4f}, max={submission_df['Y2'].max():.4f}")

print(f"\nTraining Y1 - Mean: {y_all['Y1'].mean():.4f}, Std: {y_all['Y1'].std():.4f}")
print(f"Training Y2 - Mean: {y_all['Y2'].mean():.4f}, Std: {y_all['Y2'].std():.4f}")

# Feature importance (per-target)
print(f"\n=== Feature Importance Analysis ===")
feature_importance_df = None
if 'XGBoost' in best_model_name:
    y1_importance = best_model.estimators_[0].feature_importances_
    y2_importance = best_model.estimators_[1].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Y1_Importance': y1_importance,
        'Y2_Importance': y2_importance,
        'Avg_Importance': (y1_importance + y2_importance) / 2
    }).sort_values('Avg_Importance', ascending=False)
    print("Top 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))

elif 'LightGBM' in best_model_name:
    y1_importance = best_model.estimators_[0].feature_importances_
    y2_importance = best_model.estimators_[1].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Y1_Importance': y1_importance,
        'Y2_Importance': y2_importance,
        'Avg_Importance': (y1_importance + y2_importance) / 2
    }).sort_values('Avg_Importance', ascending=False)
    print("Top 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))

# ======================
# ENSEMBLE (optional)
# ======================
print(f"\n=== Creating Ensemble Predictions ===")
best_xgb.fit(X_all, y_all)
best_lgb.fit(X_all, y_all)

xgb_test_pred = best_xgb.predict(X_test)
lgb_test_pred = best_lgb.predict(X_test)
ensemble_pred = (xgb_test_pred + lgb_test_pred) / 2

ensemble_df = pd.DataFrame({
    'id': test_ids.values,
    'Y1': ensemble_pred[:, 0],
    'Y2': ensemble_pred[:, 1]
})

print("Ensemble prediction stats:")
print(f"Y1 - mean={ensemble_df['Y1'].mean():.4f}, std={ensemble_df['Y1'].std():.4f}")
print(f"Y2 - mean={ensemble_df['Y2'].mean():.4f}, std={ensemble_df['Y2'].std():.4f}")

# ======================
# SAVE RESULTS
# ======================
submission_df.to_csv(f'predictions_{best_model_name.lower()}_combined.csv', index=False)
ensemble_df.to_csv('predictions_ensemble_combined.csv', index=False)
pd.DataFrame(results).T.to_csv('model_comparison_combined.csv')

if feature_importance_df is not None:
    feature_importance_df.to_csv('feature_importance_combined.csv', index=False)

# ======================
# CROSS-VALIDATION ANALYSIS (on best model)
# ======================
print(f"\n=== Cross-Validation Analysis ===")
cv_scores = cross_val_score(
    best_model, X_all, y_all,
    cv=5, scoring=multi_rmse_scorer, n_jobs=-1
)
print(f"5-Fold CV Scores (negative RMSE): {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\nDone. Files written:")
print(f"- predictions_{best_model_name.lower()}_combined.csv")
print(f"- predictions_ensemble_combined.csv")
print(f"- model_comparison_combined.csv")
print(f"- feature_importance_combined.csv (if computed)")
