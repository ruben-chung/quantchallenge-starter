import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import uniform, randint
import time

def rmse_scorer(y_true, y_pred):
    """Custom RMSE scorer for cross-validation"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Load data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Prepare features
feature_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
target_cols = ['Y1', 'Y2']

X_train = train_df[feature_cols]
y_train = train_df[target_cols]
X_test = test_df[feature_cols]

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Create validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=None
)

# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

print("\n=== Starting Hyperparameter Tuning ===")

# XGBoost parameter distributions
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

# LightGBM parameter distributions  
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

# Custom scorer for multi-output RMSE
def multi_output_rmse(y_true, y_pred):
    """Calculate average RMSE across both targets"""
    rmse_y1 = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_y2 = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    return -(rmse_y1 + rmse_y2) / 2  # Negative because sklearn maximizes

multi_rmse_scorer = make_scorer(multi_output_rmse, greater_is_better=False, needs_proba=False)

# Tune XGBoost
print("Tuning XGBoost...")
start_time = time.time()

xgb_base = MultiOutputRegressor(xgb.XGBRegressor(random_state=42, n_jobs=-1))
xgb_random = RandomizedSearchCV(
    xgb_base, 
    xgb_param_dist, 
    n_iter=50,  # Adjust based on computation time
    cv=3,
    scoring=multi_rmse_scorer,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

xgb_random.fit(X_train_split, y_train_split)
xgb_tune_time = time.time() - start_time

print(f"XGBoost tuning completed in {xgb_tune_time:.2f} seconds")
print(f"Best XGBoost score: {-xgb_random.best_score_:.4f}")
print(f"Best XGBoost params: {xgb_random.best_params_}")

# Tune LightGBM
print("\nTuning LightGBM...")
start_time = time.time()

lgb_base = MultiOutputRegressor(lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1))
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
print(f"Best LightGBM score: {-lgb_random.best_score_:.4f}")
print(f"Best LightGBM params: {lgb_random.best_params_}")

# =============================================================================
# MODEL EVALUATION
# =============================================================================

print("\n=== Evaluating Tuned Models ===")

# Get best models
best_xgb = xgb_random.best_estimator_
best_lgb = lgb_random.best_estimator_

# Evaluate on validation set
models = {
    'XGBoost_Tuned': best_xgb,
    'LightGBM_Tuned': best_lgb
}

results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Predict on validation set
    y_pred_val = model.predict(X_val_split)
    
    # Calculate metrics
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

# Select best model
best_model_name = min(results.keys(), key=lambda x: results[x]['Avg_RMSE'])
best_model = models[best_model_name]

print(f"\n=== Best Model: {best_model_name} ===")
print(f"Average RMSE: {results[best_model_name]['Avg_RMSE']:.4f}")

# =============================================================================
# FINAL PREDICTIONS
# =============================================================================

print(f"\nTraining {best_model_name} on full training set...")

# Train on full dataset
best_model.fit(X_train, y_train)

# Generate test predictions
print("Generating test set predictions...")
test_predictions = best_model.predict(X_test)

# Create submission dataframe
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Y1': test_predictions[:, 0],
    'Y2': test_predictions[:, 1]
})

# =============================================================================
# ANALYSIS AND INSIGHTS
# =============================================================================

print("\n=== Prediction Analysis ===")
print(f"Test predictions shape: {test_predictions.shape}")
print(f"\nY1 Statistics:")
print(f"  Mean: {submission_df['Y1'].mean():.4f}")
print(f"  Std:  {submission_df['Y1'].std():.4f}")
print(f"  Min:  {submission_df['Y1'].min():.4f}")
print(f"  Max:  {submission_df['Y1'].max():.4f}")

print(f"\nY2 Statistics:")
print(f"  Mean: {submission_df['Y2'].mean():.4f}")
print(f"  Std:  {submission_df['Y2'].std():.4f}")
print(f"  Min:  {submission_df['Y2'].min():.4f}")
print(f"  Max:  {submission_df['Y2'].max():.4f}")

# Compare with training statistics
print(f"\nTraining Y1 - Mean: {y_train['Y1'].mean():.4f}, Std: {y_train['Y1'].std():.4f}")
print(f"Training Y2 - Mean: {y_train['Y2'].mean():.4f}, Std: {y_train['Y2'].std():.4f}")

# Feature importance analysis
print(f"\n=== Feature Importance Analysis ===")
if 'XGBoost' in best_model_name:
    # XGBoost feature importance
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
    # LightGBM feature importance
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

# =============================================================================
# ENSEMBLE APPROACH (Optional)
# =============================================================================

print(f"\n=== Creating Ensemble Predictions ===")

# Train both tuned models on full dataset
best_xgb.fit(X_train, y_train)
best_lgb.fit(X_train, y_train)

# Get predictions from both models
xgb_test_pred = best_xgb.predict(X_test)
lgb_test_pred = best_lgb.predict(X_test)

# Create ensemble (simple average)
ensemble_pred = (xgb_test_pred + lgb_test_pred) / 2

ensemble_df = pd.DataFrame({
    'id': test_df['id'],
    'Y1': ensemble_pred[:, 0],
    'Y2': ensemble_pred[:, 1]
})

print("Ensemble prediction statistics:")
print(f"Y1 - Mean: {ensemble_df['Y1'].mean():.4f}, Std: {ensemble_df['Y1'].std():.4f}")
print(f"Y2 - Mean: {ensemble_df['Y2'].mean():.4f}, Std: {ensemble_df['Y2'].std():.4f}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

# Save individual best model predictions
submission_df.to_csv(f'predictions_{best_model_name.lower()}.csv', index=False)
print(f"\nBest model predictions saved to 'predictions_{best_model_name.lower()}.csv'")

# Save ensemble predictions
ensemble_df.to_csv('predictions_ensemble.csv', index=False)
print("Ensemble predictions saved to 'predictions_ensemble.csv'")

# Save detailed results
results_summary = pd.DataFrame(results).T
results_summary.to_csv('model_comparison.csv')
print("Model comparison saved to 'model_comparison.csv'")

# Save feature importance
if 'feature_importance_df' in locals():
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to 'feature_importance.csv'")

# =============================================================================
# CROSS-VALIDATION ANALYSIS
# =============================================================================

print(f"\n=== Cross-Validation Analysis ===")

# Perform 5-fold cross-validation on the best model
cv_scores = cross_val_score(
    best_model, X_train, y_train, 
    cv=5, scoring=multi_rmse_scorer, n_jobs=-1
)

print(f"5-Fold CV Scores (negative RMSE): {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n" + "="*60)
print(f"FINAL SUMMARY")
print(f"="*60)
print(f"Best Model: {best_model_name}")
print(f"Validation RMSE Y1: {results[best_model_name]['RMSE_Y1']:.4f}")
print(f"Validation RMSE Y2: {results[best_model_name]['RMSE_Y2']:.4f}")
print(f"Validation R² Y1: {results[best_model_name]['R2_Y1']:.4f}")
print(f"Validation R² Y2: {results[best_model_name]['R2_Y2']:.4f}")
print(f"Average RMSE: {results[best_model_name]['Avg_RMSE']:.4f}")
print(f"Cross-Validation Score: {cv_scores.mean():.4f}")
print(f"\nFiles Generated:")
print(f"- predictions_{best_model_name.lower()}.csv (best model)")
print(f"- predictions_ensemble.csv (ensemble)")
print(f"- model_comparison.csv (model metrics)")
print(f"- feature_importance.csv (feature analysis)")
print(f"="*60)

print(f"\nModel training and prediction pipeline completed successfully!")

# Optional: Print hyperparameter details
print(f"\n=== Best Hyperparameters ===")
if best_model_name == 'XGBoost_Tuned':
    print("XGBoost parameters:")
    for param, value in xgb_random.best_params_.items():
        print(f"  {param}: {value}")
else:
    print("LightGBM parameters:")
    for param, value in lgb_random.best_params_.items():
        print(f"  {param}: {value}")