import os
import pickle
import warnings
import numpy as np
import pandas as pd
from time import time
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

print("=" * 80)
print("BASELINE MODEL TRAINING")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "n_folds": 5,
    "random_state": 42,
    "verbose": True,
    "early_stopping_rounds": 50,
    # Model configs
    "lightgbm": {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": -1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_estimators": 2000,
    },
    "xgboost": {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_estimators": 2000,
        "random_state": 42,
        "verbosity": 0,
    },
    "catboost": {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": 42,
        "verbose": False,
        "early_stopping_rounds": 50,
    },
}

# ============================================================================
# SMAPE METRIC
# ============================================================================


def calculate_smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    SMAPE = (1/n) * Σ |y_true - y_pred| / ((|y_true| + |y_pred|)/2)

    Returns value between 0 and 2 (multiply by 100 for percentage)
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(numerator / denominator)
    return smape


def smape_lgb(y_pred, dtrain):
    """SMAPE for LightGBM (requires log-transformed predictions)"""
    y_true = dtrain.get_label()
    y_pred_original = np.expm1(y_pred)
    y_true_original = np.expm1(y_true)
    smape = calculate_smape(y_true_original, y_pred_original)
    return "smape", smape, False


# ============================================================================
# LOAD DATA
# ============================================================================

print("LOADING DATA")
print("\n Loading features...")

X_train = pd.read_pickle("dataset/train_features.pkl")
X_test = pd.read_pickle("dataset/test_features.pkl")
y_train = np.load("dataset/y_train.npy")
train_ids = np.load("dataset/train_ids.npy")
test_ids = np.load("dataset/test_ids.npy")

print(f" Training features: {X_train.shape}")
print(f" Test features: {X_test.shape}")
print(f" Target: {y_train.shape}")

# Transform target (LOG TRANSFORMATION - CRITICAL!)
y_train_log = np.log1p(y_train)
print("\n Target transformed with log1p")
print(f"   Original range: ${y_train.min():.2f} - ${y_train.max():.2f}")
print(f"   Log range: {y_train_log.min():.3f} - {y_train_log.max():.3f}")

# ============================================================================
# CROSS-VALIDATION SETUP
# ============================================================================

print("\n" + "=" * 80)
print("CROSS-VALIDATION SETUP")
print("=" * 80)

kf = KFold(
    n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["random_state"]
)
print(f"\n {CONFIG['n_folds']}-Fold Cross-Validation")
print(f"   Random state: {CONFIG['random_state']}")

# Storage for results
cv_results = {
    "lightgbm": {"smape": [], "mae": [], "rmse": [], "fold_time": []},
    "xgboost": {"smape": [], "mae": [], "rmse": [], "fold_time": []},
    "catboost": {"smape": [], "mae": [], "rmse": [], "fold_time": []},
}

# Storage for models
models = {"lightgbm": [], "xgboost": [], "catboost": []}

# Storage for feature importance
feature_importance_lgb = np.zeros(X_train.shape[1])
feature_importance_xgb = np.zeros(X_train.shape[1])

# ============================================================================
# MODEL TRAINING - LIGHTGBM
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING LIGHTGBM CROSS-VALIDATION")
print("=" * 80)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"Fold {fold}/{CONFIG['n_folds']}")

    fold_start = time()

    # Split data
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_log[train_idx], y_train_log[val_idx]
    y_val_original = y_train[val_idx]

    print(f"Train size: {len(X_tr):,} | Val size: {len(X_val):,}")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train model
    print("\n Training LightGBM...")
    model = lgb.train(
        CONFIG["lightgbm"],
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        feval=smape_lgb,
        callbacks=[
            lgb.early_stopping(stopping_rounds=CONFIG["early_stopping_rounds"]),
            lgb.log_evaluation(period=200),
        ],
    )

    # Predict
    y_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = np.expm1(y_pred_log)

    # Calculate metrics
    smape = calculate_smape(y_val_original, y_pred)
    mae = mean_absolute_error(y_val_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))

    fold_time = time() - fold_start

    # Store results
    cv_results["lightgbm"]["smape"].append(smape)
    cv_results["lightgbm"]["mae"].append(mae)
    cv_results["lightgbm"]["rmse"].append(rmse)
    cv_results["lightgbm"]["fold_time"].append(fold_time)

    # Store model
    models["lightgbm"].append(model)

    # Accumulate feature importance
    feature_importance_lgb += model.feature_importance(importance_type="gain")

    print(f"\n Fold {fold} Results:")
    print(f"   SMAPE: {smape:.4f}")
    print(f"   MAE: ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   Time: {fold_time:.1f}s")

# Average results
print(f"\n{'=' * 60}")
print("LightGBM Cross-Validation Summary")
print(f"{'=' * 60}")
print(
    f"Average SMAPE: {np.mean(cv_results['lightgbm']['smape']):.4f} ± {np.std(cv_results['lightgbm']['smape']):.4f}"
)
print(
    f"Average MAE: ${np.mean(cv_results['lightgbm']['mae']):.2f} ± ${np.std(cv_results['lightgbm']['mae']):.2f}"
)
print(f"Average RMSE: ${np.mean(cv_results['lightgbm']['rmse']):.2f}")
print(f"Total time: {sum(cv_results['lightgbm']['fold_time']):.1f}s")

# ============================================================================
# MODEL TRAINING - XGBOOST
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING XGBOOST CROSS-VALIDATION")
print("=" * 80)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"Fold {fold}/{CONFIG['n_folds']}")

    fold_start = time()

    # Split data
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_log[train_idx], y_train_log[val_idx]
    y_val_original = y_train[val_idx]

    # Create XGBoost datasets
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train model
    print("\n Training XGBoost...")
    model = xgb.train(
        CONFIG["xgboost"],
        dtrain,
        num_boost_round=CONFIG["xgboost"]["n_estimators"],
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=CONFIG["early_stopping_rounds"],
        verbose_eval=200,
    )

    # Predict
    y_pred_log = model.predict(dval, iteration_range=(0, model.best_iteration))
    y_pred = np.expm1(y_pred_log)

    # Calculate metrics
    smape = calculate_smape(y_val_original, y_pred)
    mae = mean_absolute_error(y_val_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))

    fold_time = time() - fold_start

    # Store results
    cv_results["xgboost"]["smape"].append(smape)
    cv_results["xgboost"]["mae"].append(mae)
    cv_results["xgboost"]["rmse"].append(rmse)
    cv_results["xgboost"]["fold_time"].append(fold_time)

    # Store model
    models["xgboost"].append(model)

    # Accumulate feature importance
    importance_dict = model.get_score(importance_type="gain")
    for i in range(X_train.shape[1]):
        feature_importance_xgb[i] += importance_dict.get(f"f{i}", 0)

    print(f"\n Fold {fold} Results:")
    print(f"   SMAPE: {smape:.4f}")
    print(f"   MAE: ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   Time: {fold_time:.1f}s")

# Average results
print(f"\n{'=' * 60}")
print("XGBoost Cross-Validation Summary")
print(f"{'=' * 60}")
print(
    f"Average SMAPE: {np.mean(cv_results['xgboost']['smape']):.4f} ± {np.std(cv_results['xgboost']['smape']):.4f}"
)
print(
    f"Average MAE: ${np.mean(cv_results['xgboost']['mae']):.2f} ± ${np.std(cv_results['xgboost']['mae']):.2f}"
)
print(f"Average RMSE: ${np.mean(cv_results['xgboost']['rmse']):.2f}")
print(f"Total time: {sum(cv_results['xgboost']['fold_time']):.1f}s")

# ============================================================================
# MODEL TRAINING - CATBOOST
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING CATBOOST CROSS-VALIDATION")
print("=" * 80)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"Fold {fold}/{CONFIG['n_folds']}")

    fold_start = time()

    # Split data
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_log[train_idx], y_train_log[val_idx]
    y_val_original = y_train[val_idx]

    # Train model
    print("\n Training CatBoost...")
    model = CatBoostRegressor(**CONFIG["catboost"])
    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=CONFIG["early_stopping_rounds"],
        verbose=200,
    )

    # Predict
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)

    # Calculate metrics
    smape = calculate_smape(y_val_original, y_pred)
    mae = mean_absolute_error(y_val_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))

    fold_time = time() - fold_start

    # Store results
    cv_results["catboost"]["smape"].append(smape)
    cv_results["catboost"]["mae"].append(mae)
    cv_results["catboost"]["rmse"].append(rmse)
    cv_results["catboost"]["fold_time"].append(fold_time)

    # Store model
    models["catboost"].append(model)

    print(f"\n Fold {fold} Results:")
    print(f"   SMAPE: {smape:.4f}")
    print(f"   MAE: ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   Time: {fold_time:.1f}s")

# Average results
print(f"\n{'=' * 60}")
print("CatBoost Cross-Validation Summary")
print(f"{'=' * 60}")
print(
    f"Average SMAPE: {np.mean(cv_results['catboost']['smape']):.4f} ± {np.std(cv_results['catboost']['smape']):.4f}"
)
print(
    f"Average MAE: ${np.mean(cv_results['catboost']['mae']):.2f} ± ${np.std(cv_results['catboost']['mae']):.2f}"
)
print(f"Average RMSE: ${np.mean(cv_results['catboost']['rmse']):.2f}")
print(f"Total time: {sum(cv_results['catboost']['fold_time']):.1f}s")

# ============================================================================
# OVERALL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame(
    {
        "Model": ["LightGBM", "XGBoost", "CatBoost"],
        "SMAPE": [
            np.mean(cv_results["lightgbm"]["smape"]),
            np.mean(cv_results["xgboost"]["smape"]),
            np.mean(cv_results["catboost"]["smape"]),
        ],
        "SMAPE_std": [
            np.std(cv_results["lightgbm"]["smape"]),
            np.std(cv_results["xgboost"]["smape"]),
            np.std(cv_results["catboost"]["smape"]),
        ],
        "MAE": [
            np.mean(cv_results["lightgbm"]["mae"]),
            np.mean(cv_results["xgboost"]["mae"]),
            np.mean(cv_results["catboost"]["mae"]),
        ],
        "Training_Time": [
            sum(cv_results["lightgbm"]["fold_time"]),
            sum(cv_results["xgboost"]["fold_time"]),
            sum(cv_results["catboost"]["fold_time"]),
        ],
    }
)

comparison_df = comparison_df.sort_values("SMAPE")
print("\n" + comparison_df.to_string(index=False))

best_model = comparison_df.iloc[0]["Model"]
best_smape = comparison_df.iloc[0]["SMAPE"]

print(f"\n BEST MODEL: {best_model}")
print(f"   SMAPE: {best_smape:.4f}")

# Save comparison
os.makedirs("results", exist_ok=True)
comparison_df.to_csv("results/cv_scores.csv", index=False)
print("\n Saved results/cv_scores.csv")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Average feature importance across folds
feature_importance_lgb /= CONFIG["n_folds"]
feature_importance_xgb /= CONFIG["n_folds"]

# Create DataFrame
feature_names = X_train.columns
importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "lightgbm": feature_importance_lgb,
        "xgboost": feature_importance_xgb,
    }
)

# Average across models
importance_df["average"] = (importance_df["lightgbm"] + importance_df["xgboost"]) / 2
importance_df = importance_df.sort_values("average", ascending=False)

print("\n Top 20 Most Important Features:")
print(importance_df.head(20)[["feature", "average"]].to_string(index=False))

# Save
importance_df.to_csv("results/feature_importance.csv", index=False)
print("\n Saved results/feature_importance.csv")

# Plot
plt.figure(figsize=(12, 8))
top_20 = importance_df.head(20)
plt.barh(range(20), top_20["average"].values[::-1], color="skyblue", edgecolor="black")
plt.yticks(range(20), top_20["feature"].values[::-1])
plt.xlabel("Average Importance")
plt.title(
    "Top 20 Feature Importance (Average of LightGBM & XGBoost)",
    fontweight="bold",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(
    "results/visualizations/feature_importance.png", dpi=300, bbox_inches="tight"
)
plt.close()
print(" Saved results/visualizations/feature_importance.png")

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

os.makedirs("models/saved_models", exist_ok=True)

# Save all folds
print("\n Saving trained models...")

with open("models/saved_models/lightgbm_models.pkl", "wb") as f:
    pickle.dump(models["lightgbm"], f)
print(" Saved models/saved_models/lightgbm_models.pkl")

with open("models/saved_models/xgboost_models.pkl", "wb") as f:
    pickle.dump(models["xgboost"], f)
print(" Saved models/saved_models/xgboost_models.pkl")

with open("models/saved_models/catboost_models.pkl", "wb") as f:
    pickle.dump(models["catboost"], f)
print(" Saved models/saved_models/catboost_models.pkl")

# ============================================================================
# GENERATE TEST PREDICTIONS
# ============================================================================

print("\n Generating predictions for test set...")

test_predictions = {
    "lightgbm": np.zeros(len(X_test)),
    "xgboost": np.zeros(len(X_test)),
    "catboost": np.zeros(len(X_test)),
}

# Average predictions across folds
print("\n Averaging predictions from all folds...")

for i, model in enumerate(models["lightgbm"], 1):
    pred_log = model.predict(X_test, num_iteration=model.best_iteration)
    test_predictions["lightgbm"] += np.expm1(pred_log)
test_predictions["lightgbm"] /= CONFIG["n_folds"]

dtest = xgb.DMatrix(X_test)
for i, model in enumerate(models["xgboost"], 1):
    pred_log = model.predict(dtest, iteration_range=(0, model.best_iteration))
    test_predictions["xgboost"] += np.expm1(pred_log)
test_predictions["xgboost"] /= CONFIG["n_folds"]

for i, model in enumerate(models["catboost"], 1):
    pred_log = model.predict(X_test)
    test_predictions["catboost"] += np.expm1(pred_log)
test_predictions["catboost"] /= CONFIG["n_folds"]

# Ensemble (simple average)
test_predictions["ensemble"] = (
    test_predictions["lightgbm"]
    + test_predictions["xgboost"]
    + test_predictions["catboost"]
) / 3

print(" Test predictions generated")

# Save predictions
predictions_df = pd.DataFrame(
    {
        "sample_id": test_ids,
        "lightgbm": test_predictions["lightgbm"],
        "xgboost": test_predictions["xgboost"],
        "catboost": test_predictions["catboost"],
        "ensemble": test_predictions["ensemble"],
    }
)

predictions_df.to_csv("results/test_predictions.csv", index=False)
print("\n Saved results/test_predictions.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print(" BASELINE MODEL TRAINING COMPLETE!")
print("=" * 80)

print("\n Final Cross-Validation Scores:")
print(f"   LightGBM SMAPE: {np.mean(cv_results['lightgbm']['smape']):.4f}")
print(f"   XGBoost SMAPE:  {np.mean(cv_results['xgboost']['smape']):.4f}")
print(f"   CatBoost SMAPE: {np.mean(cv_results['catboost']['smape']):.4f}")

print(f"\n Best Model: {best_model} (SMAPE: {best_smape:.4f})")

print("\nFiles Created:")
print("   ✓ models/saved_models/lightgbm_models.pkl")
print("   ✓ models/saved_models/xgboost_models.pkl")
print("   ✓ models/saved_models/catboost_models.pkl")
print("   ✓ results/cv_scores.csv")
print("   ✓ results/feature_importance.csv")
print("   ✓ results/test_predictions.csv")
print("   ✓ results/visualizations/feature_importance.png")

if best_smape < 0.25:
    print("\n EXCELLENT! SMAPE < 0.25 achieved!")
    print("   Your baseline is competitive!")
elif best_smape < 0.30:
    print("\n GOOD! SMAPE < 0.30 achieved!")
    print("   Consider hyperparameter tuning for improvement")
else:
    print("\n SMAPE > 0.30 - Room for improvement")
    print("   Review feature engineering and model parameters")
