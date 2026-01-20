import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import pickle
import warnings
from time import time

warnings.filterwarnings("ignore")

print("=" * 80)
print("STEP 3: TRAINING WITH CLIP FEATURES")
print("=" * 80)

# ============================================================================
# SMAPE METRIC
# ============================================================================


def calculate_smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(numerator / denominator)
    return smape


# ============================================================================
# LOAD DATA
# ============================================================================

print("\n Loading CLIP features...")
X_train = pd.read_pickle("../dataset/train_features_clip.pkl")
X_test = pd.read_pickle("../dataset/test_features_clip.pkl")
y_train = np.load("../dataset/y_train.npy")
test_ids = np.load("../dataset/test_ids.npy")

print(f" Train: {X_train.shape} | Test: {X_test.shape}")
print("   Features breakdown:")

# Count feature types
n_original = len([c for c in X_train.columns if not c.startswith("clip_")])
n_clip_text = len([c for c in X_train.columns if c.startswith("clip_text_")])
n_clip_image = len([c for c in X_train.columns if c.startswith("clip_image_")])

print(f"   - Original features: {n_original}")
print(f"   - CLIP text: {n_clip_text}")
print(f"   - CLIP image: {n_clip_image}")

# Log transform target
y_train_log = np.log1p(y_train)
print("\n Target log-transformed")

# ============================================================================
# MODEL CONFIG
# ============================================================================

CONFIG = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.03,  # Lower learning rate
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_estimators": 2000,
}

print("\n Model Config:")
print(f"   num_leaves: {CONFIG['num_leaves']}")
print(f"   learning_rate: {CONFIG['learning_rate']}")
print(f"   n_estimators: {CONFIG['n_estimators']}")

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("5-FOLD CROSS-VALIDATION")
print("=" * 80)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_smape = []
cv_mae = []
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\n{'=' * 60}")
    print(f"Fold {fold}/5")
    print(f"{'=' * 60}")

    fold_start = time()

    # Split
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_log[train_idx], y_train_log[val_idx]
    y_val_original = y_train[val_idx]

    print(f"Train: {len(X_tr):,} | Val: {len(X_val):,}")

    # Create datasets
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train
    print(" Training with CLIP features...")
    model = lgb.train(
        CONFIG,
        train_data,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200),
        ],
    )

    # Predict
    y_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = np.expm1(y_pred_log)

    # Metrics
    smape = calculate_smape(y_val_original, y_pred)
    mae = mean_absolute_error(y_val_original, y_pred)

    fold_time = time() - fold_start

    cv_smape.append(smape)
    cv_mae.append(mae)
    models.append(model)

    print(f"\n Fold {fold} Results:")
    print(f"   SMAPE: {smape:.4f}")
    print(f"   MAE: ${mae:.2f}")
    print(f"   Pred mean: ${y_pred.mean():.2f} (actual: ${y_val_original.mean():.2f})")
    print(f"   Time: {fold_time:.1f}s")

# Summary
print(f"\n{'=' * 60}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'=' * 60}")
print(f"Average SMAPE: {np.mean(cv_smape):.4f} ± {np.std(cv_smape):.4f}")
print(f"Average MAE: ${np.mean(cv_mae):.2f} ± ${np.std(cv_mae):.2f}")

# Compare with text-only baseline
baseline_smape = 0.5674  # Your V1 score
improvement = (baseline_smape - np.mean(cv_smape)) / baseline_smape * 100

print("\n vs Text-Only Baseline:")
print(f"   Baseline SMAPE: {baseline_smape:.4f}")
print(f"   CLIP SMAPE: {np.mean(cv_smape):.4f}")
print(f"   Improvement: {improvement:.1f}%")

if np.mean(cv_smape) < 0.45:
    print("\n EXCELLENT! SMAPE < 0.45")
elif np.mean(cv_smape) < 0.50:
    print("\n GOOD! SMAPE < 0.50 - Significant improvement!")
elif np.mean(cv_smape) < baseline_smape:
    print("\n IMPROVED over baseline!")
else:
    print("\n  Not better than baseline - may need tuning")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

# Average importance
feature_importance = np.zeros(X_train.shape[1])
for model in models:
    feature_importance += model.feature_importance(importance_type="gain")
feature_importance /= len(models)

# Create DataFrame
importance_df = pd.DataFrame(
    {"feature": X_train.columns, "importance": feature_importance}
).sort_values("importance", ascending=False)

print("\n Top 30 Features:")
print(importance_df.head(30).to_string(index=False))

# Check CLIP feature importance
clip_text_importance = importance_df[
    importance_df["feature"].str.startswith("clip_text_")
]["importance"].sum()
clip_image_importance = importance_df[
    importance_df["feature"].str.startswith("clip_image_")
]["importance"].sum()
original_importance = importance_df[~importance_df["feature"].str.startswith("clip_")][
    "importance"
].sum()

total_importance = importance_df["importance"].sum()

print("\n Feature Group Importance:")
print(f"   Original features: {original_importance / total_importance * 100:.1f}%")
print(f"   CLIP text: {clip_text_importance / total_importance * 100:.1f}%")
print(f"   CLIP image: {clip_image_importance / total_importance * 100:.1f}%")

# ============================================================================
# GENERATE TEST PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING TEST PREDICTIONS")
print("=" * 80)

test_predictions = np.zeros(len(X_test))

print("\n Averaging predictions from all folds...")
for i, model in enumerate(models, 1):
    pred_log = model.predict(X_test, num_iteration=model.best_iteration)
    test_predictions += np.expm1(pred_log)

test_predictions /= len(models)

print(" Test predictions generated")
print("\n Prediction Statistics:")
print(f"   Mean: ${test_predictions.mean():.2f}")
print(f"   Median: ${np.median(test_predictions):.2f}")
print(f"   Min: ${test_predictions.min():.2f}")
print(f"   Max: ${test_predictions.max():.2f}")

# Compare with training distribution
print("\n vs Training Data:")
print(f"   Train mean: ${y_train.mean():.2f}")
print(f"   Test mean: ${test_predictions.mean():.2f}")
print(
    f"   Difference: {abs(y_train.mean() - test_predictions.mean()) / y_train.mean() * 100:.1f}%"
)

# Save predictions
predictions_df = pd.DataFrame({"sample_id": test_ids, "price": test_predictions})
predictions_df.to_csv("results/test_predictions_clip.csv", index=False)
print("\n Saved: results/test_predictions_clip.csv")

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n Saving models...")
os.makedirs("models/saved_models", exist_ok=True)

with open("models/saved_models/lightgbm_models_clip.pkl", "wb") as f:
    pickle.dump(models, f)
print(" Saved: models/saved_models/lightgbm_models_clip.pkl")

# Save feature importance
importance_df.to_csv("results/feature_importance_clip.csv", index=False)
print(" Saved: results/feature_importance_clip.csv")

# ============================================================================
# CREATE SUBMISSION FILE
# ============================================================================

print("\n" + "=" * 80)
print("CREATING SUBMISSION FILE")
print("=" * 80)

# Validate
test = pd.read_csv("dataset/test.csv")
assert len(predictions_df) == len(test), "Sample count mismatch!"
assert set(predictions_df["sample_id"]) == set(test["sample_id"]), "ID mismatch!"
assert (predictions_df["price"] > 0).all(), "Negative prices!"
assert not predictions_df["price"].isna().any(), "NaN values!"

print(" All validation checks passed")

# Create submission
submission = predictions_df[["sample_id", "price"]].sort_values("sample_id")
submission.to_csv("test_out_clip.csv", index=False)

print("\n Created: test_out_clip.csv")
print("\n Submission Preview:")
print(submission.head(10))

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print(" CLIP MODEL TRAINING COMPLETE!")
print("=" * 80)

print("\n Final Results:")
print(f"   Cross-Validation SMAPE: {np.mean(cv_smape):.4f}")
print(f"   Cross-Validation MAE: ${np.mean(cv_mae):.2f}")
print(f"   Total features: {X_train.shape[1]}")
print(f"   Training time: {sum([t for t in [fold_time] * 5]):.0f}s")

print("\n Files Created:")
print("   ✓ results/test_predictions_clip.csv")
print("   ✓ results/feature_importance_clip.csv")
print("   ✓ models/saved_models/lightgbm_models_clip.pkl")
print("   ✓ test_out_clip.csv (READY TO SUBMIT)")

print("\n Expected Leaderboard Performance:")
if np.mean(cv_smape) < 0.40:
    print("    Top 20-30% likely!")
    print("   Your CV SMAPE < 0.40 is competitive")
elif np.mean(cv_smape) < 0.45:
    print("    Top 30-40% likely")
    print("   Solid performance with multimodal features")
elif np.mean(cv_smape) < 0.50:
    print("    Better than text-only baseline")
    print("   Room for improvement with hyperparameter tuning")
else:
    print("     Similar to baseline - may need investigation")

print("\n" + "=" * 80)
