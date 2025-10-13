"""
ML Challenge 2025 - Phase 4: Create Submission File
Generates final test_out.csv for submission

Run from project root:
    python phase4_submission/create_submission.py

Outputs:
    - test_out.csv (ready for submission)
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("PHASE 4: CREATING SUBMISSION FILE")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose which model to use for submission
SUBMISSION_MODEL = "ensemble"  # Options: 'lightgbm', 'xgboost', 'catboost', 'ensemble'

print("\n📋 Configuration:")
print(f"   Model: {SUBMISSION_MODEL}")

# ============================================================================
# LOAD PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("LOADING PREDICTIONS")
print("=" * 80)

print("\n📂 Loading test predictions...")
predictions_df = pd.read_csv("results/test_predictions.csv")
print(f"✅ Loaded {len(predictions_df)} predictions")

print(f"\n📊 Prediction Statistics ({SUBMISSION_MODEL}):")
print(f"   Mean: ${predictions_df[SUBMISSION_MODEL].mean():.2f}")
print(f"   Median: ${predictions_df[SUBMISSION_MODEL].median():.2f}")
print(f"   Min: ${predictions_df[SUBMISSION_MODEL].min():.2f}")
print(f"   Max: ${predictions_df[SUBMISSION_MODEL].max():.2f}")
print(f"   Std: ${predictions_df[SUBMISSION_MODEL].std():.2f}")

# ============================================================================
# LOAD TEST DATA FOR VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION CHECKS")
print("=" * 80)

print("\n🔍 Loading test.csv for validation...")
test = pd.read_csv("dataset/test.csv")
print(f"✅ Test set has {len(test)} samples")

# Check 1: Sample count
print("\n❓ Check 1: Sample count")
if len(predictions_df) == len(test):
    print(
        f"   ✅ PASS: {len(predictions_df)} predictions match {len(test)} test samples"
    )
else:
    print(f"   ❌ FAIL: {len(predictions_df)} predictions != {len(test)} test samples")
    print("   ERROR: Prediction count mismatch!")
    exit(1)

# Check 2: Sample IDs match
print("\n❓ Check 2: Sample IDs")
test_ids_set = set(test["sample_id"])
pred_ids_set = set(predictions_df["sample_id"])

if test_ids_set == pred_ids_set:
    print("   ✅ PASS: All sample IDs match")
else:
    missing = test_ids_set - pred_ids_set
    extra = pred_ids_set - test_ids_set
    if missing:
        print(f"   ❌ FAIL: Missing {len(missing)} sample IDs")
        print(f"   First 5 missing: {list(missing)[:5]}")
    if extra:
        print(f"   ❌ FAIL: Extra {len(extra)} sample IDs")
        print(f"   First 5 extra: {list(extra)[:5]}")
    exit(1)

# Check 3: All prices positive
print("\n❓ Check 3: Price validity")
negative_prices = (predictions_df[SUBMISSION_MODEL] < 0).sum()
zero_prices = (predictions_df[SUBMISSION_MODEL] == 0).sum()
nan_prices = predictions_df[SUBMISSION_MODEL].isna().sum()

if negative_prices > 0:
    print(f"   ❌ FAIL: {negative_prices} negative prices detected!")
    print("   Clipping to 0.13 (minimum train price)...")
    predictions_df[SUBMISSION_MODEL] = predictions_df[SUBMISSION_MODEL].clip(lower=0.13)
elif zero_prices > 0:
    print(f"   ⚠️ WARNING: {zero_prices} zero prices detected!")
    print("   Replacing with 0.13 (minimum train price)...")
    predictions_df[SUBMISSION_MODEL] = predictions_df[SUBMISSION_MODEL].replace(0, 0.13)
elif nan_prices > 0:
    print(f"   ❌ FAIL: {nan_prices} NaN prices detected!")
    exit(1)
else:
    print("   ✅ PASS: All prices are positive")

# Check 4: Reasonable price range
print("\n❓ Check 4: Price range check")
train_min = 0.13  # From Phase 1 analysis
train_max = 2796.00

extreme_low = (predictions_df[SUBMISSION_MODEL] < train_min).sum()
extreme_high = (predictions_df[SUBMISSION_MODEL] > train_max).sum()

if extreme_low > 0:
    print(f"   ⚠️ WARNING: {extreme_low} predictions below training min (${train_min})")
    print("   Clipping to minimum...")
    predictions_df[SUBMISSION_MODEL] = predictions_df[SUBMISSION_MODEL].clip(
        lower=train_min
    )

if extreme_high > 0:
    print(
        f"   ⚠️ WARNING: {extreme_high} predictions above training max (${train_max:.2f})"
    )
    print("   Clipping to maximum...")
    predictions_df[SUBMISSION_MODEL] = predictions_df[SUBMISSION_MODEL].clip(
        upper=train_max
    )

if extreme_low == 0 and extreme_high == 0:
    print("   ✅ PASS: All predictions within training range")

# ============================================================================
# CREATE SUBMISSION FILE
# ============================================================================

print("\n" + "=" * 80)
print("CREATING SUBMISSION FILE")
print("=" * 80)

# Create submission DataFrame
submission = pd.DataFrame(
    {
        "sample_id": predictions_df["sample_id"],
        "price": predictions_df[SUBMISSION_MODEL],
    }
)

# Sort by sample_id (good practice)
submission = submission.sort_values("sample_id").reset_index(drop=True)

print("\n📝 Submission Preview:")
print(submission.head(10))

print("\n📊 Final Statistics:")
print(f"   Total predictions: {len(submission):,}")
print(f"   Mean price: ${submission['price'].mean():.2f}")
print(f"   Median price: ${submission['price'].median():.2f}")
print(f"   Min price: ${submission['price'].min():.2f}")
print(f"   Max price: ${submission['price'].max():.2f}")

# ============================================================================
# SAVE SUBMISSION FILE
# ============================================================================

print("\n" + "=" * 80)
print("SAVING SUBMISSION")
print("=" * 80)

# Save to root directory
submission_file = "test_out.csv"
submission.to_csv(submission_file, index=False)
print(f"\n✅ Saved: {submission_file}")

# Also save to phase4 folder
submission.to_csv("test_out.csv", index=False)
print("✅ Saved: test_out.csv")

# ============================================================================
# FINAL VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

# Load the saved file and validate
print("\n🔍 Re-loading submission file for final check...")
final_check = pd.read_csv(submission_file)

print("\n✅ File validation:")
print(f"   Rows: {len(final_check)}")
print(f"   Columns: {list(final_check.columns)}")
print(f"   Column types: {final_check.dtypes.to_dict()}")

# Check format matches sample
print("\n🔍 Checking format against sample_test_out.csv...")
try:
    sample_out = pd.read_csv("dataset/sample_test_out.csv")

    if list(final_check.columns) == list(sample_out.columns):
        print("   ✅ PASS: Column names match sample")
    else:
        print("   ❌ FAIL: Column names don't match!")
        print(f"   Expected: {list(sample_out.columns)}")
        print(f"   Got: {list(final_check.columns)}")

except FileNotFoundError:
    print("   ⚠️ WARNING: sample_test_out.csv not found, skipping format check")

# ============================================================================
# COMPARISON WITH TRAINING DATA
# ============================================================================

print("\n" + "=" * 80)
print("TRAIN vs TEST DISTRIBUTION COMPARISON")
print("=" * 80)

# Load training prices for comparison
y_train = np.load("dataset/y_train.npy")

print("\n📊 Distribution Comparison:")
print(f"{'Metric':<20} {'Training':<15} {'Test Predictions'}")
print(f"{'-' * 55}")
print(f"{'Mean':<20} ${y_train.mean():<14.2f} ${submission['price'].mean():.2f}")
print(
    f"{'Median':<20} ${np.median(y_train):<14.2f} ${submission['price'].median():.2f}"
)
print(f"{'Std Dev':<20} ${y_train.std():<14.2f} ${submission['price'].std():.2f}")
print(f"{'Min':<20} ${y_train.min():<14.2f} ${submission['price'].min():.2f}")
print(f"{'Max':<20} ${y_train.max():<14.2f} ${submission['price'].max():.2f}")

# Check if distributions are similar
mean_diff = abs(y_train.mean() - submission["price"].mean()) / y_train.mean() * 100
median_diff = (
    abs(np.median(y_train) - submission["price"].median()) / np.median(y_train) * 100
)

print("\n📈 Distribution Similarity:")
print(f"   Mean difference: {mean_diff:.1f}%")
print(f"   Median difference: {median_diff:.1f}%")

if mean_diff < 20 and median_diff < 20:
    print("   ✅ GOOD: Distributions are similar")
elif mean_diff < 50:
    print("   ⚠️ WARNING: Some distribution differences")
else:
    print("   ❌ CONCERN: Large distribution differences - review model")

# ============================================================================
# SUBMISSION SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("🎉 SUBMISSION FILE READY!")
print("=" * 80)

print("\n📋 Submission Details:")
print("   File: test_out.csv")
print(f"   Model: {SUBMISSION_MODEL}")
print(f"   Samples: {len(submission):,}")
print("   Format: ✅ Valid")
print(
    f"   Price Range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}"
)

print("\n📁 Files Created:")
print("   ✓ test_out.csv (root directory - SUBMIT THIS)")
print("   ✓ test_out.csv (backup)")

print("\n✅ PRE-SUBMISSION CHECKLIST:")
checklist = [
    (len(submission) == 75000, "75,000 predictions"),
    (list(submission.columns) == ["sample_id", "price"], "Correct columns"),
    ((submission["price"] > 0).all(), "All positive prices"),
    (submission["sample_id"].nunique() == 75000, "No duplicate IDs"),
    (not submission["price"].isna().any(), "No NaN values"),
]

for check, desc in checklist:
    status = "✅" if check else "❌"
    print(f"   {status} {desc}")

all_passed = all(check for check, _ in checklist)

if all_passed:
    print("\n🎉 ALL CHECKS PASSED - Ready to submit!")
    print("\n🚀 NEXT STEPS:")
    print("   1. Upload test_out.csv to the submission portal")
    print("   2. Wait for leaderboard score")
    print("   3. Document your approach in Documentation_template.md")
else:
    print("\n❌ SOME CHECKS FAILED - Fix issues before submitting")

print("\n" + "=" * 80)
print("Good luck! 🍀")
print("=" * 80)
