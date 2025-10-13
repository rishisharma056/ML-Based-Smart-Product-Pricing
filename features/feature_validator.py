"""
Feature Quality Validator
Checks the quality and validity of extracted features

Run after feature_engineering.py:
    python phase2_features/feature_validator.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("FEATURE QUALITY VALIDATION")
print("=" * 80)

# ============================================================================
# LOAD FEATURES
# ============================================================================

print("\n📂 Loading features...")
train_features = pd.read_pickle("dataset/train_features.pkl")
test_features = pd.read_pickle("dataset/test_features.pkl")
y_train = np.load("dataset/y_train.npy")

print(f"✅ Train features: {train_features.shape}")
print(f"✅ Test features: {test_features.shape}")
print(f"✅ Target: {y_train.shape}")

# ============================================================================
# CHECK 1: BASIC VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("CHECK 1: BASIC VALIDATION")
print("=" * 80)

# Check for NaN
train_nan = train_features.isna().sum().sum()
test_nan = test_features.isna().sum().sum()

print("\n❓ NaN values:")
print(f"   Train: {train_nan}")
print(f"   Test: {test_nan}")

if train_nan > 0 or test_nan > 0:
    print("   ❌ FAIL: NaN values detected!")
else:
    print("   ✅ PASS: No NaN values")

# Check for inf
train_inf = np.isinf(train_features.select_dtypes(include=[np.number])).sum().sum()
test_inf = np.isinf(test_features.select_dtypes(include=[np.number])).sum().sum()

print("\n❓ Inf values:")
print(f"   Train: {train_inf}")
print(f"   Test: {test_inf}")

if train_inf > 0 or test_inf > 0:
    print("   ❌ FAIL: Inf values detected!")
else:
    print("   ✅ PASS: No inf values")

# Check shapes match
print("\n❓ Feature dimensions:")
print(f"   Train: {train_features.shape[1]} features")
print(f"   Test: {test_features.shape[1]} features")

if train_features.shape[1] == test_features.shape[1]:
    print("   ✅ PASS: Train and test have same features")
else:
    print("   ❌ FAIL: Feature mismatch!")

# Check column names match
if list(train_features.columns) == list(test_features.columns):
    print("   ✅ PASS: Column names match")
else:
    print("   ❌ FAIL: Column names don't match!")

# ============================================================================
# CHECK 2: FEATURE VARIANCE
# ============================================================================

print("\n" + "=" * 80)
print("CHECK 2: FEATURE VARIANCE")
print("=" * 80)

# Check for zero variance features
train_var = train_features.var()
zero_var = train_var[train_var == 0]

print(f"\n❓ Zero variance features: {len(zero_var)}")
if len(zero_var) > 0:
    print("   ⚠️ WARNING: Features with zero variance detected")
    print(f"   Features: {list(zero_var.index)}")
else:
    print("   ✅ PASS: All features have variance")

# Check for low variance features (< 0.01)
low_var = train_var[(train_var > 0) & (train_var < 0.01)]
print(f"\n❓ Low variance features (var < 0.01): {len(low_var)}")
if len(low_var) > 10:
    print(f"   ⚠️ WARNING: {len(low_var)} features have very low variance")
    print(f"   Top 5: {list(low_var.head().index)}")

# ============================================================================
# CHECK 3: FEATURE CORRELATIONS WITH TARGET
# ============================================================================

print("\n" + "=" * 80)
print("CHECK 3: FEATURE-TARGET CORRELATIONS")
print("=" * 80)

print("\n📊 Calculating correlations...")
correlations = {}
for col in train_features.columns:
    corr = np.corrcoef(train_features[col], y_train)[0, 1]
    if not np.isnan(corr):
        correlations[col] = abs(corr)

# Sort by correlation
corr_sorted = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

print("\n🥇 Top 20 Features by Correlation with Price:")
for i, (feat, corr) in enumerate(corr_sorted[:20], 1):
    strength = "🔥" if corr > 0.1 else "✅" if corr > 0.05 else "⚪"
    print(f"{i:2d}. {feat:30s}: {corr:.4f} {strength}")

print("\n📉 Bottom 10 Features by Correlation:")
for i, (feat, corr) in enumerate(corr_sorted[-10:], 1):
    print(f"{i:2d}. {feat:30s}: {corr:.4f}")

# Check if we have good predictors
top_corr = corr_sorted[0][1] if corr_sorted else 0
if top_corr > 0.1:
    print("\n✅ PASS: Have features with correlation > 0.1")
elif top_corr > 0.05:
    print(f"\n⚠️ WARNING: Best correlation is {top_corr:.4f} (weak)")
else:
    print("\n❌ FAIL: No strong correlations found!")

# ============================================================================
# CHECK 4: FEATURE DISTRIBUTIONS
# ============================================================================

print("\n" + "=" * 80)
print("CHECK 4: FEATURE DISTRIBUTIONS")
print("=" * 80)

# Sample features to check
check_features = [
    "text_length",
    "word_count",
    "ipq",
    "has_premium",
    "has_brand",
    "category_encoded",
]

print("\n📊 Distribution statistics:")
for feat in check_features:
    if feat in train_features.columns:
        print(f"\n{feat}:")
        print(f"   Mean: {train_features[feat].mean():.4f}")
        print(f"   Std: {train_features[feat].std():.4f}")
        print(f"   Min: {train_features[feat].min():.4f}")
        print(f"   Max: {train_features[feat].max():.4f}")
        print(f"   Unique values: {train_features[feat].nunique()}")

# ============================================================================
# CHECK 5: TRAIN-TEST DISTRIBUTION SIMILARITY
# ============================================================================

print("\n" + "=" * 80)
print("CHECK 5: TRAIN-TEST DISTRIBUTION SIMILARITY")
print("=" * 80)

print("\n📊 Comparing train vs test distributions...")

# Compare key features
distribution_checks = []
for feat in check_features:
    if feat in train_features.columns:
        # Kolmogorov-Smirnov test
        statistic, pvalue = ks_2samp(train_features[feat], test_features[feat])
        distribution_checks.append(
            {
                "feature": feat,
                "ks_statistic": statistic,
                "p_value": pvalue,
                "similar": pvalue > 0.05,  # Similar if p > 0.05
            }
        )

print("\n   Feature Distribution Comparison:")
print(f"   {'Feature':<25} {'KS Stat':<12} {'P-value':<12} {'Status'}")
print("   " + "-" * 60)
for check in distribution_checks:
    status = "✅ Similar" if check["similar"] else "⚠️ Different"
    print(
        f"   {check['feature']:<25} {check['ks_statistic']:<12.4f} {check['p_value']:<12.4f} {status}"
    )

similar_count = sum(1 for c in distribution_checks if c["similar"])
if similar_count >= len(distribution_checks) * 0.7:
    print(
        f"\n✅ PASS: {similar_count}/{len(distribution_checks)} features have similar distributions"
    )
else:
    print(
        f"\n⚠️ WARNING: Only {similar_count}/{len(distribution_checks)} features similar"
    )

# ============================================================================
# CHECK 6: MULTICOLLINEARITY
# ============================================================================

print("\n" + "=" * 80)
print("CHECK 6: MULTICOLLINEARITY CHECK")
print("=" * 80)

print("\n📊 Checking for highly correlated features...")

# Sample of features to check (skip TF-IDF for speed)
non_tfidf_cols = [col for col in train_features.columns if not col.startswith("tfidf_")]

if len(non_tfidf_cols) <= 50:
    check_cols = non_tfidf_cols
else:
    check_cols = non_tfidf_cols[:50]

corr_matrix = train_features[check_cols].corr().abs()

# Find pairs with correlation > 0.9
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.9:
            high_corr_pairs.append(
                {
                    "feature1": corr_matrix.columns[i],
                    "feature2": corr_matrix.columns[j],
                    "correlation": corr_matrix.iloc[i, j],
                }
            )

if len(high_corr_pairs) > 0:
    print(f"\n⚠️ WARNING: Found {len(high_corr_pairs)} highly correlated pairs (>0.9)")
    print("\n   Top 5 pairs:")
    for pair in high_corr_pairs[:5]:
        print(
            f"   - {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}"
        )
    print("\n   Consider removing one from each pair")
else:
    print("\n✅ PASS: No severe multicollinearity detected")

# ============================================================================
# CHECK 7: FEATURE SCALING
# ============================================================================

print("\n" + "=" * 80)
print("CHECK 7: FEATURE SCALING CHECK")
print("=" * 80)

print("\n📊 Checking feature scales...")

# Check if features are on very different scales
feature_ranges = []
for col in check_features:
    if col in train_features.columns:
        feat_range = train_features[col].max() - train_features[col].min()
        feature_ranges.append(
            {"feature": col, "range": feat_range, "mean": train_features[col].mean()}
        )

print("\n   Feature Ranges:")
for item in feature_ranges:
    print(
        f"   {item['feature']:<25} Range: {item['range']:<15.2f} Mean: {item['mean']:.2f}"
    )

max_range = max(item["range"] for item in feature_ranges)
min_range = min(item["range"] for item in feature_ranges if item["range"] > 0)

if max_range / min_range > 1000:
    print("\n⚠️ WARNING: Large scale differences detected")
    print(f"   Ratio: {max_range / min_range:.0f}x")
    print("   Consider scaling for some models (not needed for tree-based)")
else:
    print("\n✅ PASS: Feature scales are reasonable for tree-based models")

# ============================================================================
# CHECK 8: VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("CHECK 8: GENERATING VALIDATION PLOTS")
print("=" * 80)

print("\n📊 Creating validation visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Top feature correlations
top_20_feats = [f[0] for f in corr_sorted[:20]]
top_20_corrs = [f[1] for f in corr_sorted[:20]]

axes[0, 0].barh(range(20), top_20_corrs[::-1], color="skyblue", edgecolor="black")
axes[0, 0].set_yticks(range(20))
axes[0, 0].set_yticklabels([f[:25] for f in top_20_feats[::-1]], fontsize=8)
axes[0, 0].set_xlabel("Absolute Correlation with Price")
axes[0, 0].set_title("Top 20 Features by Correlation", fontweight="bold")
axes[0, 0].grid(axis="x", alpha=0.3)

# Plot 2: Feature variance distribution
non_zero_var = train_var[train_var > 0]
axes[0, 1].hist(np.log10(non_zero_var), bins=30, color="lightcoral", edgecolor="black")
axes[0, 1].set_xlabel("Log10(Variance)")
axes[0, 1].set_ylabel("Number of Features")
axes[0, 1].set_title("Feature Variance Distribution", fontweight="bold")
axes[0, 1].grid(axis="y", alpha=0.3)

# Plot 3: Sample feature distributions (train vs test)
if "text_length" in train_features.columns:
    axes[1, 0].hist(
        train_features["text_length"],
        bins=50,
        alpha=0.5,
        label="Train",
        color="blue",
        edgecolor="black",
    )
    axes[1, 0].hist(
        test_features["text_length"],
        bins=50,
        alpha=0.5,
        label="Test",
        color="red",
        edgecolor="black",
    )
    axes[1, 0].set_xlabel("Text Length")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Text Length Distribution (Train vs Test)", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

# Plot 4: Category distribution
if "category_encoded" in train_features.columns:
    cat_counts_train = train_features["category_encoded"].value_counts().sort_index()
    cat_counts_test = test_features["category_encoded"].value_counts().sort_index()

    x = np.arange(len(cat_counts_train))
    width = 0.35

    axes[1, 1].bar(
        x - width / 2,
        cat_counts_train.values,
        width,
        label="Train",
        color="green",
        alpha=0.7,
        edgecolor="black",
    )
    axes[1, 1].bar(
        x + width / 2,
        cat_counts_test.values,
        width,
        label="Test",
        color="orange",
        alpha=0.7,
        edgecolor="black",
    )
    axes[1, 1].set_xlabel("Category")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Category Distribution (Train vs Test)", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "results/visualizations/feature_validation.png", dpi=300, bbox_inches="tight"
)
plt.close()

print("✅ Saved results/visualizations/feature_validation.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

checks_passed = 0
total_checks = 8

print("\n✅ Quality Checks:")

# Check 1: No NaN/Inf
if train_nan == 0 and test_nan == 0 and train_inf == 0 and test_inf == 0:
    print("   ✅ No NaN or Inf values")
    checks_passed += 1
else:
    print("   ❌ Data quality issues detected")

# Check 2: Feature count
if train_features.shape[1] == test_features.shape[1]:
    print("   ✅ Feature dimensions match")
    checks_passed += 1
else:
    print("   ❌ Feature dimension mismatch")

# Check 3: Zero variance
if len(zero_var) == 0:
    print("   ✅ All features have variance")
    checks_passed += 1
else:
    print(f"   ⚠️ {len(zero_var)} features with zero variance")

# Check 4: Correlations
if top_corr > 0.1:
    print(f"   ✅ Strong predictors found (max corr: {top_corr:.4f})")
    checks_passed += 1
elif top_corr > 0.05:
    print(f"   ⚠️ Moderate predictors (max corr: {top_corr:.4f})")
    checks_passed += 0.5
else:
    print(f"   ❌ Weak correlations (max corr: {top_corr:.4f})")

# Check 5: Train-test similarity
if similar_count >= len(distribution_checks) * 0.7:
    print("   ✅ Train-test distributions similar")
    checks_passed += 1
else:
    print("   ⚠️ Train-test distribution differences detected")

# Check 6: Multicollinearity
if len(high_corr_pairs) < 5:
    print("   ✅ Low multicollinearity")
    checks_passed += 1
else:
    print(f"   ⚠️ {len(high_corr_pairs)} highly correlated pairs")

# Check 7: Feature scales
if max_range / min_range < 1000:
    print("   ✅ Reasonable feature scales")
    checks_passed += 1
else:
    print("   ⚠️ Large scale differences")

# Check 8: Visualization
print("   ✅ Validation plots generated")
checks_passed += 1

print(f"\n📊 Overall Score: {checks_passed}/{total_checks} checks passed")

if checks_passed >= 7:
    print("\n🎉 VALIDATION PASSED - Features are ready for modeling!")
    print("\n🚀 Next Steps:")
    print("   1. cd phase3_modeling")
    print("   2. python train_baseline.py")
    print("   3. Start model training")
elif checks_passed >= 5:
    print("\n⚠️ VALIDATION PASSED WITH WARNINGS")
    print("   Features are usable but could be improved")
    print("\n🚀 Next Steps:")
    print("   1. Review warnings above")
    print("   2. Proceed with caution to modeling")
else:
    print("\n❌ VALIDATION FAILED")
    print("   Please fix issues before modeling")
    print("\n🔧 Recommended Actions:")
    print("   1. Review failed checks")
    print("   2. Fix feature engineering pipeline")
    print("   3. Re-run feature_engineering.py")

print("\n" + "=" * 80)
