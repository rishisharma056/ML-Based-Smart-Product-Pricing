import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

# Load training data
train = pd.read_csv("../dataset/train.csv")
test = pd.read_csv("../dataset/test.csv")

print(f"\nTraining Set Shape: {train.shape}")
print(f"Test Set Shape: {test.shape}")

# Display first few rows
print("\n" + "=" * 80)
print("First 5 rows of training data:")
print("=" * 80)
print(train.head())

# ============================================================================
# 2. BASIC DATA INSPECTION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: BASIC DATA INSPECTION")
print("=" * 80)

print("\n--- Column Information ---")
print(train.info())

print("\n--- Statistical Summary ---")
print(train.describe())

print("\n--- Missing Values ---")
missing = train.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

print("\n--- Data Types ---")
print(train.dtypes)

# ============================================================================
# 3. PRICE DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: PRICE DISTRIBUTION ANALYSIS")
print("=" * 80)

# Basic price statistics
print("\n--- Price Statistics ---")
print(f"Mean Price: ${train['price'].mean():.2f}")
print(f"Median Price: ${train['price'].median():.2f}")
print(f"Std Dev: ${train['price'].std():.2f}")
print(f"Min Price: ${train['price'].min():.2f}")
print(f"Max Price: ${train['price'].max():.2f}")
print(f"25th Percentile: ${train['price'].quantile(0.25):.2f}")
print(f"75th Percentile: ${train['price'].quantile(0.75):.2f}")
print(f"95th Percentile: ${train['price'].quantile(0.95):.2f}")
print(f"99th Percentile: ${train['price'].quantile(0.99):.2f}")

# Check for skewness
skewness = train["price"].skew()
print(f"\nPrice Skewness: {skewness:.2f}")
if skewness > 1:
    print("Highly right-skewed distribution - consider log transformation")
elif skewness > 0.5:
    print("Moderately right-skewed distribution")

# Visualize price distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Original distribution
axes[0, 0].hist(train["price"], bins=100, color="skyblue", edgecolor="black")
axes[0, 0].set_title("Price Distribution (Original)", fontsize=14, fontweight="bold")
axes[0, 0].set_xlabel("Price ($)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].axvline(
    train["price"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: ${train['price'].mean():.2f}",
)
axes[0, 0].axvline(
    train["price"].median(),
    color="green",
    linestyle="--",
    label=f"Median: ${train['price'].median():.2f}",
)
axes[0, 0].legend()

# Log-transformed distribution
axes[0, 1].hist(
    np.log1p(train["price"]), bins=100, color="lightcoral", edgecolor="black"
)
axes[0, 1].set_title(
    "Price Distribution (Log-Transformed)", fontsize=14, fontweight="bold"
)
axes[0, 1].set_xlabel("Log(Price + 1)")
axes[0, 1].set_ylabel("Frequency")

# Box plot
axes[1, 0].boxplot(train["price"], vert=True)
axes[1, 0].set_title(
    "Price Box Plot (Outlier Detection)", fontsize=14, fontweight="bold"
)
axes[1, 0].set_ylabel("Price ($)")

# Price ranges
price_ranges = pd.cut(
    train["price"],
    bins=[0, 10, 25, 50, 100, 200, 500, 1000, float("inf")],
    labels=[
        "$0-10",
        "$10-25",
        "$25-50",
        "$50-100",
        "$100-200",
        "$200-500",
        "$500-1000",
        "$1000+",
    ],
)
price_range_counts = price_ranges.value_counts().sort_index()
axes[1, 1].bar(
    range(len(price_range_counts)),
    price_range_counts.values,
    color="lightgreen",
    edgecolor="black",
)
axes[1, 1].set_xticks(range(len(price_range_counts)))
axes[1, 1].set_xticklabels(price_range_counts.index, rotation=45)
axes[1, 1].set_title("Products by Price Range", fontsize=14, fontweight="bold")
axes[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("../results/price_distribution_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nPrice distribution plots saved as 'price_distribution_analysis.png'")

# ============================================================================
# 4. TEXT ANALYSIS - CATALOG CONTENT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: CATALOG CONTENT TEXT ANALYSIS")
print("=" * 80)

# Text length analysis
train["text_length"] = train["catalog_content"].str.len()
train["word_count"] = train["catalog_content"].str.split().str.len()

print("\n--- Text Statistics ---")
print(f"Average Text Length: {train['text_length'].mean():.2f} characters")
print(f"Median Text Length: {train['text_length'].median():.2f} characters")
print(f"Average Word Count: {train['word_count'].mean():.2f} words")
print(f"Min Text Length: {train['text_length'].min()} characters")
print(f"Max Text Length: {train['text_length'].max()} characters")

# Correlation between text features and price
print("\n--- Correlation with Price ---")
print(f"Text Length vs Price: {train['text_length'].corr(train['price']):.3f}")
print(f"Word Count vs Price: {train['word_count'].corr(train['price']):.3f}")


# Extract Item Pack Quantity (IPQ)
def extract_ipq(text):
    """Extract Item Pack Quantity from catalog content"""
    if pd.isna(text):
        return 1

    text_lower = str(text).lower()

    patterns = [
        r"pack of (\d{1,2})\b",
        r"\b(\d{1,2})\s*pack\b",
        r"\b(\d{1,2})-pack\b",
        r"set of (\d{1,2})\b",
        r"\b(\d{1,2})\s*count\b",
        r"item pack quantity:\s*(\d{1,3})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 100:
                return qty

    return 1


train["ipq"] = train["catalog_content"].apply(extract_ipq)

print("\n--- Item Pack Quantity (IPQ) Analysis ---")
print(
    f"Products with IPQ > 1: {(train['ipq'] > 1).sum()} ({(train['ipq'] > 1).sum() / len(train) * 100:.1f}%)"
)
print(f"Average IPQ: {train['ipq'].mean():.2f}")
print(f"Max IPQ: {train['ipq'].max()}")
print("\nIPQ Distribution:")
print(train["ipq"].value_counts().head(10))

# IPQ vs Price correlation
print(f"\nIPQ vs Price Correlation: {train['ipq'].corr(train['price']):.3f}")


# Extract brand indicators
def has_brand_keywords(text):
    """Check if text contains brand-like patterns"""
    if pd.isna(text):
        return 0

    text = str(text).lower()

    # Common brand indicators
    brand_keywords = [
        "brand",
        "by ",
        "amazon",
        "basics",
        "premium",
        "pro",
        "professional",
        "deluxe",
        "ultra",
        "plus",
        "max",
        "elite",
    ]

    return int(any(keyword in text for keyword in brand_keywords))


train["has_brand"] = train["catalog_content"].apply(has_brand_keywords)

print("\n--- Brand Presence Analysis ---")
print(
    f"Products with brand indicators: {train['has_brand'].sum()} ({train['has_brand'].sum() / len(train) * 100:.1f}%)"
)
print(f"Avg price WITH brand: ${train[train['has_brand'] == 1]['price'].mean():.2f}")
print(f"Avg price WITHOUT brand: ${train[train['has_brand'] == 0]['price'].mean():.2f}")
print(
    f"Price difference: {((train[train['has_brand'] == 1]['price'].mean() / train[train['has_brand'] == 0]['price'].mean() - 1) * 100):.1f}%"
)

# Visualize text features
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Text length distribution
axes[0, 0].hist(train["text_length"], bins=50, color="skyblue", edgecolor="black")
axes[0, 0].set_title("Text Length Distribution", fontsize=14, fontweight="bold")
axes[0, 0].set_xlabel("Number of Characters")
axes[0, 0].set_ylabel("Frequency")

# Word count distribution
axes[0, 1].hist(train["word_count"], bins=50, color="lightcoral", edgecolor="black")
axes[0, 1].set_title("Word Count Distribution", fontsize=14, fontweight="bold")
axes[0, 1].set_xlabel("Number of Words")
axes[0, 1].set_ylabel("Frequency")

# IPQ distribution
ipq_counts = train["ipq"].value_counts().head(15).sort_index()
axes[1, 0].bar(
    range(len(ipq_counts)), ipq_counts.values, color="lightgreen", edgecolor="black"
)
axes[1, 0].set_xticks(range(len(ipq_counts)))
axes[1, 0].set_xticklabels(ipq_counts.index)
axes[1, 0].set_title("Item Pack Quantity Distribution", fontsize=14, fontweight="bold")
axes[1, 0].set_xlabel("IPQ Value")
axes[1, 0].set_ylabel("Count")

# Price by brand presence
brand_prices = train.groupby("has_brand")["price"].mean()
axes[1, 1].bar(
    ["No Brand", "Has Brand"],
    brand_prices.values,
    color=["lightcoral", "lightgreen"],
    edgecolor="black",
)
axes[1, 1].set_title("Average Price by Brand Presence", fontsize=14, fontweight="bold")
axes[1, 1].set_ylabel("Average Price ($)")
for i, v in enumerate(brand_prices.values):
    axes[1, 1].text(i, v + 5, f"${v:.2f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("../results/text_features_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nText analysis plots saved as 'text_features_analysis.png'")

# ============================================================================
# 5. KEYWORD ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: KEYWORD ANALYSIS")
print("=" * 80)

# Extract common words
all_words = []
for text in train["catalog_content"].dropna():
    words = str(text).lower().split()
    # Filter out very short words
    words = [w for w in words if len(w) > 3]
    all_words.extend(words)

word_freq = Counter(all_words)
print("\n--- Top 30 Most Common Keywords ---")
for word, count in word_freq.most_common(30):
    print(f"{word:20s}: {count:6d}")

# Premium keywords analysis
premium_keywords = [
    "premium",
    "professional",
    "pro",
    "deluxe",
    "ultra",
    "elite",
    "luxury",
    "advanced",
    "superior",
    "quality",
]
budget_keywords = ["basic", "standard", "value", "economy", "simple", "generic"]

train["has_premium_keyword"] = (
    train["catalog_content"]
    .str.lower()
    .str.contains("|".join(premium_keywords), na=False)
    .astype(int)
)
train["has_budget_keyword"] = (
    train["catalog_content"]
    .str.lower()
    .str.contains("|".join(budget_keywords), na=False)
    .astype(int)
)

print("\n--- Premium vs Budget Keywords ---")
print(f"Products with premium keywords: {train['has_premium_keyword'].sum()}")
print(
    f"Avg price with premium keywords: ${train[train['has_premium_keyword'] == 1]['price'].mean():.2f}"
)
print(f"Products with budget keywords: {train['has_budget_keyword'].sum()}")
print(
    f"Avg price with budget keywords: ${train[train['has_budget_keyword'] == 0]['price'].mean():.2f}"
)

# ============================================================================
# 6. IMAGE LINK ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: IMAGE LINK ANALYSIS")
print("=" * 80)

print("\n--- Image Link Statistics ---")
print(f"Total image links: {train['image_link'].notna().sum()}")
print(f"Missing image links: {train['image_link'].isna().sum()}")

# Sample image links
print("\n--- Sample Image Links ---")
for i, link in enumerate(train["image_link"].head(5)):
    print(f"{i + 1}. {link}")

print("\nNote: Use src/utils.py download_images() function to download images")
print(
    "   Image features can be extracted using pre-trained models (ResNet, EfficientNet, etc.)"
)

# ============================================================================
# 7. IPQ vs PRICE DETAILED ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: IPQ vs PRICE DETAILED ANALYSIS")
print("=" * 80)

ipq_price = train.groupby("ipq")["price"].agg(["mean", "median", "count"]).round(2)
ipq_price = ipq_price[ipq_price["count"] >= 10].head(
    20
)  # Filter for sufficient samples

print("\n--- Average Price by IPQ ---")
print(ipq_price)

# Plot IPQ vs Price
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(train["ipq"], train["price"], alpha=0.3, s=10)
plt.xlabel("Item Pack Quantity (IPQ)", fontsize=12)
plt.ylabel("Price ($)", fontsize=12)
plt.title("IPQ vs Price Scatter Plot", fontsize=14, fontweight="bold")
plt.xlim(0, 50)  # Limit x-axis for better visualization

plt.subplot(1, 2, 2)
ipq_avg = train.groupby("ipq")["price"].mean().head(20)
plt.plot(
    ipq_avg.index, ipq_avg.values, marker="o", linewidth=2, markersize=8, color="coral"
)
plt.xlabel("Item Pack Quantity (IPQ)", fontsize=12)
plt.ylabel("Average Price ($)", fontsize=12)
plt.title("Average Price by IPQ", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../results/ipq_price_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nIPQ analysis plots saved as 'ipq_price_analysis.png'")

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: FEATURE CORRELATION ANALYSIS")
print("=" * 80)

# Create correlation matrix for numerical features
numerical_features = [
    "price",
    "text_length",
    "word_count",
    "ipq",
    "has_brand",
    "has_premium_keyword",
    "has_budget_keyword",
]

correlation_matrix = train[numerical_features].corr()

print("\n--- Correlation with Price ---")
price_corr = correlation_matrix["price"].sort_values(ascending=False)
print(price_corr)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
)
plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("../results/correlation_matrix(1).png", dpi=300, bbox_inches="tight")
plt.show()

print("\nCorrelation matrix saved as 'correlation_matrix(1).png'")

# ============================================================================
# 9. OUTLIER DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: OUTLIER DETECTION")
print("=" * 80)

# IQR method for outlier detection
Q1 = train["price"].quantile(0.25)
Q3 = train["price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train[(train["price"] < lower_bound) | (train["price"] > upper_bound)]

print("\n--- Outlier Statistics ---")
print(f"Lower Bound: ${lower_bound:.2f}")
print(f"Upper Bound: ${upper_bound:.2f}")
print(f"Number of outliers: {len(outliers)} ({len(outliers) / len(train) * 100:.2f}%)")
print("\nHighest priced products:")
print(train.nlargest(10, "price")[["sample_id", "catalog_content", "price"]])

# ============================================================================
# 10. SUMMARY & KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: SUMMARY & KEY INSIGHTS")
print("=" * 80)

print("\nKEY FINDINGS:")
print("-" * 80)
print(f"1. Dataset Size: {len(train):,} training samples")
print(f"2. Price Range: ${train['price'].min():.2f} - ${train['price'].max():.2f}")
print(
    f"3. Price Distribution: {'Right-skewed' if skewness > 0.5 else 'Normal'} (skewness: {skewness:.2f})"
)
print(f"4. IPQ Correlation: {train['ipq'].corr(train['price']):.3f} (strong indicator)")
print(
    f"5. Brand Impact: ~{((train[train['has_brand'] == 1]['price'].mean() / train[train['has_brand'] == 0]['price'].mean() - 1) * 100):.0f}% price premium"
)
print(
    f"6. Text Length Impact: {train['text_length'].corr(train['price']):.3f} correlation"
)
print(
    f"7. Outliers: {len(outliers)} products ({len(outliers) / len(train) * 100:.1f}%)"
)

print("\n" + "=" * 80)
print("EDA COMPLETE! Ready for Phase 2: Feature Engineering & Modeling")
print("=" * 80)

# Save processed data with extracted features for later use
train_processed = train.copy()
train_processed.to_csv("dataset/train_with_features.csv", index=False)
print("\n Processed training data saved to '../dataset/train_with_features.csv'")
print(
    "   New features added: text_length, word_count, ipq, has_brand, has_premium_keyword, has_budget_keyword"
)
