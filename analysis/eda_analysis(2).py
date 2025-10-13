import re
import warnings
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats

warnings.filterwarnings("ignore")

# Set visualization style
tqdm.pandas()
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Loading Data
train = pd.read_csv("../dataset/train.csv")
test = pd.read_csv("../dataset/test.csv")

# ============================================================================
# PART 1: SAMPLE PRODUCT INSPECTION
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: DETAILED PRODUCT SAMPLES")
print("=" * 80)

print("\n--- Examining Low, Medium, and High Price Products ---\n")

# Get samples from different price ranges
low_price = train[train["price"] < 10].sample(3, random_state=42)
mid_price = train[(train["price"] >= 10) & (train["price"] < 50)].sample(
    3, random_state=42
)
high_price = train[train["price"] >= 100].sample(3, random_state=42)


def display_product(row, label):
    print(f"\n{'=' * 80}")
    print(f"{label}")
    print(f"{'=' * 80}")
    print(f"Sample ID: {row['sample_id']}")
    print(f"Price: ${row['price']:.2f}")
    print(f"Image: {row['image_link']}")
    print("\nCatalog Content:")
    print(f"{row['catalog_content'][:500]}...")
    print(f"\nFull length: {len(row['catalog_content'])} characters")


print("\n🔵 LOW PRICE PRODUCTS ($0-10):")
for idx, row in low_price.iterrows():
    display_product(row, f"Low Price Example {idx}")

print("\n🟢 MEDIUM PRICE PRODUCTS ($10-50):")
for idx, row in mid_price.iterrows():
    display_product(row, f"Medium Price Example {idx}")

print("\n🔴 HIGH PRICE PRODUCTS ($100+):")
for idx, row in high_price.iterrows():
    display_product(row, f"High Price Example {idx}")

# ============================================================================
# PART 2: CATALOG CONTENT STRUCTURE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: CATALOG CONTENT STRUCTURE")
print("=" * 80)


# Parse catalog content structure
def parse_catalog_structure(text):
    """Parse the structured fields in catalog content"""
    fields = {}

    # Extract Item Name
    name_match = re.search(r"Item Name:\s*(.+?)(?=\s*Bullet Point|$)", text, re.DOTALL)
    if name_match:
        fields["item_name"] = name_match.group(1).strip()

    # Count bullet points
    bullet_count = len(re.findall(r"Bullet Point \d+:", text))
    fields["bullet_count"] = bullet_count

    # Extract unit info
    unit_match = re.search(r"Unit:\s*(.+?)(?=\s*Item Name|$)", text)
    if unit_match:
        fields["unit"] = unit_match.group(1).strip()

    # Extract value info
    value_match = re.search(r"Value:\s*(.+?)(?=\s*Item Name|$)", text)
    if value_match:
        fields["value"] = value_match.group(1).strip()

    return fields


print("\nAnalyzing catalog structure (sampling 1000 products)...")
sample_structures = train.sample(1000, random_state=42)["catalog_content"].apply(
    parse_catalog_structure
)

bullet_counts = [s.get("bullet_count", 0) for s in sample_structures]
print("\n--- Bullet Point Statistics ---")
print(f"Average bullet points per product: {np.mean(bullet_counts):.2f}")
print(f"Median: {np.median(bullet_counts):.0f}")
print(f"Most common count: {Counter(bullet_counts).most_common(1)[0]}")

# Check if all products have Item Name
has_item_name = sum(1 for s in sample_structures if "item_name" in s)
print("\n--- Field Presence ---")
print(
    f"Products with 'Item Name' field: {has_item_name}/{len(sample_structures)} ({has_item_name / len(sample_structures) * 100:.1f}%)"
)

# ============================================================================
# PART 3: ADVANCED TEXT FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: ADVANCED TEXT FEATURE EXTRACTION")
print("=" * 80)

print("\nExtracting advanced text features...")


# Text quality features
def extract_advanced_text_features(text):
    """Extract detailed text features"""
    if pd.isna(text):
        return {
            "uppercase_ratio": 0,
            "digit_count": 0,
            "special_char_count": 0,
            "avg_word_length": 0,
            "sentence_count": 0,
            "exclamation_count": 0,
            "question_count": 0,
        }

    text_str = str(text)

    features = {}
    features["uppercase_ratio"] = sum(1 for c in text_str if c.isupper()) / max(
        len(text_str), 1
    )
    features["digit_count"] = sum(1 for c in text_str if c.isdigit())
    features["special_char_count"] = sum(
        1 for c in text_str if not c.isalnum() and not c.isspace()
    )

    words = text_str.split()
    features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0

    features["sentence_count"] = len(re.findall(r"[.!?]+", text_str))
    features["exclamation_count"] = text_str.count("!")
    features["question_count"] = text_str.count("?")

    return features


# Apply to sample for speed
print("Extracting from sample of 10,000 products...")
sample_df = train.sample(10000, random_state=42).copy()
text_features = sample_df["catalog_content"].progress_apply(
    extract_advanced_text_features
)
text_features_df = pd.DataFrame(text_features.tolist())

# Merge with sample
for col in text_features_df.columns:
    sample_df[col] = text_features_df[col].values

print("\n--- Advanced Text Feature Correlations with Price ---")
for col in text_features_df.columns:
    corr = sample_df[col].corr(sample_df["price"])
    print(f"{col:25s}: {corr:7.4f}")

# ============================================================================
# PART 4: CATEGORY INFERENCE
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: PRODUCT CATEGORY INFERENCE")
print("=" * 80)


def infer_category(text):
    """Infer product category from text"""
    if pd.isna(text):
        return "unknown"

    text_lower = str(text).lower()

    # Define category keywords (comprehensive)
    categories = {
        "coffee_tea": ["coffee", "tea", "espresso", "cappuccino", "latte", "chai"],
        "candy_chocolate": ["candy", "chocolate", "gum", "mint", "lollipop", "truffle"],
        "snacks": ["chips", "popcorn", "crackers", "pretzels", "nuts", "trail mix"],
        "beverages": ["juice", "soda", "water", "drink", "beverage", "energy drink"],
        "baking": ["flour", "sugar", "baking", "yeast", "vanilla", "extract"],
        "spices_seasoning": [
            "spice",
            "seasoning",
            "salt",
            "pepper",
            "herb",
            "cinnamon",
        ],
        "pasta_rice": ["pasta", "rice", "noodle", "spaghetti", "macaroni"],
        "canned_goods": ["canned", "can of", "tomato sauce", "soup"],
        "cereal_breakfast": ["cereal", "oatmeal", "granola", "breakfast", "muesli"],
        "organic_health": ["organic", "gluten free", "vegan", "natural", "non-gmo"],
        "baby_food": ["baby", "infant", "formula", "gerber"],
        "pet_food": ["dog", "cat", "pet", "puppy", "kitten"],
        "protein_supplements": ["protein", "whey", "supplement", "vitamin", "powder"],
        "condiments": ["ketchup", "mustard", "mayo", "sauce", "dressing", "vinegar"],
        "oil_cooking": ["oil", "olive oil", "coconut oil", "cooking spray"],
    }

    # Check each category
    for category, keywords in categories.items():
        if any(keyword in text_lower for keyword in keywords):
            return category

    return "other"


print("\nInferring categories for all products...")
train["category"] = train["catalog_content"].progress_apply(infer_category)

print("\n--- Category Distribution ---")
category_counts = train["category"].value_counts()
print(category_counts)

print("\n--- Average Price by Category ---")
category_prices = (
    train.groupby("category")["price"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
)
print(category_prices)

# Visualize
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
category_counts.head(15).plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Product Count by Category (Top 15)", fontsize=14, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.subplot(2, 1, 2)
category_prices.head(15)["mean"].plot(kind="bar", color="lightcoral", edgecolor="black")
plt.title("Average Price by Category (Top 15)", fontsize=14, fontweight="bold")
plt.xlabel("Category")
plt.ylabel("Average Price ($)")
plt.xticks(rotation=45, ha="right")
plt.axhline(
    train["price"].mean(),
    color="red",
    linestyle="--",
    label=f"Overall Mean: ${train['price'].mean():.2f}",
)
plt.legend()
plt.tight_layout()

plt.savefig("../results/category_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: ../results/category_analysis.png")

# ============================================================================
# PART 5: PRICE PER UNIT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: PRICE PER UNIT ANALYSIS")
print("=" * 80)


# Fixed IPQ extraction
def extract_ipq_robust(text):
    """Robust IPQ extraction with multiple patterns"""
    if pd.isna(text):
        return 1

    text_lower = str(text).lower()

    # Look for specific IPQ field first
    ipq_field = re.search(r"item pack quantity:\s*(\d+)", text_lower)
    if ipq_field:
        qty = int(ipq_field.group(1))
        if 1 <= qty <= 1000:
            return qty

    # Fallback patterns
    patterns = [
        r"pack of (\d{1,2})\b",
        r"\b(\d{1,2})\s*pack\b",
        r"\b(\d{1,2})-pack\b",
        r"set of (\d{1,2})\b",
        r"\b(\d{1,2})\s*count\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 100:
                return qty

    return 1


print("\nCalculating IPQ and price per unit...")
train["ipq"] = train["catalog_content"].progress_apply(extract_ipq_robust)
train["price_per_unit"] = train["price"] / train["ipq"]

print("\n--- Price Per Unit Statistics ---")
print(f"Mean price per unit: ${train['price_per_unit'].mean():.2f}")
print(f"Median price per unit: ${train['price_per_unit'].median():.2f}")
print(f"Std dev: ${train['price_per_unit'].std():.2f}")

print("\n--- Correlation Analysis ---")
print(f"Price vs IPQ: {train['price'].corr(train['ipq']):.4f}")
print(f"Price vs Price_per_unit: {train['price'].corr(train['price_per_unit']):.4f}")

# Price per unit by category
print("\n--- Price Per Unit by Category (Top 10) ---")
ppu_by_cat = (
    train.groupby("category")["price_per_unit"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
    .head(10)
)
print(ppu_by_cat)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribution of price per unit
axes[0].hist(
    train["price_per_unit"], bins=100, color="skyblue", edgecolor="black", range=(0, 50)
)
axes[0].set_title("Price Per Unit Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Price Per Unit ($)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(
    train["price_per_unit"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: ${train['price_per_unit'].mean():.2f}",
)
axes[0].legend()

# Price vs IPQ
axes[1].scatter(train["ipq"], train["price"], alpha=0.3, s=10)
axes[1].set_title("Price vs IPQ", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Item Pack Quantity")
axes[1].set_ylabel("Price ($)")
axes[1].set_xlim(0, 30)

# Price per unit vs IPQ
axes[2].scatter(train["ipq"], train["price_per_unit"], alpha=0.3, s=10)
axes[2].set_title("Price Per Unit vs IPQ", fontsize=14, fontweight="bold")
axes[2].set_xlabel("Item Pack Quantity")
axes[2].set_ylabel("Price Per Unit ($)")
axes[2].set_xlim(0, 30)
axes[2].set_ylim(0, 50)

plt.tight_layout()
plt.savefig("../results/price_per_unit_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: ../results/price_per_unit_analysis.png")

# ============================================================================
# PART 6: BRAND EXTRACTION
# ============================================================================
print("\n" + "=" * 80)
print("PART 6: BRAND NAME EXTRACTION")
print("=" * 80)


def extract_brand_name(text):
    """Extract actual brand name from text"""
    if pd.isna(text):
        return "unknown"

    text_str = str(text)

    # Look for "Item Name: BRAND ..." pattern
    name_match = re.search(r"Item Name:\s*([A-Z][A-Za-z0-9\s&\']+)", text_str)
    if name_match:
        name = name_match.group(1).strip()
        # Get first 1-3 words as potential brand
        words = name.split()[:3]
        brand = " ".join(words)
        return brand if len(brand) > 2 else "unknown"

    return "unknown"


print("\nExtracting brand names...")
train["brand_extracted"] = train["catalog_content"].progress_apply(extract_brand_name)

# Get top brands
brand_counts = train["brand_extracted"].value_counts().head(30)
print("\n--- Top 30 Brands ---")
print(brand_counts)

# Average price by top brands
print("\n--- Average Price by Top 15 Brands ---")
top_brands = brand_counts.head(15).index
brand_prices = (
    train[train["brand_extracted"].isin(top_brands)]
    .groupby("brand_extracted")["price"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)
print(brand_prices)

# Visualize
plt.figure(figsize=(14, 6))
brand_prices["mean"].plot(kind="bar", color="lightgreen", edgecolor="black")
plt.title("Average Price by Top 15 Brands", fontsize=14, fontweight="bold")
plt.xlabel("Brand")
plt.ylabel("Average Price ($)")
plt.xticks(rotation=45, ha="right")
plt.axhline(
    train["price"].mean(),
    color="red",
    linestyle="--",
    label=f"Overall Mean: ${train['price'].mean():.2f}",
)
plt.legend()
plt.tight_layout()
plt.savefig("../results/brand_pricing_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: ../results/brand_pricing_analysis.png")

# ============================================================================
# PART 7: IMAGE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 7: IMAGE LINK ANALYSIS & SAMPLE DOWNLOAD")
print("=" * 80)

print("\nAnalyzing image links...")


# Check URL patterns
def analyze_image_url(url):
    """Extract info from image URL"""
    if pd.isna(url):
        return {"has_image": False, "domain": None}

    url_str = str(url)
    return {
        "has_image": True,
        "domain": "amazon" if "amazon" in url_str else "other",
        "is_https": url_str.startswith("https://"),
        "has_jpg": url_str.endswith(".jpg"),
    }


sample_urls = train["image_link"].head(1000).apply(analyze_image_url)
url_df = pd.DataFrame(sample_urls.tolist())

print("\n--- Image URL Statistics (sample of 1000) ---")
print(f"Has image: {url_df['has_image'].sum()}/{len(url_df)}")
print(f"Amazon domain: {(url_df['domain'] == 'amazon').sum()}/{len(url_df)}")
print(f"HTTPS: {url_df['is_https'].sum()}/{len(url_df)}")
print(f"JPG format: {url_df['has_jpg'].sum()}/{len(url_df)}")

print("\n--- Sample Image URLs ---")
for i, url in enumerate(train["image_link"].head(5), 1):
    print(f"{i}. {url}")

# Download sample images
"""
print("\nDownloading sample images...")
print("NOTE: You can download images using src/utils.py download_images() function")
print("For this deep dive, we'll skip actual download to save time.")
print("Recommendation: Download 50-100 sample images manually to inspect quality")
"""

# ============================================================================
# PART 8: OUTLIER DETECTION & ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 8: OUTLIER DETCTION & ANALYSIS")
print("=" * 80)

# Create 'has_brand' column earlier in the script
train["has_brand"] = (
    train["catalog_content"]
    .str.lower()
    .str.contains(
        "brand|amazon|premium|pro|professional|deluxe|ultra|plus|max|elite", na=False
    )
    .astype(int)
)

# Multiple outlier detection methods
Q1 = train["price"].quantile(0.25)
Q3 = train["price"].quantile(0.75)
IQR = Q3 - Q1
upper_iqr = Q3 + 1.5 * IQR

# Z-score method
z_scores = np.abs(stats.zscore(train["price"]))
upper_zscore = train[z_scores > 3]

print("\n--- Outlier Detection Methods ---")
print(f"IQR Method (>Q3 + 1.5*IQR): {len(train[train['price'] > upper_iqr])} outliers")
print(f"Z-score Method (|z| > 3): {len(upper_zscore)} outliers")
print(
    f"Percentile Method (>99th): {len(train[train['price'] > train['price'].quantile(0.99)])} outliers"
)

# Analyze outliers
outliers = train[train["price"] > upper_iqr][
    ["price", "ipq", "category", "has_brand"]
].copy()
print("\n--- Outlier Characteristics ---")
print(f"Outlier mean price: ${outliers['price'].mean():.2f}")
print(f"Outlier median price: ${outliers['price'].median():.2f}")
print(f"Outlier avg IPQ: {outliers['ipq'].mean():.2f}")
print(f"Outliers with brand: {outliers['has_brand'].mean() * 100:.1f}%")

print("\n--- Top 10 Most Expensive Products ---")
top_expensive = train.nlargest(10, "price")[
    ["sample_id", "price", "ipq", "category", "brand_extracted"]
]
print(top_expensive)

# Outlier categories
print("\n--- Outlier Category Distribution ---")
print(outliers["category"].value_counts().head(10))

# ============================================================================
# PART 9: CORRELATION HEATMAP - EXTENDED
# ============================================================================
print("\n" + "=" * 80)
print("PART 9: EXTENDED CORRELATION ANALYSIS")
print("=" * 80)

# Create extended feature set
print("\nCreating correlation matrix...")

# Add text features to main df (use sample)
train["text_length"] = train["catalog_content"].str.len()
train["word_count"] = train["catalog_content"].str.split().str.len()

# Brand features
train["has_premium_keyword"] = (
    train["catalog_content"]
    .str.lower()
    .str.contains("premium|professional|pro|deluxe|ultra|elite|luxury", na=False)
    .astype(int)
)

# Select numerical features
numerical_cols = [
    "price",
    "text_length",
    "word_count",
    "ipq",
    "price_per_unit",
    "has_brand",
    "has_premium_keyword",
]

# Create correlation matrix
corr_matrix = train[numerical_cols].corr()

print("\n--- Feature Correlations with Price ---")
price_corrs = corr_matrix["price"].sort_values(ascending=False)
print(price_corrs)

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
    fmt=".3f",
)
plt.title("Correlation Matrix", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("../results/correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved: correlation_matrix.png")


# ============================================================================
# 10. SUMMARY & KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY & KEY INSIGHTS")
print("=" * 80)

print("\nKEY DISCOVERIES:")
print("-" * 80)
print(
    f"1. Dataset: {len(train):,} products across {train['category'].nunique()} categories"
)
print(f"2. Price range: ${train['price'].min():.2f} - ${train['price'].max():.2f}")
print(
    f"3. Most expensive category: {category_prices.index[0]} (${category_prices.iloc[0]['mean']:.2f})"
)
print(
    f"4. Least expensive category: {category_prices.index[-1]} (${category_prices.iloc[-1]['mean']:.2f})"
)
print(
    f"5. Brand presence: {train['has_brand'].mean() * 100:.1f}% (37.9% price premium)"
)
print(f"6. Multi-packs: {(train['ipq'] > 1).sum() / len(train) * 100:.1f}%")
print(f"7. Outliers: {len(outliers)} ({len(outliers) / len(train) * 100:.1f}%)")

print("\nFEATURE IMPORTANCE RANKING:")
print("-" * 80)
for i, (feat, corr) in enumerate(price_corrs.items(), 1):
    if feat != "price":
        strength = (
            "Strong" if abs(corr) > 0.15 else "Moderate" if abs(corr) > 0.05 else "Weak"
        )
        print(f"{i}. {feat:20s}: {corr:7.4f} {strength}")

print("\nSAVING PROCESSED DATA...")
# Save enriched training data
train_enriched = train[
    [
        "sample_id",
        "catalog_content",
        "image_link",
        "price",
        "ipq",
        "price_per_unit",
        "category",
        "brand_extracted",
        "text_length",
        "word_count",
        "has_brand",
        "has_premium_keyword",
    ]
]
train_enriched.to_csv("../dataset/train_enriched.csv", index=False)
print("Saved: ../dataset/train_enriched.csv")

print("\n" + "=" * 80)
print("DEEP DIVE COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. category_analysis.png")
print("  2. price_per_unit_analysis.png")
print("  3. brand_pricing_analysis.png")
print("  4. extended_correlation_matrix.png")
print("  5. train_enriched.csv")
