import pandas as pd
import numpy as np
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from tqdm import tqdm

warnings.filterwarnings("ignore")

tqdm.pandas()

print("=" * 80)
print("FEATURE ENGINEERING PIPELINE - TEXT FEATURES")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "tfidf_max_features": 150,  # Top 150 TF-IDF features
    "tfidf_ngram_range": (1, 2),  # Unigrams and bigrams
    "tfidf_min_df": 10,  # Minimum document frequency
    "tfidf_max_df": 0.8,  # Maximum document frequency
    "use_category_tfidf": True,  # Category-specific TF-IDF
    "category_tfidf_features": 20,  # Features per category
}

# ============================================================================
# PART 1: LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: LOADING DATA")
print("=" * 80)

print("\nLoading training data...")
train = pd.read_csv("dataset/train.csv")
print(f"Training set: {train.shape}")

print("\nLoading test data...")
test = pd.read_csv("dataset/test.csv")
print(f"Test set: {test.shape}")

# Save target separately
y_train = train["price"].values
print(f"\nTarget (price) range: ${y_train.min():.2f} - ${y_train.max():.2f}")

# ============================================================================
# PART 2: TEXT CLEANING
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: TEXT CLEANING")
print("=" * 80)


def clean_text(text):
    """Clean catalog content for feature extraction"""
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove common formatting artifacts
    text = re.sub(r"bullet point \d+:", "", text)
    text = re.sub(r"item name:", "", text)
    text = re.sub(r"unit:", "", text)
    text = re.sub(r"value:", "", text)
    text = re.sub(r"item pack quantity:", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("\nCleaning text...")
train["cleaned_text"] = train["catalog_content"].progress_apply(clean_text)
test["cleaned_text"] = test["catalog_content"].progress_apply(clean_text)

print("Text cleaning complete")
print(f"   Sample: {train['cleaned_text'].iloc[0][:100]}...")

# ============================================================================
# PART 3: BASIC TEXT FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: BASIC TEXT FEATURES")
print("=" * 80)


def extract_basic_text_features(text):
    """Extract basic statistical features from text"""
    if pd.isna(text):
        return {
            "text_length": 0,
            "word_count": 0,
            "avg_word_length": 0,
            "uppercase_ratio": 0,
            "digit_count": 0,
            "special_char_count": 0,
            "sentence_count": 0,
        }

    text_str = str(text)
    words = text_str.split()

    features = {
        "text_length": len(text_str),
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "uppercase_ratio": sum(1 for c in text_str if c.isupper())
        / max(len(text_str), 1),
        "digit_count": sum(1 for c in text_str if c.isdigit()),
        "special_char_count": sum(
            1 for c in text_str if not c.isalnum() and not c.isspace()
        ),
        "sentence_count": len(re.findall(r"[.!?]+", text_str)),
    }

    return features


print("\nExtracting basic text features...")
train_text_features = train["catalog_content"].progress_apply(
    extract_basic_text_features
)
test_text_features = test["catalog_content"].progress_apply(extract_basic_text_features)

train_text_df = pd.DataFrame(train_text_features.tolist())
test_text_df = pd.DataFrame(test_text_features.tolist())

print(f"Extracted {len(train_text_df.columns)} basic text features")
print(f"   Features: {list(train_text_df.columns)}")

# ============================================================================
# PART 4: IPQ EXTRACTION
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: ITEM PACK QUANTITY (IPQ) EXTRACTION")
print("=" * 80)


def extract_ipq(text):
    """Extract Item Pack Quantity from text"""
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


print("\n Extracting IPQ...")
train["ipq"] = train["catalog_content"].progress_apply(extract_ipq)
test["ipq"] = test["catalog_content"].progress_apply(extract_ipq)

print(" IPQ extracted")
print(
    f"   Train - Products with IPQ > 1: {(train['ipq'] > 1).sum()} ({(train['ipq'] > 1).mean() * 100:.1f}%)"
)
print(
    f"   Test  - Products with IPQ > 1: {(test['ipq'] > 1).sum()} ({(test['ipq'] > 1).mean() * 100:.1f}%)"
)
print(f"   Train - Average IPQ: {train['ipq'].mean():.2f}")
print(f"   Test  - Average IPQ: {test['ipq'].mean():.2f}")

# Create log IPQ
train["log_ipq"] = np.log1p(train["ipq"])
test["log_ipq"] = np.log1p(test["ipq"])

# ============================================================================
# PART 5: KEYWORD FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("PART 5: KEYWORD FEATURES")
print("=" * 80)

# Premium keywords
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
    "gourmet",
    "artisan",
]

# Budget keywords
budget_keywords = ["basic", "standard", "value", "economy", "simple", "generic"]

# Positive descriptive words
positive_words = [
    "delicious",
    "quality",
    "best",
    "perfect",
    "fresh",
    "natural",
    "organic",
    "healthy",
    "tasty",
    "rich",
    "smooth",
    "creamy",
]

# Brand indicators
brand_keywords = ["brand", "amazon", "basics"]

print("\n Extracting keyword features...")

# Has premium keywords
train["has_premium"] = (
    train["cleaned_text"]
    .str.contains("|".join(premium_keywords), case=False, na=False)
    .astype(int)
)
test["has_premium"] = (
    test["cleaned_text"]
    .str.contains("|".join(premium_keywords), case=False, na=False)
    .astype(int)
)

# Has budget keywords
train["has_budget"] = (
    train["cleaned_text"]
    .str.contains("|".join(budget_keywords), case=False, na=False)
    .astype(int)
)
test["has_budget"] = (
    test["cleaned_text"]
    .str.contains("|".join(budget_keywords), case=False, na=False)
    .astype(int)
)

# Has positive words
train["has_positive"] = (
    train["cleaned_text"]
    .str.contains("|".join(positive_words), case=False, na=False)
    .astype(int)
)
test["has_positive"] = (
    test["cleaned_text"]
    .str.contains("|".join(positive_words), case=False, na=False)
    .astype(int)
)


# Positive word count
def count_positive_words(text):
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    return sum(text_lower.count(word) for word in positive_words)


train["positive_word_count"] = train["cleaned_text"].apply(count_positive_words)
test["positive_word_count"] = test["cleaned_text"].apply(count_positive_words)

# Has brand
train["has_brand"] = (
    train["cleaned_text"]
    .str.contains("|".join(brand_keywords + premium_keywords[:3]), case=False, na=False)
    .astype(int)
)
test["has_brand"] = (
    test["cleaned_text"]
    .str.contains("|".join(brand_keywords + premium_keywords[:3]), case=False, na=False)
    .astype(int)
)

# Organic/Natural flags
train["has_organic"] = (
    train["cleaned_text"].str.contains("organic", case=False, na=False).astype(int)
)
test["has_organic"] = (
    test["cleaned_text"].str.contains("organic", case=False, na=False).astype(int)
)

train["has_natural"] = (
    train["cleaned_text"].str.contains("natural", case=False, na=False).astype(int)
)
test["has_natural"] = (
    test["cleaned_text"].str.contains("natural", case=False, na=False).astype(int)
)

print(" Keyword features extracted")
print(
    f"   has_premium: {train['has_premium'].sum()} ({train['has_premium'].mean() * 100:.1f}%)"
)
print(
    f"   has_budget: {train['has_budget'].sum()} ({train['has_budget'].mean() * 100:.1f}%)"
)
print(
    f"   has_positive: {train['has_positive'].sum()} ({train['has_positive'].mean() * 100:.1f}%)"
)
print(
    f"   has_brand: {train['has_brand'].sum()} ({train['has_brand'].mean() * 100:.1f}%)"
)
print(
    f"   has_organic: {train['has_organic'].sum()} ({train['has_organic'].mean() * 100:.1f}%)"
)

# ============================================================================
# PART 6: CATEGORY INFERENCE
# ============================================================================

print("\n" + "=" * 80)
print("PART 6: CATEGORY INFERENCE")
print("=" * 80)


def infer_category(text):
    """Infer product category from text"""
    if pd.isna(text):
        return "other"

    text_lower = str(text).lower()

    categories = {
        "coffee_tea": ["coffee", "tea", "espresso", "cappuccino"],
        "candy_chocolate": ["candy", "chocolate", "gum", "mint"],
        "snacks": ["chips", "popcorn", "crackers", "pretzels", "nuts"],
        "beverages": ["juice", "soda", "water", "drink", "beverage"],
        "baking": ["flour", "sugar", "baking", "yeast", "vanilla"],
        "spices": ["spice", "seasoning", "salt", "pepper", "herb"],
        "pasta_rice": ["pasta", "rice", "noodle", "spaghetti"],
        "cereal": ["cereal", "oatmeal", "granola", "breakfast"],
        "organic": ["organic", "gluten free", "vegan", "non-gmo"],
        "protein": ["protein", "whey", "supplement", "powder"],
        "condiments": ["ketchup", "mustard", "mayo", "sauce", "dressing"],
        "oil": ["oil", "olive oil", "coconut oil"],
    }

    for category, keywords in categories.items():
        if any(keyword in text_lower for keyword in keywords):
            return category

    return "other"


print("\n Inferring categories...")
train["category"] = train["cleaned_text"].progress_apply(infer_category)
test["category"] = test["cleaned_text"].progress_apply(infer_category)

print(" Categories inferred")
print("\n   Category distribution (train):")
print(train["category"].value_counts())

# ============================================================================
# PART 7: TF-IDF FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("PART 7: TF-IDF VECTORIZATION")
print("=" * 80)

print("\n Training TF-IDF with config:")
print(f"   Max features: {CONFIG['tfidf_max_features']}")
print(f"   N-gram range: {CONFIG['tfidf_ngram_range']}")
print(f"   Min df: {CONFIG['tfidf_min_df']}")
print(f"   Max df: {CONFIG['tfidf_max_df']}")

# Initialize TF-IDF
tfidf = TfidfVectorizer(
    max_features=CONFIG["tfidf_max_features"],
    ngram_range=CONFIG["tfidf_ngram_range"],
    min_df=CONFIG["tfidf_min_df"],
    max_df=CONFIG["tfidf_max_df"],
    stop_words="english",
    sublinear_tf=True,  # Use log scaling
)

# Fit on training data
print("\n🔧 Fitting TF-IDF on training data...")
tfidf_train = tfidf.fit_transform(train["cleaned_text"])
tfidf_test = tfidf.transform(test["cleaned_text"])

print("TF-IDF features created")
print(f"   Shape: {tfidf_train.shape}")
print(f"   Features: {len(tfidf.get_feature_names_out())}")
print("\n   Top 20 TF-IDF features:")
for i, feature in enumerate(tfidf.get_feature_names_out()[:20], 1):
    print(f"   {i:2d}. {feature}")

# Convert to DataFrame
tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_train.shape[1])]
train_tfidf_df = pd.DataFrame(tfidf_train.toarray(), columns=tfidf_feature_names)
test_tfidf_df = pd.DataFrame(tfidf_test.toarray(), columns=tfidf_feature_names)

# Save TF-IDF vectorizer
os.makedirs("models/feature_extractors", exist_ok=True)
with open("models/feature_extractors/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("\nSaved TF-IDF vectorizer to models/feature_extractors/tfidf_vectorizer.pkl")

# ============================================================================
# PART 8: CATEGORY ENCODING
# ============================================================================

print("\n" + "=" * 80)
print("PART 8: CATEGORY ENCODING")
print("=" * 80)

print("\nEncoding categories...")

# Label encode categories
le = LabelEncoder()
train["category_encoded"] = le.fit_transform(train["category"])
test["category_encoded"] = le.transform(test["category"])

print(" Categories encoded")
print(f"   Unique categories: {len(le.classes_)}")
print(f"   Categories: {list(le.classes_)}")

# Save label encoder
with open("models/feature_extractors/category_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("\n Saved category encoder to models/feature_extractors/category_encoder.pkl")

# ============================================================================
# PART 9: COMBINE ALL FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("PART 9: COMBINING ALL FEATURES")
print("=" * 80)

print("\n🔗 Combining all features...")

# Numerical features
numerical_cols = ["ipq", "log_ipq"]

# Keyword features
keyword_cols = [
    "has_premium",
    "has_budget",
    "has_positive",
    "positive_word_count",
    "has_brand",
    "has_organic",
    "has_natural",
]

# Category features
category_cols = ["category_encoded"]

# Combine train features
train_features = pd.concat(
    [
        train_text_df.reset_index(drop=True),  # Basic text features
        train[numerical_cols].reset_index(drop=True),  # Numerical features
        train[keyword_cols].reset_index(drop=True),  # Keyword features
        train[category_cols].reset_index(drop=True),  # Category features
        train_tfidf_df.reset_index(drop=True),  # TF-IDF features
    ],
    axis=1,
)

# Combine test features
test_features = pd.concat(
    [
        test_text_df.reset_index(drop=True),
        test[numerical_cols].reset_index(drop=True),
        test[keyword_cols].reset_index(drop=True),
        test[category_cols].reset_index(drop=True),
        test_tfidf_df.reset_index(drop=True),
    ],
    axis=1,
)

print(" Features combined")
print(f"   Training features shape: {train_features.shape}")
print(f"   Test features shape: {test_features.shape}")

# ============================================================================
# PART 10: FEATURE SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PART 10: FEATURE SUMMARY")
print("=" * 80)

feature_groups = {
    "Basic Text": list(train_text_df.columns),
    "Numerical": numerical_cols,
    "Keywords": keyword_cols,
    "Category": category_cols,
    "TF-IDF": tfidf_feature_names,
}

print("\n Feature Groups:")
total_features = 0
for group, features in feature_groups.items():
    print(f"\n   {group}: {len(features)} features")
    if len(features) <= 10:
        for feat in features:
            print(f"      - {feat}")
    else:
        print(f"      - {features[0]}, {features[1]}, ... {features[-1]}")
    total_features += len(features)

print(f"\n    TOTAL FEATURES: {total_features}")

# Check for NaN or inf values
nan_count = train_features.isna().sum().sum()
inf_count = np.isinf(train_features.select_dtypes(include=[np.number])).sum().sum()

if nan_count > 0:
    print(f"\n    WARNING: {nan_count} NaN values detected!")
    print("   Filling NaN with 0...")
    train_features = train_features.fillna(0)
    test_features = test_features.fillna(0)

if inf_count > 0:
    print(f"\n    WARNING: {inf_count} inf values detected!")
    print("   Clipping extreme values...")
    train_features = train_features.replace([np.inf, -np.inf], 0)
    test_features = test_features.replace([np.inf, -np.inf], 0)

# ============================================================================
# PART 11: SAVE FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("PART 11: SAVING FEATURES")
print("=" * 80)

# Save feature names
feature_names_file = "features/feature_names.txt"
os.makedirs("features", exist_ok=True)
with open(feature_names_file, "w") as f:
    for col in train_features.columns:
        f.write(f"{col}\n")
print(f"\nSaved feature names to {feature_names_file}")

# Save features
print("\nSaving feature files...")
train_features.to_pickle("dataset/train_features.pkl")
test_features.to_pickle("dataset/test_features.pkl")
print("Saved dataset/train_features.pkl")
print("Saved dataset/test_features.pkl")

# Save target
np.save("dataset/y_train.npy", y_train)
print(" Saved dataset/y_train.npy")

# Save sample IDs
train_ids = train["sample_id"].values
test_ids = test["sample_id"].values
np.save("dataset/train_ids.npy", train_ids)
np.save("dataset/test_ids.npy", test_ids)
print("Saved dataset/train_ids.npy")
print("Saved dataset/test_ids.npy")

# ============================================================================
# PART 12: FINAL VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("PART 12: FINAL VALIDATION")
print("=" * 80)

print("\n FEATURE ENGINEERING COMPLETE!")
print("\n Summary:")
print(f"   Training samples: {len(train_features):,}")
print(f"   Test samples: {len(test_features):,}")
print(f"   Total features: {train_features.shape[1]}")
print(f"   Target range: ${y_train.min():.2f} - ${y_train.max():.2f}")

print("=" * 80)
