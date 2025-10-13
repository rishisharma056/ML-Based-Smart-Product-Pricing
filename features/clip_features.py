import os
import torch
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings("ignore")

print("=" * 80)
print("STEP 2: CLIP FEATURE EXTRACTION")
print("=" * 80)

# ============================================================================
# SETUP
# ============================================================================

print("\n Checking dependencies...")

try:
    print(f" PyTorch installed (version: {torch.__version__})")

    if torch.cuda.is_available():
        device = "cuda"
        print(f" GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("  Using CPU (will be slower)")
except ImportError:
    print(" PyTorch not installed!")
    exit(1)

# ============================================================================
# LOAD CLIP MODEL
# ============================================================================

print("\n Loading CLIP model...")
print("   Model: openai/clip-vit-base-patch32")
print("   This may take a few minutes on first run...")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)
model.eval()

print(" CLIP model loaded")
print("   Text embedding dim: 512")
print("   Image embedding dim: 512")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n Loading dataset...")
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

print(f" Train: {len(train):,} | Test: {len(test):,}")

# ============================================================================
# EXTRACT TEXT EMBEDDINGS
# ============================================================================

print("\n" + "=" * 80)
print("EXTRACTING TEXT EMBEDDINGS")
print("=" * 80)


def extract_text_embedding_batch(texts, batch_size=32):
    """Extract CLIP text embeddings in batches"""
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
        batch_texts = texts[i : i + batch_size]

        # Process texts
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract embeddings
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )  # Normalize

        all_embeddings.append(text_features.cpu().numpy())

    return np.vstack(all_embeddings)


print("\n Extracting text embeddings for training data...")
train_text_embeddings = extract_text_embedding_batch(train["catalog_content"].tolist())
print(f" Shape: {train_text_embeddings.shape}")

print("\n Extracting text embeddings for test data...")
test_text_embeddings = extract_text_embedding_batch(test["catalog_content"].tolist())
print(f" Shape: {test_text_embeddings.shape}")

# Save text embeddings
np.save("dataset/train_clip_text_embeddings.npy", train_text_embeddings)
np.save("dataset/test_clip_text_embeddings.npy", test_text_embeddings)
print("\n Saved text embeddings")

# ============================================================================
# EXTRACT IMAGE EMBEDDINGS
# ============================================================================

print("\n" + "=" * 80)
print("EXTRACTING IMAGE EMBEDDINGS")
print("=" * 80)


def extract_image_embedding_batch(image_paths, batch_size=32):
    """Extract CLIP image embeddings in batches"""
    all_embeddings = []
    failed_indices = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Image embeddings"):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        batch_valid_indices = []

        # Load images
        for j, img_path in enumerate(batch_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                batch_valid_indices.append(i + j)
            except:
                # Image failed to load - will use zero embedding
                failed_indices.append(i + j)

        if batch_images:
            # Process images
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Extract embeddings
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )  # Normalize

            all_embeddings.append((batch_valid_indices, image_features.cpu().numpy()))

    # Combine embeddings (fill failed with zeros)
    final_embeddings = np.zeros((len(image_paths), 512))

    for indices, embeddings in all_embeddings:
        for idx, emb in zip(indices, embeddings):
            final_embeddings[idx] = emb

    return final_embeddings, failed_indices


# Get image paths
def get_image_paths(df, image_dir):
    """Get local image paths from URLs"""
    paths = []
    for url in df["image_link"]:
        filename = os.path.basename(url)
        path = os.path.join(image_dir, filename)
        paths.append(path)
    return paths


print("\n  Extracting image embeddings for training data...")
train_image_paths = get_image_paths(train, "dataset/images/train")
train_image_embeddings, train_failed = extract_image_embedding_batch(
    train_image_paths, batch_size=32
)

print(f" Shape: {train_image_embeddings.shape}")
print(
    f"  Failed to load: {len(train_failed)} images ({len(train_failed) / len(train) * 100:.2f}%)"
)

print("\n  Extracting image embeddings for test data...")
test_image_paths = get_image_paths(test, "dataset/images/test")
test_image_embeddings, test_failed = extract_image_embedding_batch(
    test_image_paths, batch_size=32
)

print(f" Shape: {test_image_embeddings.shape}")
print(
    f"  Failed to load: {len(test_failed)} images ({len(test_failed) / len(test) * 100:.2f}%)"
)

# Save image embeddings
np.save("dataset/train_clip_image_embeddings.npy", train_image_embeddings)
np.save("dataset/test_clip_image_embeddings.npy", test_image_embeddings)
print("\n Saved image embeddings")

# ============================================================================
# CREATE COMBINED FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("COMBINING FEATURES")
print("=" * 80)

# Load existing text features (from V1)
try:
    train_text_features = pd.read_pickle("dataset/train_features.pkl")
    test_text_features = pd.read_pickle("dataset/test_features.pkl")
    print(f" Loaded existing features: {train_text_features.shape}")
except:
    print("  Warning: Could not load existing features")
    print("   Will use CLIP embeddings only")
    train_text_features = None
    test_text_features = None

# Create CLIP feature DataFrames
clip_text_cols = [f"clip_text_{i}" for i in range(512)]
clip_image_cols = [f"clip_image_{i}" for i in range(512)]

train_clip_text_df = pd.DataFrame(train_text_embeddings, columns=clip_text_cols)
test_clip_text_df = pd.DataFrame(test_text_embeddings, columns=clip_text_cols)

train_clip_image_df = pd.DataFrame(train_image_embeddings, columns=clip_image_cols)
test_clip_image_df = pd.DataFrame(test_image_embeddings, columns=clip_image_cols)

# Combine all features
if train_text_features is not None:
    # Combine: Original features + CLIP text + CLIP image
    train_combined = pd.concat(
        [
            train_text_features.reset_index(drop=True),
            train_clip_text_df,
            train_clip_image_df,
        ],
        axis=1,
    )

    test_combined = pd.concat(
        [
            test_text_features.reset_index(drop=True),
            test_clip_text_df,
            test_clip_image_df,
        ],
        axis=1,
    )
else:
    # Just CLIP features
    train_combined = pd.concat([train_clip_text_df, train_clip_image_df], axis=1)
    test_combined = pd.concat([test_clip_text_df, test_clip_image_df], axis=1)

print("\n Combined features shape:")
print(f"   Train: {train_combined.shape}")
print(f"   Test: {test_combined.shape}")

# Save combined features
train_combined.to_pickle("dataset/train_features_clip.pkl")
test_combined.to_pickle("dataset/test_features_clip.pkl")

print("\n Saved combined features:")
print("   - dataset/train_features_clip.pkl")
print("   - dataset/test_features_clip.pkl")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print(" CLIP FEATURE EXTRACTION COMPLETE!")
print("=" * 80)

print("\n Feature Summary:")
if train_text_features is not None:
    print(f"   Original features: {train_text_features.shape[1]}")
print("   CLIP text embeddings: 512")
print("   CLIP image embeddings: 512")
print(f"   Total features: {train_combined.shape[1]}")

print("\n Files Created:")
print("   ✓ train_clip_text_embeddings.npy")
print("   ✓ test_clip_text_embeddings.npy")
print("   ✓ train_clip_image_embeddings.npy")
print("   ✓ test_clip_image_embeddings.npy")
print("   ✓ train_features_clip.pkl")
print("   ✓ test_features_clip.pkl")

print("\n" + "=" * 80)
