import pandas as pd
from utils import download_images
import os

# --- Configuration ---
# Define the folder where you want to save the images.
DOWNLOAD_FOLDER = "dataset/images"
TRAIN_CSV_PATH = "dataset/train.csv"
TEST_CSV_PATH = "dataset/test.csv"

if __name__ == "__main__":
    print("--- Starting Image Download Process ---")

    # 1. Create the download directory if it doesn't exist
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)
        print(f"Created directory: {DOWNLOAD_FOLDER}")

    # 2. Load the train and test data to get the image links
    print("Loading CSV files to get image links...")
    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        test_df = pd.read_csv(TEST_CSV_PATH)
    except FileNotFoundError as e:
        print(
            f"ERROR: Could not find data file. Make sure your CSVs are at: {e.filename}"
        )
        exit()

    # 3. Combine all unique image links from both datasets
    # Using .unique() prevents downloading the same image twice if it appears in multiple places
    train_links = train_df["image_link"].dropna()
    test_links = test_df["image_link"].dropna()

    all_image_links = pd.concat([train_links, test_links]).unique().tolist()

    print(f"Found a total of {len(all_image_links):,} unique images to download.")

    # 4. Call the download function from utils.py
    print("\nStarting download... This may take a very long time.")
    download_images(image_links=all_image_links, download_folder=DOWNLOAD_FOLDER)

    print("\n--- Download Process Complete ---")
    print(f"All images have been downloaded to the '{DOWNLOAD_FOLDER}' directory.")
