#!/usr/bin/env python3
"""Download and extract the AllRecipes dataset from Kaggle."""

import os
import zipfile
import requests
from pathlib import Path


def download_dataset():
    """Download the AllRecipes dataset from Kaggle and extract it."""

    # Setup paths
    output_dir = Path("kaggle_dataset")
    output_dir.mkdir(exist_ok=True)

    zip_path = output_dir / "allrecipes.zip"

    # Kaggle dataset URL
    url = "https://www.kaggle.com/api/v1/datasets/download/nguyentuongquang/all-recipes"

    # Check if dataset already exists
    csv_path = output_dir / "allrecipes.csv"
    if csv_path.exists():
        print(f"Dataset already exists at {csv_path}")
        return csv_path

    print(f"Downloading dataset from Kaggle...")

    # Get Kaggle credentials from environment
    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_API_TOKEN")

    if not kaggle_username or not kaggle_key:
        # Try to read from kaggle.json
        kaggle_json_path = Path("kaggle.json")
        if kaggle_json_path.exists():
            import json
            with open(kaggle_json_path) as f:
                creds = json.load(f)
                kaggle_username = creds.get("username")
                kaggle_key = creds.get("key")
        else:
            raise ValueError(
                "Kaggle credentials not found. Please either:\n"
                "1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables, or\n"
                "2. Create ~/.kaggle/kaggle.json with your credentials"
            )

    # Download with authentication
    response = requests.get(
        url,
        auth=(kaggle_username, kaggle_key),
        stream=True
    )
    response.raise_for_status()

    # Get total size for progress
    total_size = int(response.headers.get('content-length', 0))

    # Write to file
    downloaded = 0
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {percent:.1f}%", end="", flush=True)

    print("\nDownload complete!")

    # Extract the zip file
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Clean up zip file
    zip_path.unlink()
    print(f"Dataset extracted to {output_dir}/")

    # List extracted files
    extracted_files = list(output_dir.iterdir())
    print(f"Extracted files: {[f.name for f in extracted_files]}")

    return csv_path


if __name__ == "__main__":
    download_dataset()
