"""
Download CelebA (New Train Split + Original Validation)

This script downloads:
1. Train split from 'nielsr/CelebA-faces' (More images, ~200k)
2. Validation split from 'electronickale/cmu-10799-celeba64-subset' (Original subset)

Usage:
    python download_dataset.py --output_dir ./data/celeba-subset
"""

import os
import argparse
from pathlib import Path

# Define dataset sources
DATASET_SOURCES = {
    "train": {
        "repo": "nielsr/CelebA-faces",
        "split": "train"
    },
    "validation": {
        "repo": "electronickale/cmu-10799-celeba64-subset",
        "split": "validation"
    }
}

def download_from_huggingface(
    output_dir: str = "./data",
    target_split: str = "all",
):
    """
    Download dataset from Hugging Face Hub handling different sources for train/val.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the datasets library:")
        print("  pip install datasets")
        return
    
    # Determine which splits to process
    if target_split == "all":
        splits_to_process = ["train", "validation"]
    else:
        splits_to_process = [target_split]

    print("=" * 60)
    print("Downloading CelebA Dataset (Hybrid Source)")
    print("=" * 60)
    
    # Process each requested split
    for split_name in splits_to_process:
        if split_name not in DATASET_SOURCES:
            print(f"Skipping unknown split: {split_name}")
            continue
            
        source_info = DATASET_SOURCES[split_name]
        repo = source_info["repo"]
        hf_split = source_info["split"]
        
        print(f"\nProcessing '{split_name}' split...")
        print(f"  Source: {repo}")
        print(f"  HF Split: {hf_split}")
        
        # Load dataset
        try:
            dataset = load_dataset(repo, split=hf_split)
        except Exception as e:
            print(f"  Error downloading {split_name}: {e}")
            continue

        # Create output directory specific to this split
        split_output_dir = Path(output_dir) / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to disk
        save_split(dataset, split_output_dir)

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nDataset saved to: {Path(output_dir).absolute()}")
    print("\nTo use in training:")
    print(f"  python train.py --method ddpm --config configs/ddpm.yaml")
    print(f"\n  (Make sure data.root in config points to {output_dir})")


def save_split(dataset, output_dir: Path):
    """Save a dataset split to disk."""
    import pandas as pd
    from tqdm import tqdm
    
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Saving {len(dataset)} images to {images_dir}...")
    
    # Extract attribute columns (everything except 'image' and 'image_id')
    all_columns = dataset.column_names
    attr_columns = [c for c in all_columns if c not in ['image', 'image_id']]
    
    if not attr_columns:
        print("  Note: No attributes found in this split (saving images only).")
    
    # Save images and collect attributes
    attributes = []
    
    # Iterate with tqdm
    for i, item in enumerate(tqdm(dataset, desc=f"  Writing {output_dir.name}")):
        # Save image
        img = item['image']
        
        # Handle ID: Use existing 'image_id' or generate one if missing (common in nielsr dataset)
        img_id = item.get('image_id', f"{i:06d}.jpg")
        
        # Ensure we save as PNG as per original script logic
        img_name = img_id.replace('.jpg', '.png')
        
        # Convert to RGB to ensure consistency (some CelebA images might be RGBA or L)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img.save(images_dir / img_name)
        
        # Collect attributes if they exist
        if attr_columns:
            attrs = {'image_id': img_id}
            for col in attr_columns:
                attrs[col] = item[col]
            attributes.append(attrs)
    
    # Save attributes CSV if we found any attributes
    if attributes:
        df = pd.DataFrame(attributes)
        df = df.set_index('image_id')
        df.to_csv(output_dir / "attributes.csv")
        print(f"  Saved attributes: {attr_columns[:5]}{'...' if len(attr_columns) > 5 else ''}")
    else:
        # Create an empty attributes file or a basic mapping to prevent training crashes
        # if the loader expects a CSV file.
        df = pd.DataFrame({'image_id': [f"{i:06d}.jpg" for i in range(len(dataset))]})
        df = df.set_index('image_id')
        df.to_csv(output_dir / "attributes.csv")
        print("  Created index-only attributes.csv (source had no attributes).")


def main():
    parser = argparse.ArgumentParser(description='Download CelebA dataset (Hybrid)')
    parser.add_argument('--output_dir', type=str, default='./data/celeba-subset',
                        help='Output directory')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'validation', 'all'],
                        help='Which split to download')
    
    args = parser.parse_args()
    
    download_from_huggingface(
        output_dir=args.output_dir,
        target_split=args.split,
    )


if __name__ == '__main__':
    main()