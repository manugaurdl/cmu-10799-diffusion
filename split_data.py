import os
import shutil
import random
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = Path("/data_new/manu/celebA/data/celeba-subset/200k/images")
TRAIN_DIR = Path("/data_new/manu/celebA/data/celeba-subset/200k/train")
VAL_DIR = Path("/data_new/manu/celebA/data/celeba-subset/200k/val")
VAL_COUNT = 13000

def split_dataset():
    # 1. Validation checks
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    # Get all files (filtering for common image extensions if needed, or take all files)
    # This example takes all files in the directory
    all_files = [f for f in SOURCE_DIR.iterdir() if f.is_file()]
    total_files = len(all_files)

    if total_files < VAL_COUNT:
        print(f"Error: Not enough files ({total_files}) to create a validation set of {VAL_COUNT}.")
        return

    print(f"Found {total_files} images. Starting split...")

    # 2. Create Train and Val directories
    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)

    # 3. Shuffle and Split
    random.shuffle(all_files)
    
    val_files = all_files[:VAL_COUNT]
    train_files = all_files[VAL_COUNT:]

    # 4. Move files to respective directories
    print(f"Moving {len(val_files)} images to {VAL_DIR}...")
    for file in val_files:
        shutil.move(str(file), str(VAL_DIR / file.name))

    print(f"Moving {len(train_files)} images to {TRAIN_DIR}...")
    for file in train_files:
        shutil.move(str(file), str(TRAIN_DIR / file.name))

    print("Success! Dataset split complete.")
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    # Optional: Remove empty source directory
    # os.rmdir(SOURCE_DIR) 

if __name__ == "__main__":
    split_dataset()