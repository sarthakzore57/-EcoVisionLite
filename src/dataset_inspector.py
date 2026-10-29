"""
EcoVision Lite - Dataset Inspection and Profiling Module.

Analyzes dataset directory:
- Counts images per category
- Detects image dimensions, color channels, file extensions
- Checks for missing or corrupted images
- Generates structured summary statistics
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def inspect_dataset(data_dir: str | Path = "data/garbage_classification") -> dict:
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_path.resolve()}")

    categories = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    if not categories:
        raise ValueError(f"No category subdirectories found in {data_path.resolve()}")

    print("=" * 65)
    print("🔍 EcoVision Lite - Dataset Inspection Report")
    print("=" * 65)
    print(f"Dataset root: {data_path.resolve()}")
    print(f"Total categories identified: {len(categories)}")
    print("-" * 65)

    stats = {
        "dataset_root": str(data_path.resolve()),
        "num_classes": len(categories),
        "classes": categories,
        "class_counts": {},
        "total_images": 0,
        "corrupt_images": 0,
        "sample_shapes": {},
        "extensions": defaultdict(int)
    }

    sample_inspections = 5  # Check first N images per class for shape validation

    for cat in categories:
        cat_dir = data_path / cat
        files = [f for f in cat_dir.iterdir() if f.is_file()]
        valid_count = 0

        for f in files:
            ext = f.suffix.lower()
            stats["extensions"][ext] += 1
            if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                valid_count += 1

        stats["class_counts"][cat] = valid_count
        stats["total_images"] += valid_count

        # Inspect sample images for resolution and integrity
        sample_files = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}][:sample_inspections]
        shapes = []
        for sf in sample_files:
            try:
                img = cv2.imread(str(sf))
                if img is not None:
                    shapes.append(f"{img.shape[1]}x{img.shape[0]}x{img.shape[2]}")
                else:
                    stats["corrupt_images"] += 1
            except Exception:
                stats["corrupt_images"] += 1
        stats["sample_shapes"][cat] = list(set(shapes))

    # Print table of categories and counts
    print(f"{'Category Name':<20} | {'Image Count':<12} | {'Percentage':<10} | {'Sample Shape'}")
    print("-" * 65)
    for cat in categories:
        count = stats["class_counts"][cat]
        pct = (count / stats["total_images"] * 100) if stats["total_images"] > 0 else 0
        shape_str = ", ".join(stats["sample_shapes"].get(cat, [])) or "N/A"
        print(f"{cat:<20} | {count:<12} | {pct:>8.2f}% | {shape_str}")

    print("-" * 65)
    print(f"Total Valid Images:  {stats['total_images']:,}")
    print(f"Corrupt Images:      {stats['corrupt_images']}")
    print(f"Detected Formats:    {dict(stats['extensions'])}")
    print("=" * 65)

    # Save metadata JSON for the Streamlit app to display
    meta_path = Path("models")
    meta_path.mkdir(exist_ok=True)
    stats_serializable = {
        **stats,
        "extensions": dict(stats["extensions"])
    }
    with open(meta_path / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_serializable, f, indent=2)

    return stats_serializable

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/garbage_classification"
    inspect_dataset(target)
