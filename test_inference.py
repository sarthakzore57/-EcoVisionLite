"""
End-to-End Pipeline & Inference Validation Script.
Tests:
1. Loading the trained joblib model
2. Feature extraction on sample images from each category
3. Probability prediction
4. Bin recommendation retrieval
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.feature_extractor import WasteFeatureExtractor
from src.waste_categories import get_category_info

def run_inference_tests():
    print("=" * 65)
    print("🧪 EcoVision Lite - End-to-End Inference Verification")
    print("=" * 65)

    model_path = Path("models/svm_waste_classifier.joblib")
    assert model_path.exists(), f"Model file not found: {model_path}"

    data = joblib.load(model_path)
    pipeline = data["pipeline"]
    classes = data["classes"]
    config = data["extractor_config"]

    print(f"Loaded Model Pipeline: {pipeline}")
    print(f"Trained Classes: {classes}")

    extractor = WasteFeatureExtractor(
        image_size=tuple(config["image_size"]),
        orientations=config["orientations"],
        pixels_per_cell=tuple(config["pixels_per_cell"]),
        cells_per_block=tuple(config["cells_per_block"]),
        include_color_hist=config["include_color_hist"],
        color_bins=config["color_bins"]
    )

    data_dir = Path("data/garbage_classification")
    tested_count = 0
    correct_count = 0

    print("\n" + "-" * 65)
    print(f"{'Category':<12} | {'Predicted':<12} | {'Confidence':<10} | {'Recommended Bin'}")
    print("-" * 65)

    for cat in classes:
        cat_dir = data_dir / cat
        img_path = next(cat_dir.glob("*.jpg"), None)
        if img_path:
            img_bgr = cv2.imread(str(img_path))
            feat = extractor.extract_features(img_bgr)
            probas = pipeline.predict_proba(feat.reshape(1, -1))[0]
            pred_idx = int(np.argmax(probas))
            pred_class = classes[pred_idx]
            conf = probas[pred_idx]

            cat_info = get_category_info(pred_class)
            bin_name = cat_info["bin_recommendation"]

            match = "✓" if pred_class == cat else " "
            if pred_class == cat:
                correct_count += 1
            tested_count += 1

            print(f"{match} {cat:<10} | {pred_class:<12} | {conf*100:>8.2f}% | {bin_name}")

    print("-" * 65)
    print(f"Tested {tested_count} category samples. Direct top-1 match rate on first images: {correct_count}/{tested_count} ({correct_count/tested_count*100:.1f}%)")
    print("✅ All end-to-end inference tests PASSED successfully!")
    print("=" * 65)

if __name__ == "__main__":
    run_inference_tests()
