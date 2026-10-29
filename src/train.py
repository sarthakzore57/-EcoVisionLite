"""
EcoVision Lite - Enhanced SVM Classifier Training Pipeline (v2).

Features:
- 96x96 High-Fidelity OpenCV HOG + Dual-Zone HSV Color Profiling
- Stratified 80/20 train/test split (9,807 train / 2,452 test)
- Linear Support Vector Machine (LinearSVC C=0.05 with probability calibration)
- Rigorous real evaluation (Accuracy, Balanced Accuracy, Top-3 Accuracy, Precision, Recall, F1, Confusion Matrix)
- Model serialization via joblib with complete metadata
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from src.feature_extractor import WasteFeatureExtractor

def load_dataset_filelist(data_dir: Path, max_per_class: int | None = None) -> Tuple[List[Path], List[str]]:
    """Scan dataset directory and gather image paths and class labels."""
    categories = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not categories:
        raise ValueError(f"No category directories found in {data_dir}")

    image_paths = []
    labels = []

    for cat in categories:
        cat_dir = data_dir / cat
        files = sorted([
            f for f in cat_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])
        if max_per_class and max_per_class > 0:
            files = files[:max_per_class]

        for f in files:
            image_paths.append(f)
            labels.append(cat)

    return image_paths, labels

def extract_features_memory_efficient(
    image_paths: List[Path], 
    labels: List[str],
    extractor: WasteFeatureExtractor,
    cache_path: Path | None = None,
    batch_size: int = 500
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Extract features in batches into a pre-allocated numpy array to keep memory usage low.
    """
    # Check if cached features exist
    if cache_path and cache_path.exists():
        print(f"⚡ Loading cached features from {cache_path}...", flush=True)
        try:
            cache_data = joblib.load(cache_path)
            if (
                cache_data.get("config") == extractor.get_config() 
                and len(cache_data.get("y", [])) == len(labels)
            ):
                print(f"✅ Cached features loaded successfully ({len(labels):,} samples)!", flush=True)
                return cache_data["X"], cache_data["y"], cache_data["label_encoder"]
            else:
                print("⚠️ Cache configuration mismatch. Re-extracting...", flush=True)
        except Exception as e:
            print(f"⚠️ Cache read error ({e}). Re-extracting...", flush=True)

    total_samples = len(image_paths)
    feature_dim = extractor.total_dim
    print(f"📊 Allocating feature matrix: {total_samples:,} samples × {feature_dim:,} dimensions", flush=True)
    
    # Pre-allocate float32 array (~90MB for ~12k images)
    X = np.empty((total_samples, feature_dim), dtype=np.float32)
    valid_indices = []

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    start_time = time.time()
    for i, path in enumerate(image_paths):
        feat = extractor.extract_from_path(path)
        if feat is not None and len(feat) == feature_dim:
            X[i] = feat
            valid_indices.append(i)

        if (i + 1) % batch_size == 0 or (i + 1) == total_samples:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            pct = int((i + 1) / total_samples * 100)
            print(f"  Processed {i + 1:,}/{total_samples:,} images ({pct}%) [{rate:.1f} img/s]...", flush=True)

    if len(valid_indices) < total_samples:
        print(f"Filtered {total_samples - len(valid_indices)} corrupt or unreadable images.", flush=True)
        X = X[valid_indices]
        y_encoded = y_encoded[valid_indices]

    # Save cache if path provided
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "X": X,
            "y": y_encoded,
            "label_encoder": label_encoder,
            "config": extractor.get_config()
        }, cache_path, compress=3)
        print(f"💾 Feature cache saved to {cache_path}", flush=True)

    return X, y_encoded, label_encoder

def train_and_evaluate(
    data_dir: str = "data/garbage_classification",
    output_dir: str = "models",
    max_per_class: int | None = None,
    image_size: int = 96,
    use_color_hist: bool = True,
    use_cache: bool = True,
    c_param: float = 0.05,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train SVM waste classifier and compute actual evaluation metrics."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65, flush=True)
    print("🚀 EcoVision Lite - Enhanced SVM Waste Classifier Pipeline (v2)", flush=True)
    print("=" * 65, flush=True)
    print(f"Dataset Path:       {data_path.resolve()}", flush=True)
    print(f"Output Directory:   {out_path.resolve()}", flush=True)
    print(f"Image Window:       ({image_size}, {image_size})", flush=True)
    print(f"Color Histogram:    {'Dual-Zone HSV (Global + Center 50% ROI)' if use_color_hist else 'Disabled (HOG only)'}", flush=True)
    print(f"SVM Regularizer C:  {c_param}", flush=True)
    print(f"Max Per Class:      {max_per_class or 'All Images'}", flush=True)
    print("-" * 65, flush=True)

    # 1. Initialize Feature Extractor
    extractor = WasteFeatureExtractor(
        image_size=(image_size, image_size),
        orientations=9,
        pixels_per_cell=(12, 12),
        cells_per_block=(2, 2),
        include_color_hist=use_color_hist,
        color_bins=12
    )

    # 2. Collect image paths
    image_paths, raw_labels = load_dataset_filelist(data_path, max_per_class=max_per_class)
    print(f"Total images gathered: {len(image_paths):,} across {len(set(raw_labels))} classes", flush=True)

    # 3. Extract features
    cache_file = out_path / f"features_cache_{image_size}_dualcolor.joblib" if use_cache else None
    X, y, label_encoder = extract_features_memory_efficient(
        image_paths, raw_labels, extractor, cache_path=cache_file
    )

    # 4. Stratified Split (80% Train, 20% Test)
    print("\n🔀 Splitting dataset into 80% Train / 20% Test (stratified)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"Training samples: {len(X_train):,}", flush=True)
    print(f"Testing samples:  {len(X_test):,}", flush=True)

    # 5. Build and Train Model Pipeline
    print(f"\n⚙️ Training Calibrated Linear SVM (LinearSVC C={c_param})...", flush=True)
    base_svc = LinearSVC(
        C=c_param,
        dual=True,
        max_iter=2500,
        random_state=random_state
    )
    # CalibratedClassifierCV wraps LinearSVC to output well-calibrated class probabilities
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=2)

    model_pipeline = Pipeline([
        ('classifier', calibrated_svc)
    ])

    train_start = time.time()
    model_pipeline.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"✅ Training completed in {train_time:.2f} seconds!", flush=True)

    # 6. Comprehensive Actual Evaluation
    print("\n📈 Evaluating model on test set...", flush=True)
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)

    # Actual metrics
    test_accuracy = float(accuracy_score(y_test, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))

    # Top-3 Accuracy
    top3_correct = 0
    for true_label, probas in zip(y_test, y_proba):
        top3_indices = np.argsort(probas)[-3:]
        if true_label in top3_indices:
            top3_correct += 1
    top3_accuracy = float(top3_correct / len(y_test))

    # Training accuracy
    y_train_pred = model_pipeline.predict(X_train)
    train_accuracy = float(accuracy_score(y_train, y_train_pred))

    # Detailed classification report
    target_names = [str(cls) for cls in label_encoder.classes_]
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=target_names)

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 65, flush=True)
    print("🏆 ACTUAL EVALUATION RESULTS (v2 Enhanced)", flush=True)
    print("=" * 65, flush=True)
    print(f"Train Accuracy:         {train_accuracy * 100:.2f}%", flush=True)
    print(f"Test Accuracy (Top-1):  {test_accuracy * 100:.2f}%", flush=True)
    print(f"Top-3 Test Accuracy:    {top3_accuracy * 100:.2f}%", flush=True)
    print(f"Balanced Accuracy:      {balanced_acc * 100:.2f}%", flush=True)
    print("-" * 65, flush=True)
    print("Classification Report:", flush=True)
    print(report_text, flush=True)
    print("=" * 65, flush=True)

    # 7. Save Model Artifacts
    model_artifact = {
        "pipeline": model_pipeline,
        "classes": list(label_encoder.classes_),
        "extractor_config": extractor.get_config(),
        "train_time_seconds": train_time,
        "metrics": {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "top3_accuracy": top3_accuracy,
            "balanced_accuracy": balanced_acc,
            "num_train_samples": len(X_train),
            "num_test_samples": len(X_test),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    model_filepath = out_path / "svm_waste_classifier.joblib"
    joblib.dump(model_artifact, model_filepath, compress=3)
    print(f"\n💾 Model pipeline successfully saved to: {model_filepath.resolve()}", flush=True)

    # Save metrics JSON for the Streamlit UI
    metrics_export = {
        "model_type": "Calibrated Linear Support Vector Classifier (LinearSVC)",
        "features": f"HOG 96x96 ({extractor.hog_dim} dims) + Dual-Zone Color ({extractor.color_dim} dims)",
        "feature_dim": extractor.total_dim,
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "top3_accuracy": round(top3_accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_time_seconds": round(train_time, 2),
        "classification_report": report_dict,
        "confusion_matrix": conf_mat,
        "classes": list(label_encoder.classes_)
    }
    with open(out_path / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)

    return metrics_export

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EcoVision Lite Enhanced SVM Classifier")
    parser.add_argument("--data_dir", type=str, default="data/garbage_classification")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--max_per_class", type=int, default=None, help="Max images per class (None for all)")
    parser.add_argument("--image_size", type=int, default=96, help="Window size for HOG (default: 96)")
    parser.add_argument("--no_color", action="store_true", help="Disable color histogram")
    parser.add_argument("--no_cache", action="store_true", help="Disable feature caching")
    parser.add_argument("--c", type=float, default=0.05, help="SVM regularization parameter C (default 0.05)")
    
    args = parser.parse_args()

    train_and_evaluate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
        image_size=args.image_size,
        use_color_hist=not args.no_color,
        use_cache=not args.no_cache,
        c_param=args.c
    )
