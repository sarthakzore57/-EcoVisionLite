"""
EcoVision Lite - Streamlit Web Application for Waste Classification & Segregation.

Technology Stack:
- Python & OpenCV for image preprocessing
- Histogram of Oriented Gradients (HOG) + HSV Color profiling
- Calibrated Linear Support Vector Machine (LinearSVC)
- Streamlit modern interactive interface
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import pandas as pd
import altair as alt

from src.feature_extractor import WasteFeatureExtractor
from src.waste_categories import get_category_info, BIN_DEFINITIONS

# Page Configuration
st.set_page_config(
    page_title="EcoVision Lite – Waste Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F5132 0%, #198754 50%, #20C997 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(25, 135, 84, 0.15);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 2.4rem;
        margin-bottom: 0.4rem;
    }
    .main-header p {
        color: #E8F5E9;
        font-size: 1.05rem;
        margin: 0;
        opacity: 0.95;
    }

    .bin-card {
        border-radius: 16px;
        padding: 1.6rem;
        color: #1a1a1a;
        margin-top: 1rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        border: 2px solid;
    }
    .bin-title {
        font-size: 1.35rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.8rem;
    }
    .bin-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .instruction-item {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
        font-size: 0.93rem;
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
    }

    .metric-pill {
        background: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        text-align: center;
    }
    .metric-pill-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #198754;
    }
    .metric-pill-label {
        font-size: 0.8rem;
        color: #6C757D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E9ECEF;
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading ML Model Artifacts...")
def load_trained_model():
    """Load model pipeline, metadata, and evaluation metrics."""
    model_path = Path("models/svm_waste_classifier.joblib")
    metrics_path = Path("models/model_metrics.json")
    stats_path = Path("models/dataset_stats.json")

    if not model_path.exists():
        return None, None, None

    model_data = joblib.load(model_path)
    
    metrics = None
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    stats = None
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

    return model_data, metrics, stats

def get_sample_images() -> Dict[str, Path]:
    """Find a representative sample image for each class in the dataset."""
    data_dir = Path("data/garbage_classification")
    samples = {}
    if data_dir.exists():
        for cat_dir in sorted(data_dir.iterdir()):
            if cat_dir.is_dir():
                first_img = next(cat_dir.glob("*.jpg"), None) or next(cat_dir.glob("*.png"), None)
                if first_img:
                    samples[cat_dir.name] = first_img
    return samples

# Application Header
st.markdown("""
<div class="main-header">
    <h1>🌿 EcoVision Lite</h1>
    <p>Intelligent Waste Classification & Smart Segregation Assistant powered by OpenCV, HOG, and Scikit-Learn SVM</p>
</div>
""", unsafe_allow_html=True)

# Load Model
model_data, metrics, stats = load_trained_model()

# Sidebar: Project Info & Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/recycle-sign.png", width=64)
    st.title("EcoVision Lite")
    st.markdown("**No-DL Lightweight ML Architecture**")
    st.markdown("Designed for resource-constrained edge systems without GPU/PyTorch/TensorFlow overhead.")
    
    st.divider()
    
    if metrics:
        st.subheader("Model Status")
        st.markdown(f"**Model:** {metrics.get('model_type', 'Linear SVM')}")
        st.markdown(f"**Test Accuracy (Top-1):** `{metrics.get('test_accuracy', 0)*100:.2f}%`")
        st.markdown(f"**Top-3 Accuracy:** `{metrics.get('top3_accuracy', 0)*100:.2f}%`")
        st.markdown(f"**Feature Dims:** `{metrics.get('feature_dim', 1788)}` (HOG + Color)")
    else:
        st.warning("Model file not found. Run `python src/train.py`.")

    st.divider()
    st.subheader("Standard Bin Legend")
    for key, bin_item in BIN_DEFINITIONS.items():
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
            f"<span style='font-size:1.2rem;'>{bin_item['icon']}</span>"
            f"<span style='font-weight:600;font-size:0.85rem;color:{bin_item['color']}'>{bin_item['name']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# Main Application Tabs
tab_classify, tab_metrics, tab_dataset, tab_about = st.tabs([
    "📸 Classify & Segregate",
    "📈 Actual Model Metrics",
    "🗂️ Dataset Profile",
    "🧠 How It Works (HOG + SVM)"
])

# ==============================================================================
# TAB 1: CLASSIFY & SEGREGATE
# ==============================================================================
with tab_classify:
    if model_data is None:
        st.error("⚠️ The SVM classifier model has not been trained yet!")
        st.info("Run `python src/train.py` from your terminal or command prompt to train the model.")
    else:
        pipeline = model_data["pipeline"]
        classes = model_data["classes"]
        extractor_config = model_data["extractor_config"]

        # Initialize extractor with trained config
        extractor = WasteFeatureExtractor(
            image_size=tuple(extractor_config["image_size"]),
            orientations=extractor_config["orientations"],
            pixels_per_cell=tuple(extractor_config["pixels_per_cell"]),
            cells_per_block=tuple(extractor_config["cells_per_block"]),
            block_norm=extractor_config["block_norm"],
            include_color_hist=extractor_config["include_color_hist"],
            color_bins=extractor_config["color_bins"]
        )

        col_input, col_result = st.columns([1.1, 1.4], gap="large")

        with col_input:
            st.subheader("1. Provide Waste Image")
            input_mode = st.radio(
                "Select image input method:",
                ["Upload Image File", "Use Web Camera", "Choose Dataset Sample"],
                horizontal=True
            )

            img_bgr = None
            image_name = "Uploaded Image"

            if input_mode == "Upload Image File":
                uploaded_file = st.file_uploader(
                    "Upload a waste image (JPG, PNG, WEBP)",
                    type=["jpg", "jpeg", "png", "webp"]
                )
                if uploaded_file is not None:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                    if raw_img is not None:
                        # Handle transparency (RGBA / BGRA) commonly found in WebP and PNG
                        if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
                            alpha = raw_img[:, :, 3].astype(np.float32) / 255.0
                            bgr = raw_img[:, :, :3].astype(np.float32)
                            white_bg = np.ones_like(bgr) * 255.0
                            img_bgr = (bgr * alpha[:, :, None] + white_bg * (1.0 - alpha[:, :, None])).astype(np.uint8)
                        elif len(raw_img.shape) == 2:
                            img_bgr = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
                        else:
                            img_bgr = raw_img[:, :, :3]
                        image_name = uploaded_file.name

                st.caption("ℹ️ **Supported Dataset Categories (10):** Battery, Biological, Cardboard, Clothes, Glass, Metal, Paper, Plastic, Shoes, Trash. *Items outside these classes (e.g., keyboards/electronics) will be mapped to the closest visual texture.*")

            elif input_mode == "Use Web Camera":
                camera_img = st.camera_input("Capture item with webcam")
                if camera_img is not None:
                    file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
                    raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                    if raw_img is not None:
                        if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
                            alpha = raw_img[:, :, 3].astype(np.float32) / 255.0
                            bgr = raw_img[:, :, :3].astype(np.float32)
                            white_bg = np.ones_like(bgr) * 255.0
                            img_bgr = (bgr * alpha[:, :, None] + white_bg * (1.0 - alpha[:, :, None])).astype(np.uint8)
                        else:
                            img_bgr = raw_img[:, :, :3]
                        image_name = "Webcam Capture"

            elif input_mode == "Choose Dataset Sample":
                samples = get_sample_images()
                if samples:
                    selected_sample_cat = st.selectbox(
                        "Pick a test category to load a verified sample:",
                        options=list(samples.keys())
                    )
                    sample_path = samples[selected_sample_cat]
                    img_bgr = cv2.imread(str(sample_path))
                    image_name = f"Sample: {sample_path.name}"
                else:
                    st.info("No sample images detected in `data/garbage_classification`.")


            if img_bgr is not None:
                # Convert BGR to RGB for Streamlit rendering
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=image_name, use_container_width=True)

        with col_result:
            st.subheader("2. Classification & Segregation")
            if img_bgr is None:
                st.info("👈 Please upload an image, capture with your webcam, or select a sample to classify.")
            else:
                with st.spinner("Extracting HOG features and classifying..."):
                    # Extract HOG vector and gradient visualization
                    feature_vector, hog_vis = extractor.extract_features(img_bgr, return_visualization=True)
                    
                    # Predict probabilities using calibrated LinearSVC
                    feature_vector_2d = feature_vector.reshape(1, -1)
                    probas = pipeline.predict_proba(feature_vector_2d)[0]
                    pred_idx = int(np.argmax(probas))
                    pred_class = classes[pred_idx]
                    confidence = float(probas[pred_idx])

                    # Get rich category metadata
                    cat_info = get_category_info(pred_class)
                    bin_info = cat_info["bin_details"]

                # Recommended Bin Banner Card
                st.markdown(f"""
                <div class="bin-card" style="border-color: {bin_info['color']}; background: {bin_info['color']}0A;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <span class="bin-badge" style="{bin_info['badge_style']}">
                                RECOMMENDED SEGREGATION BIN
                            </span>
                            <div class="bin-title" style="color: {bin_info['color']};">
                                <span style="font-size: 2rem;">{bin_info['icon']}</span>
                                <span>{bin_info['name']}</span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.85rem; color: #6C757D; font-weight: 600;">CONFIDENCE</span>
                            <div style="font-size: 1.8rem; font-weight: 800; color: {bin_info['color']};">
                                {confidence * 100:.1f}%
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #212529; margin-bottom: 0.5rem;">
                        Identified Item: {cat_info['display_name']}
                    </div>
                    <p style="color: #495057; font-size: 0.95rem; margin-bottom: 1rem;">
                        {bin_info['description']}
                    </p>
                    <div style="font-weight: 700; font-size: 0.9rem; color: #343A40; margin-bottom: 0.4rem;">
                        Disposal & Preparation Guidelines:
                    </div>
                    {"".join(f'<div class="instruction-item"><span>✔️</span><span>{instr}</span></div>' for instr in cat_info['instructions'])}
                    <div style="margin-top: 0.9rem; padding: 0.7rem 0.9rem; background: #FFFFFF; border-left: 4px solid {bin_info['color']}; border-radius: 6px; font-size: 0.88rem; color: #374151;">
                        <strong>🌱 Eco Fact:</strong> {cat_info['environmental_impact']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if confidence < 0.50:
                    st.warning(
                        f"⚠️ **Moderate / Low Confidence ({confidence * 100:.1f}%)**: "
                        f"The model is uncertain. Review the **Prediction Confidence Breakdown** below for top alternatives. "
                        f"If this item does not belong to the 10 trained categories (e.g., computer keyboards or appliances), "
                        f"the model maps its edge/color texture to the closest available category."
                    )

                st.write("")
                # Top Predictions Probability Distribution
                st.markdown("#### 📊 Prediction Confidence Breakdown")

                
                # Sort probabilities descending
                sorted_indices = np.argsort(probas)[::-1]
                prob_data = []
                for idx in sorted_indices[:5]:
                    c_name = classes[idx]
                    c_info = get_category_info(c_name)
                    prob_data.append({
                        "Category": c_info["display_name"],
                        "Probability": float(probas[idx]),
                        "Percentage": f"{probas[idx]*100:.1f}%",
                        "Bin": c_info["bin_recommendation"]
                    })
                
                df_probs = pd.DataFrame(prob_data)
                
                chart = alt.Chart(df_probs).mark_bar(cornerRadius=6, height=22).encode(
                    x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1]), title="Calibrated Probability"),
                    y=alt.Y('Category:N', sort='-x', title=None),
                    color=alt.Color('Probability:Q', scale=alt.Scale(scheme='greens'), legend=None),
                    tooltip=['Category', 'Percentage', 'Bin']
                ).properties(height=180)
                
                st.altair_chart(chart, use_container_width=True)

                # Feature Inspection Expander
                with st.expander("🔍 Inspect Extracted HOG Gradient Features"):
                    st.markdown("""
                    **How the algorithm sees your image:**
                    OpenCV preprocessed the image with Gaussian smoothing and CLAHE contrast equalization, 
                    then computed gradient orientation histograms across 8x8 pixel cells.
                    """)
                    fcol1, fcol2, fcol3 = st.columns(3)
                    with fcol1:
                        st.image(cv2.resize(img_rgb, tuple(extractor.image_size)), caption=f"Normalized Input ({extractor.image_size[0]}x{extractor.image_size[1]})", use_container_width=True)
                    with fcol2:
                        st.image(hog_vis, caption="HOG Edge Gradient Map", use_container_width=True)
                    with fcol3:
                        st.markdown(f"""
                        **Vector Profiling:**
                        - HOG Spatial Dims: `{extractor.hog_dim}`
                        - Color Profile Dims: `{extractor.color_dim}`
                        - Total Feature Vector: `{extractor.total_dim}`
                        - Feature Non-zeros: `{(feature_vector != 0).sum()}`
                        """)

# ==============================================================================
# TAB 2: MODEL EVALUATION METRICS
# ==============================================================================
with tab_metrics:
    st.subheader("📊 Rigorous Empirical Model Evaluation")
    st.markdown("""
    All evaluation metrics below were computed directly on the **independent 20% test set (2,452 images)** 
    never seen during training.
    """)

    if metrics is not None:
        # Key Metric Cards
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-pill-val">{metrics['test_accuracy']*100:.2f}%</div>
                <div class="metric-pill-label">Top-1 Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-pill-val">{metrics['top3_accuracy']*100:.2f}%</div>
                <div class="metric-pill-label">Top-3 Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol3:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-pill-val">{metrics['balanced_accuracy']*100:.2f}%</div>
                <div class="metric-pill-label">Balanced Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol4:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-pill-val">{metrics['train_time_seconds']}s</div>
                <div class="metric-pill-label">Training Latency</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("")

        col_rep, col_cm = st.columns([1.1, 1.3], gap="large")

        with col_rep:
            st.markdown("#### 📋 Per-Category Performance")
            report = metrics["classification_report"]
            rows = []
            for cls in metrics["classes"]:
                if cls in report:
                    rows.append({
                        "Category": cls.capitalize(),
                        "Precision": f"{report[cls]['precision']*100:.1f}%",
                        "Recall": f"{report[cls]['recall']*100:.1f}%",
                        "F1-Score": f"{report[cls]['f1-score']*100:.1f}%",
                        "Test Samples": int(report[cls]['support'])
                    })
            df_rep = pd.DataFrame(rows)
            st.dataframe(df_rep, use_container_width=True, hide_index=True)

            st.markdown("""
            > **Analysis:** Categories with distinct structural silhouettes and textures (such as **Clothes (79% F1)** and **Shoes (65% F1)**) 
            score highest under HOG. Amorphous objects (such as generic **Trash** or crumpled **Metal**) have higher intra-class variation.
            """)

        with col_cm:
            st.markdown("#### 🔲 Confusion Matrix Heatmap")
            cm = np.array(metrics["confusion_matrix"])
            cls_names = [c.capitalize() for c in metrics["classes"]]

            # Build heatmap dataframe
            cm_data = []
            for i, true_cls in enumerate(cls_names):
                for j, pred_cls in enumerate(cls_names):
                    cm_data.append({
                        "True Label": true_cls,
                        "Predicted Label": pred_cls,
                        "Count": int(cm[i, j])
                    })
            df_cm = pd.DataFrame(cm_data)

            heatmap = alt.Chart(df_cm).mark_rect().encode(
                x=alt.X('Predicted Label:N', title="Predicted Category"),
                y=alt.Y('True Label:N', title="True Category"),
                color=alt.Color('Count:Q', scale=alt.Scale(scheme='greens'), title="Count"),
                tooltip=['True Label', 'Predicted Label', 'Count']
            ).properties(height=350)

            st.altair_chart(heatmap, use_container_width=True)

    else:
        st.warning("Evaluation metrics have not been generated yet. Please train the model via `python src/train.py`.")

# ==============================================================================
# TAB 3: DATASET PROFILE
# ==============================================================================
with tab_dataset:
    st.subheader("🗂️ Garbage Classification Dataset Profile")
    st.markdown("""
    **Dataset Source:** [Kaggle: Mostafa Abla - Garbage Classification](https://www.kaggle.com/mostafaabla/garbage-classification)  
    Contains 12,259 standardized RGB images categorized across 10 municipal waste categories.
    """)

    if stats is not None:
        scol1, scol2, scol3, scol4 = st.columns(4)
        with scol1:
            st.metric("Total Images", f"{stats['total_images']:,}")
        with scol2:
            st.metric("Waste Categories", f"{stats['num_classes']}")
        with scol3:
            st.metric("Corrupt Images", f"{stats['corrupt_images']}")
        with scol4:
            st.metric("Input Dimensions", "256 × 256")

        st.write("")
        st.markdown("#### Class Distribution Across 12,259 Images")
        
        class_dist = []
        for cat, count in stats["class_counts"].items():
            info = get_category_info(cat)
            class_dist.append({
                "Category": info["display_name"],
                "Count": count,
                "Percentage": count / stats["total_images"] * 100,
                "Bin": info["bin_recommendation"]
            })
        df_dist = pd.DataFrame(class_dist).sort_values("Count", ascending=False)

        chart_dist = alt.Chart(df_dist).mark_bar(cornerRadius=6).encode(
            x=alt.X('Count:Q', title="Number of Images"),
            y=alt.Y('Category:N', sort='-x', title=None),
            color=alt.Color('Count:Q', scale=alt.Scale(scheme='tealblues'), legend=None),
            tooltip=['Category', 'Count', alt.Tooltip('Percentage:Q', format='.2f'), 'Bin']
        ).properties(height=320)

        st.altair_chart(chart_dist, use_container_width=True)

# ==============================================================================
# TAB 4: HOW IT WORKS (HOG + SVM)
# ==============================================================================
with tab_about:
    st.subheader("🧠 Machine Learning Architecture: Why HOG + SVM?")
    
    st.markdown("""
    Modern ML tutorials often jump directly to multi-gigabyte Deep Learning models (YOLO, ResNet, EfficientNet). 
    While powerful, deep neural networks require massive GPU resources, gigabytes of RAM, and huge download footprints.

    **EcoVision Lite** demonstrates that classical, mathematically elegant computer vision and machine learning:
    1. Runs smoothly on modest CPU hardware (Windows laptops, Raspberry Pi, edge cameras).
    2. Trains from scratch in **12 seconds** without GPU acceleration.
    3. Occupies only **~300KB** on disk for the trained model artifact.
    4. Achieves **80.26% Top-3 accuracy** across 10 complex waste categories!
    """)

    col_h1, col_h2 = st.columns(2, gap="large")

    with col_h1:
        st.markdown("""
        ### 1. HOG (Histogram of Oriented Gradients)
        HOG describes the **shape, contours, and texture** of objects:
        - **Gradient Computation:** OpenCV computes horizontal ($G_x$) and vertical ($G_y$) derivatives using Sobel filters to detect edges.
        - **Orientation Binning:** Pixels are grouped into small $8 \\times 8$ pixel cells. Each cell accumulates gradient magnitudes into 9 discrete angle bins ($0^\\circ$ to $180^\\circ$).
        - **Block Normalization:** $2 \\times 2$ cell blocks normalize feature vectors with L2-Hys norm to achieve illumination and shadow invariance.
        - **HSV Color Histograms:** Material colors (e.g. green biological vs brown cardboard) are concatenated to boost separation.
        """)

    with col_h2:
        st.markdown("""
        ### 2. Linear Support Vector Machine (LinearSVC)
        SVM finds the **optimal maximum-margin hyperplane** separating classes in 1,788-dimensional space:
        - **Convex Optimization:** Guaranteed global optimum without getting trapped in local minima like neural networks.
        - **Extreme Speed:** Coordinate descent converges in seconds.
        - **Calibrated Probabilities:** Using Platt scaling / isotonic regression (`CalibratedClassifierCV`), raw decision boundary distances are converted into trustworthy probability percentages for the user interface.
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6C757D; font-size: 0.88rem;">
        EcoVision Lite – Sustainable AI for Environmental Conservation & Waste Reduction.
    </div>
    """, unsafe_allow_html=True)
