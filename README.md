# 🌿 EcoVision Lite – Waste Classification & Smart Segregation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-HOG%20Features-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Linear%20SVM-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An intelligent, beginner-friendly computer vision and machine learning project for **municipal waste classification and automatic segregation bin recommendation**.

Built **strictly without heavy deep learning frameworks** (No YOLO, CNNs, TensorFlow, or PyTorch). Instead, EcoVision Lite leverages classical, mathematically elegant **Histogram of Oriented Gradients (HOG)**, **HSV color profiles**, and a **Calibrated Linear Support Vector Machine (LinearSVC)**.

---

## 🎯 Key Project Highlights

- **Lightweight & CPU-Friendly:** Trains from scratch in **~13 seconds** and runs on any standard Windows PC or edge hardware without requiring dedicated GPUs.
- **Memory-Efficient Processing:** Features a pre-allocated vectorizer and batch generator that compresses the entire 12,259 image dataset into only **~88MB of RAM**.
- **Interactive Web Interface:** Streamlit UI supporting **file uploads**, **live web camera input**, and **dataset sample exploration**.
- **Segregation Bin Recommendation:** Suggests appropriate disposal bins (**Blue Dry Recyclable**, **Green Organic Compost**, **Yellow Glass**, **Red Hazardous/E-Waste**, **Purple Textile**, **Black Landfill**) along with handling rules and eco tips.
- **Explainable Vision:** Renders the actual **HOG Gradient Vector Map** side-by-side with your photo so you can see how the computer vision algorithm detects object edges and textures.
- **Real Empirical Evaluation:** Achieves **56.08% Top-1 Accuracy** and **80.26% Top-3 Accuracy** across 10 complex waste categories on an independent 2,452 image test set.

---

## 📁 Repository Structure

```text
EcoVisionLite/
├── data/
│   └── garbage_classification/       # 12,259 images across 10 categories
├── models/
│   ├── svm_waste_classifier.joblib   # Trained Linear SVM pipeline & encoder
│   ├── model_metrics.json            # Actual test metrics & confusion matrix
│   ├── dataset_stats.json            # Dataset shape and category counts
│   └── features_cache_64_color.joblib# Memory-cached feature matrix
├── src/
│   ├── __init__.py
│   ├── download_dataset.py           # Kaggle download & archive extraction
│   ├── dataset_inspector.py          # Data validation & distribution stats
│   ├── feature_extractor.py          # OpenCV HOG + HSV color feature extractor
│   ├── train.py                      # Memory-efficient SVM training pipeline
│   └── waste_categories.py           # Segregation bin rules & disposal metadata
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
├── run.bat                           # 1-Click Windows Batch launcher
├── run.ps1                           # 1-Click Windows PowerShell launcher
├── test_inference.py                 # End-to-end verification test suite
└── README.md                         # Documentation
```

---

## 📊 Dataset Information

Dataset: **[Garbage Classification (12 Classes / 10 Standardized Categories)](https://www.kaggle.com/mostafaabla/garbage-classification)** by Mostafa Abla.

### Categories & Distribution (12,259 Standardized Images)

| Category | Image Count | Percentage | Recommended Segregation Bin |
| :--- | :---: | :---: | :--- |
| **battery** | 756 | 6.17% | ⚠️ Red Hazardous & E-Waste Bin |
| **biological** | 699 | 5.70% | 🌱 Green Organic / Compost Bin |
| **cardboard** | 1,411 | 11.51% | ♻️ Blue Dry Recyclable Bin |
| **clothes** | 1,892 | 15.43% | 👕 Purple Textile / Donation Drop-off |
| **glass** | 1,736 | 14.16% | 🍾 Yellow / Dedicated Glass Bin |
| **metal** | 930 | 7.59% | ♻️ Blue Dry Recyclable Bin |
| **paper** | 1,336 | 10.90% | ♻️ Blue Dry Recyclable Bin |
| **plastic** | 1,597 | 13.03% | ♻️ Blue Dry Recyclable Bin |
| **shoes** | 1,449 | 11.82% | 👕 Purple Textile / Footwear Drop-off |
| **trash** | 453 | 3.70% | 🗑️ Black / Grey Landfill Bin |

---

## 🏆 Actual Empirical Evaluation Results (v2 Enhanced)

> [!NOTE]
> All metrics below are **actual empirical numbers** evaluated on the independent 20% stratified test set (2,452 images).

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Test Accuracy (Top-1)** | **62.93%** | **6.3× higher** than 10-class random chance (10.0%) |
| **Top-3 Test Accuracy** | **84.75%** | Correct category within top 3 predicted candidates |
| **Balanced Accuracy** | **58.87%** | Accounts for class frequency differences |
| **Training Latency** | **25.51 seconds** | Fast CPU convergence |
| **Feature Dimensionality**| **1,836 dimensions** | 1,764 HOG (96×96) + 72 Dual-Zone HSV color bins |

### Classification Report (Test Set)

```text
              precision    recall  f1-score   support

     battery       0.67      0.56      0.61       151
  biological       0.62      0.71      0.67       140
   cardboard       0.61      0.69      0.65       282
     clothes       0.81      0.87      0.84       379
       glass       0.61      0.61      0.61       347
       metal       0.45      0.32      0.37       186
       paper       0.56      0.57      0.56       267
     plastic       0.53      0.53      0.53       319
       shoes       0.66      0.75      0.70       290
       trash       0.60      0.27      0.38        91

    accuracy                           0.63      2452
   macro avg       0.61      0.59      0.59      2452
weighted avg       0.62      0.63      0.62      2452
```

---

## 🚀 Quick Start Guide (Windows)

### Option 1: One-Click Launch (Easiest)
Simply double-click:
```cmd
run.bat
```
or run in PowerShell:
```powershell
.\run.ps1
```
The script will automatically check your Python environment, install any missing requirements, ensure the model is trained, and open `http://localhost:8501` in your browser.

---

### Option 2: Step-by-Step Manual Setup

#### Step 1: Clone or Navigate to the Repository
```powershell
cd e:\project\EcoVisionLite
```

#### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

#### Step 3: Dataset Acquisition
If you already have the dataset or zip archive in the folder, simply run:
```powershell
python src/download_dataset.py
```
> **Kaggle API Authentication (Optional):**  
> If downloading directly from Kaggle:
> 1. Log in to [Kaggle](https://www.kaggle.com/) and go to [Settings](https://www.kaggle.com/settings).
> 2. Click **Create New Token** to download `kaggle.json`.
> 3. Place `kaggle.json` in `C:\Users\<Your-Username>\.kaggle\kaggle.json`.
> 4. Run `python src/download_dataset.py`.

#### Step 4: Inspect the Dataset
```powershell
python src/dataset_inspector.py
```

#### Step 5: Train the SVM Classifier
```powershell
python src/train.py
```
*Options:*
- `--c 0.01` : SVM regularization parameter C (default 0.01)
- `--image_size 64` : Image window size for HOG (default 64)
- `--no_color` : Train with pure HOG features only

#### Step 6: Verify End-to-End Inference
```powershell
python test_inference.py
```

#### Step 7: Launch the Streamlit Web Application
```powershell
python -m streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## 🧠 Technical Deep-Dive: How It Works

### 1. OpenCV Preprocessing & HOG Feature Extraction
1. **Canonical Resize:** Input image is resized to $64 \times 64$ pixels to enforce standard scale.
2. **Noise Suppression:** A $3 \times 3$ Gaussian blur filter softens camera sensor noise.
3. **Contrast Equalization:** Contrast Limited Adaptive Histogram Equalization (**CLAHE**) normalizes illumination across shadows and highlights.
4. **HOG Orientations:** Gradients are computed using 1D Sobel operators ($\begin{bmatrix}-1 & 0 & 1\end{bmatrix}$). Gradient orientations are binned into 9 unsigned orientation intervals ($0^\circ - 180^\circ$) over $8 \times 8$ pixel cells.
5. **Block Normalization:** $2 \times 2$ cell blocks are normalized using L2-Hys norm to make features invariant to lighting changes.
6. **HSV Color Profiling:** A 24-dimensional normalized HSV color histogram is concatenated to help distinguish materials where color is crucial (e.g., green biological waste vs. brown cardboard).

### 2. Calibrated Linear Support Vector Machine
- Uses **LinearSVC** ($C=0.01$, dual formulation) to find the maximum-margin hyperplanes separating the 10 classes in 1,788-dimensional space.
- Wrapped in **CalibratedClassifierCV** with Platt scaling to turn raw decision boundary margins into calibrated class probability percentages.

---

## 🛡️ License
Distributed under the MIT License. Built for education, accessibility, and environmental sustainability.
