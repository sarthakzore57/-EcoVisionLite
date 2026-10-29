# EcoVision Lite - Windows PowerShell Launcher
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host "                   ECOVISION LITE - WASTE CLASSIFIER" -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host ""

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python is not installed or not in your PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://www.python.org/"
    pause
    exit 1
}

# 2. Check dependencies
Write-Host "[1/3] Verifying Python libraries..." -ForegroundColor Cyan
try {
    python -c "import cv2, skimage, sklearn, streamlit, joblib"
    Write-Host "[OK] All required libraries are ready." -ForegroundColor Green
} catch {
    Write-Host "[INFO] Installing requirements from requirements.txt..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
}

# 3. Check model
if (-not (Test-Path "models/svm_waste_classifier.joblib")) {
    Write-Host "[2/3] Model not found. Running training pipeline..." -ForegroundColor Yellow
    python src/train.py
} else {
    Write-Host "[2/3] Trained SVM model found in models/." -ForegroundColor Green
}

# 4. Launch app
Write-Host "[3/3] Starting Streamlit Web App..." -ForegroundColor Cyan
Write-Host "Opening web browser at http://localhost:8501..." -ForegroundColor White
python -m streamlit run app.py
