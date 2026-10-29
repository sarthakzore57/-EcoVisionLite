@echo off
setlocal
title EcoVision Lite - Waste Classification
echo =========================================================================
echo                    ECOVISION LITE - WASTE CLASSIFIER
echo =========================================================================
echo.

REM Verify Python installation
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
python -c "import cv2, skimage, sklearn, streamlit, joblib" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing required libraries from requirements.txt...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    echo [OK] All dependencies verified.
)

REM Verify model artifact
if not exist "models\svm_waste_classifier.joblib" (
    echo [2/3] Trained model not detected. Training SVM model now...
    python src\train.py
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
) else (
    echo [2/3] Model artifact detected: models\svm_waste_classifier.joblib
)

echo [3/3] Launching EcoVision Lite Streamlit Application...
echo.
echo -------------------------------------------------------------------------
echo   The web application will open in your browser automatically.
echo   Local Address: http://localhost:8501
echo   Press Ctrl+C in this window to stop the server.
echo -------------------------------------------------------------------------
echo.

python -m streamlit run app.py --server.headless false

pause
