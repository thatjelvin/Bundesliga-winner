@echo off
REM Bundesliga Prediction System - Run Script (Windows)
REM This script sets up and runs the entire system

echo ================================================================
echo     Bundesliga Match Prediction System - Setup ^& Run
echo ================================================================

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Check if models exist
if not exist "models\bundesliga_rf.pkl" (
    echo No trained models found. Training models...
    python train.py
) else (
    echo Trained models found!
)

REM Run example usage
echo.
echo ================================================================
echo Running example predictions...
echo ================================================================
python example_usage.py

REM Start API server
echo.
echo ================================================================
echo Starting API server on http://localhost:5000
echo Press Ctrl+C to stop
echo ================================================================
python api.py
