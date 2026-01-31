#!/bin/bash

# Bundesliga Prediction System - Run Script
# This script sets up and runs the entire system

echo "════════════════════════════════════════════════════════════════"
echo "    Bundesliga Match Prediction System - Setup & Run"
echo "════════════════════════════════════════════════════════════════"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if models exist
if [ ! -f "models/bundesliga_rf.pkl" ]; then
    echo "No trained models found. Training models..."
    python train.py
else
    echo "Trained models found!"
fi

# Run example usage
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Running example predictions..."
echo "════════════════════════════════════════════════════════════════"
python example_usage.py

# Start API server
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Starting API server on http://localhost:5000"
echo "Press Ctrl+C to stop"
echo "════════════════════════════════════════════════════════════════"
python api.py
