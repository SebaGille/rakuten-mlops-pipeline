#!/bin/bash

# Streamlit Showcase Launcher Script
# This script activates the virtual environment and launches the Streamlit app

set -e  # Exit on error

echo "🚀 Launching Rakuten MLOps Showcase..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create it first: python3.11 -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📥 Streamlit not found. Installing dependencies..."
    pip install -r requirements-streamlit.txt
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "⚠️  Warning: Docker is not running!"
    echo "Please start Docker Desktop to use all features."
    echo ""
    echo "Press Enter to continue anyway, or Ctrl+C to exit..."
    read
fi

# Launch Streamlit
echo ""
echo "✅ Launching Streamlit app..."
echo "🌐 Browser should open at: http://localhost:8501"
echo ""
echo "⏳ Note: You may see a brief health check error during startup."
echo "   This is normal and will resolve in a few seconds once Streamlit is ready."
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_app/Home.py

