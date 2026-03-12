#!/bin/bash
# ─────────────────────────────────────────────────────────────
# UPI Behaviour Mystery — Local Setup
# Run this once, then use the commands at the bottom.
# ─────────────────────────────────────────────────────────────

set -e  # stop on first error

echo ""
echo "=================================================="
echo "  UPI Behaviour Mystery — Setup"
echo "=================================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED="3.10"
echo ""
echo "Python version: $PYTHON_VERSION (need $REQUIRED+)"

# Create virtual environment
echo ""
echo "Creating virtual environment (.venv)..."
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip silently
pip install --upgrade pip -q

# Install the project and all dependencies
echo "Installing dependencies..."
pip install -e ".[dev]" -q

echo ""
echo "=================================================="
echo "  Setup complete!"
echo "=================================================="
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run:"
echo ""
echo "  # Full analysis pipeline (CLI)"
echo "  python -m src.pipeline"
echo ""
echo "  # With custom parameters"
echo "  python -m src.pipeline --n-users 5000 --seed 7 --cashback 30"
echo ""
echo "  # Interactive dashboard"
echo "  streamlit run app.py"
echo ""
echo "  # Test suite"
echo "  pytest tests/ -v"
echo ""
echo "  # Quick test (small dataset)"
echo "  python -m src.pipeline --n-users 1000 --no-save"
echo "=================================================="
