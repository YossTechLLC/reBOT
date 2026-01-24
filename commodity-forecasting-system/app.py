#!/usr/bin/env python3
"""
Launcher for Volatility Prediction UI
Run with: streamlit run app.py
"""
import runpy
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Run the actual app
if __name__ == "__main__":
    runpy.run_path("src/ui/app.py", run_name="__main__")
