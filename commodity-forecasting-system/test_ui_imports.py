#!/usr/bin/env python
"""
Quick test script to verify UI imports work correctly.
Run with: python test_ui_imports.py
"""

import sys
import os

# Add src to path (same way app.py does it)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

print("Testing UI module imports...")
print(f"Project root: {project_root}")
print(f"Python path includes: {os.path.join(project_root, 'src')}")
print()

try:
    print("1. Testing ui.data_manager...")
    from ui.data_manager import DataManager
    print("   ✅ DataManager imported successfully")

    print("2. Testing ui.model_controller...")
    from ui.model_controller import ModelController
    print("   ✅ ModelController imported successfully")

    print("3. Testing ui.visualization...")
    from ui.visualization import VolatilityCharts
    print("   ✅ VolatilityCharts imported successfully")

    print("4. Testing ui.explainability...")
    from ui.explainability import FeatureAnalyzer, SHAPExplainer
    print("   ✅ FeatureAnalyzer and SHAPExplainer imported successfully")

    print("5. Testing ui.strategy...")
    from ui.strategy import SpreadRecommender, PositionSizer
    print("   ✅ SpreadRecommender and PositionSizer imported successfully")

    print("6. Testing ui.utils...")
    from ui.utils import initialize_session_state, format_percentage
    print("   ✅ UI utilities imported successfully")

    print()
    print("=" * 60)
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("=" * 60)
    print()
    print("The UI is ready to launch. Run:")
    print("  streamlit run app.py")
    print()

except ImportError as e:
    print()
    print("=" * 60)
    print("❌ IMPORT FAILED!")
    print("=" * 60)
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)
