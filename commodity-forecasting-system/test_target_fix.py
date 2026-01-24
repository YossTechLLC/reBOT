#!/usr/bin/env python
"""
Test script to verify target column fix for HMM training.
Verifies that training works with feature subsetting.

Run with: python test_target_fix.py
"""

import sys
import os

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

print("=" * 70)
print("TESTING TARGET COLUMN FIX")
print("=" * 70)
print()

# Mock streamlit.session_state
class MockSessionState(dict):
    def get(self, key, default=None):
        return super().get(key, default)
    def __setattr__(self, key, value):
        self[key] = value
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'MockSessionState' has no attribute '{key}'")

import streamlit as st
st.session_state = MockSessionState()

from ui.model_controller import ModelController
from ui.data_manager import DataManager

print("Test 1: Load data and verify target column exists")
print("-" * 70)

dm = DataManager(
    alpaca_key='PKDTSYSP4AYPZDNELOHNRPW2BR',
    alpaca_secret='Ae8HNurExREVLghBw5dw9D3Pinkc2Kv8kNcJykMB3XQE'
)

spy, vix, features = dm.load_complete_dataset(days=180)

if 'intraday_range_pct' in features.columns:
    print(f"  ✅ Target column exists")
    print(f"     Range: {features['intraday_range_pct'].min():.4f} - {features['intraday_range_pct'].max():.4f}")
    print(f"     Mean: {features['intraday_range_pct'].mean():.4f}")
else:
    print(f"  ❌ Target column MISSING!")
    sys.exit(1)
print()

print("Test 2: Train HMM with full DataFrame")
print("-" * 70)

controller = ModelController()

try:
    model, metrics = controller.train_hmm(
        df=features,
        n_regimes=3,
        n_iter=100
    )
    print(f"  ✅ Training succeeded")
    print(f"     Converged: {metrics['converged']}")
    print(f"     Samples: {metrics['n_samples']}")
    print(f"     Regimes detected: {len(model.regime_labels)}")
    for label in model.regime_labels:
        vol = model.regime_volatilities[label]
        print(f"       {label}: {vol:.3f} ({vol*100:.2f}%)")
except Exception as e:
    print(f"  ❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print()
print("The target column fix is working correctly.")
print("HMM training now works with feature subsetting.")
print()
