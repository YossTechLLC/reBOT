#!/usr/bin/env python
"""
Test script to verify session state initialization fix.
Tests that ModelController properly handles None values in session state.

Run with: python test_session_state_fix.py
"""

import sys
import os

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

print("=" * 70)
print("TESTING SESSION STATE FIX")
print("=" * 70)
print()

# Mock streamlit.session_state for testing
class MockSessionState(dict):
    """Mock Streamlit session state for testing."""
    def get(self, key, default=None):
        return super().get(key, default)

    def __setattr__(self, key, value):
        """Support attribute-style assignment."""
        self[key] = value

    def __getattr__(self, key):
        """Support attribute-style access."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'MockSessionState' has no attribute '{key}'")

# Patch streamlit BEFORE importing ModelController
import streamlit as st
mock_session_state = MockSessionState()
st.session_state = mock_session_state

print("Test 1: Initialize session state with None values (simulates bug)")
print("-" * 70)

# Simulate what initialize_session_state() does
st.session_state['hmm_model'] = None
st.session_state['hmm_metrics'] = None
st.session_state['timesfm_forecaster'] = None
st.session_state['confidence_scorer'] = None

print(f"  confidence_scorer before ModelController: {st.session_state['confidence_scorer']}")
print()

print("Test 2: Create ModelController (this should create confidence_scorer)")
print("-" * 70)

from ui.model_controller import ModelController

controller = ModelController()

print(f"  confidence_scorer after ModelController: {st.session_state['confidence_scorer']}")
print(f"  Type: {type(st.session_state['confidence_scorer'])}")

# Verify it was created
if st.session_state['confidence_scorer'] is None:
    print("  ❌ FAIL: confidence_scorer is still None!")
    sys.exit(1)
else:
    print("  ✅ PASS: confidence_scorer was created!")
print()

print("Test 3: Verify confidence_scorer has threshold attribute")
print("-" * 70)

try:
    threshold = st.session_state.confidence_scorer.threshold
    print(f"  ✅ PASS: threshold = {threshold}")
except AttributeError as e:
    print(f"  ❌ FAIL: {e}")
    sys.exit(1)
print()

print("Test 4: Test set_confidence_threshold()")
print("-" * 70)

try:
    controller.set_confidence_threshold(50.0)
    new_threshold = st.session_state.confidence_scorer.threshold
    if new_threshold == 50.0:
        print(f"  ✅ PASS: threshold updated to {new_threshold}")
    else:
        print(f"  ❌ FAIL: threshold is {new_threshold}, expected 50.0")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("Test 5: Test set_confidence_weights()")
print("-" * 70)

try:
    controller.set_confidence_weights(0.5, 0.3, 0.2)
    scorer = st.session_state.confidence_scorer
    if scorer.regime_weight == 0.5 and scorer.timesfm_weight == 0.3 and scorer.feature_weight == 0.2:
        print(f"  ✅ PASS: weights updated correctly")
        print(f"     regime_weight: {scorer.regime_weight}")
        print(f"     timesfm_weight: {scorer.timesfm_weight}")
        print(f"     feature_weight: {scorer.feature_weight}")
    else:
        print(f"  ❌ FAIL: weights not set correctly")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("Test 6: Test defensive check (confidence_scorer set to None)")
print("-" * 70)

# Simulate scorer being None again
st.session_state['confidence_scorer'] = None
print(f"  Set confidence_scorer to None")

try:
    controller.set_confidence_threshold(60.0)
    if st.session_state['confidence_scorer'] is not None:
        print(f"  ✅ PASS: Defensive check created new scorer")
        print(f"     threshold = {st.session_state.confidence_scorer.threshold}")
    else:
        print(f"  ❌ FAIL: confidence_scorer is still None")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print()
print("The session state fix is working correctly.")
print("The UI should now launch without AttributeError.")
print()
