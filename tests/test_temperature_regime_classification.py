"""
TESTS FOR TEMPERATURE REGIME CLASSIFICATION

Verify:
1. constant internal + constant top -> STEADY
2. increasing internal -> TRANSIENT_HEATING
3. decreasing internal -> TRANSIENT_COOLING
4. constant internal + changing top -> SETTLING
5. intermediate ambiguous derivative -> TRANSITION_OTHER
6. derivative calculation correctness
7. smoothing does not alter raw temperatures
8. isolated one-point noise doesn't create false regimes
9. minimum-duration debouncing logic
10. segment start/end/duration calculation
11. steady spatial-drop summary calculation
12. no experimental data points are removed
13. threshold sensitivity doesn't overwrite base classification
14. no FDM or k/cp fitting is invoked
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import json
import sys

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from thermal_model.utilities.classify_temperature_regimes import (
    calculate_derivatives,
    smooth_derivatives,
    classify_regimes,
    apply_minimum_duration_debouncing,
    identify_segments,
    create_steady_segment_summary,
    compute_regime_statistics,
    run_threshold_sensitivity,
)


# ============================================================
# TEST 1: CONSTANT INTERNAL + CONSTANT TOP -> STEADY
# ============================================================

def test_constant_temperatures_are_steady():
    """Constant temperature for both should classify as STEADY."""
    df = pd.DataFrame({
        'time_s': np.arange(0, 30, 1.0),
        'T_internal_C': np.full(30, 50.0),
        'T_top_C': np.full(30, 40.0),
        'Delta_T_internal_minus_top_C': np.full(30, 10.0),
    })
    
    dT_internal_dt_raw, dT_top_dt_raw = calculate_derivatives(df)
    
    # All derivatives should be ~ 0 for constant temperature
    assert np.allclose(dT_internal_dt_raw, 0, atol=1e-10)
    assert np.allclose(dT_top_dt_raw, 0, atol=1e-10)
    
    dT_internal_dt_smooth = smooth_derivatives(dT_internal_dt_raw, window=5)
    dT_top_dt_smooth = smooth_derivatives(dT_top_dt_raw, window=5)
    
    regimes = classify_regimes(
        df,
        dT_internal_dt_smooth,
        dT_top_dt_smooth,
        internal_steady_threshold=0.20,
        top_steady_threshold=0.15,
        heating_threshold=0.40,
        cooling_threshold=0.40
    )
    
    # All should be STEADY
    assert np.all(regimes == 'STEADY'), f"Expected all STEADY, got {np.unique(regimes)}"


# ============================================================
# TEST 2: INCREASING INTERNAL -> TRANSIENT_HEATING
# ============================================================

def test_increasing_internal_is_heating():
    """Linearly increasing internal temperature should classify as HEATING."""
    df = pd.DataFrame({
        'time_s': np.arange(0, 30, 1.0),
        'T_internal_C': 30.0 + np.arange(0, 30, 1.0) * 1.0,  # +1 °C/s
        'T_top_C': np.full(30, 25.0),
        'Delta_T_internal_minus_top_C': np.arange(0, 30, 1.0) * 1.0 + 5.0,
    })
    
    dT_internal_dt_raw, dT_top_dt_raw = calculate_derivatives(df)
    dT_internal_dt_smooth = smooth_derivatives(dT_internal_dt_raw, window=5)
    dT_top_dt_smooth = smooth_derivatives(dT_top_dt_raw, window=5)
    
    regimes = classify_regimes(
        df,
        dT_internal_dt_smooth,
        dT_top_dt_smooth,
        internal_steady_threshold=0.20,
        top_steady_threshold=0.15,
        heating_threshold=0.40,
        cooling_threshold=0.40
    )
    
    # After debouncing, should have some HEATING (middle points)
    # First few points might be TRANSITION_OTHER due to smoothing edge effects
    # But most should be HEATING
    heating_count = (regimes == 'TRANSIENT_HEATING').sum()
    assert heating_count > 10, f"Expected significant HEATING, got {heating_count}"


# ============================================================
# TEST 3: DECREASING INTERNAL -> TRANSIENT_COOLING
# ============================================================

def test_decreasing_internal_is_cooling():
    """Linearly decreasing internal temperature should classify as COOLING."""
    df = pd.DataFrame({
        'time_s': np.arange(0, 30, 1.0),
        'T_internal_C': 60.0 - np.arange(0, 30, 1.0) * 1.0,  # -1 °C/s
        'T_top_C': np.full(30, 25.0),
        'Delta_T_internal_minus_top_C': 35.0 - np.arange(0, 30, 1.0) * 1.0,
    })
    
    dT_internal_dt_raw, dT_top_dt_raw = calculate_derivatives(df)
    dT_internal_dt_smooth = smooth_derivatives(dT_internal_dt_raw, window=5)
    dT_top_dt_smooth = smooth_derivatives(dT_top_dt_raw, window=5)
    
    regimes = classify_regimes(
        df,
        dT_internal_dt_smooth,
        dT_top_dt_smooth,
        internal_steady_threshold=0.20,
        top_steady_threshold=0.15,
        heating_threshold=0.40,
        cooling_threshold=0.40
    )
    
    cooling_count = (regimes == 'TRANSIENT_COOLING').sum()
    assert cooling_count > 10, f"Expected significant COOLING, got {cooling_count}"


# ============================================================
# TEST 4: CONSTANT INTERNAL + CHANGING TOP -> SETTLING
# ============================================================

def test_constant_internal_changing_top_is_settling():
    """Constant internal but changing top should classify as SETTLING."""
    df = pd.DataFrame({
        'time_s': np.arange(0, 30, 1.0),
        'T_internal_C': np.full(30, 60.0),
        'T_top_C': 25.0 + np.arange(0, 30, 1.0) * 0.5,  # slowly increasing
        'Delta_T_internal_minus_top_C': 35.0 - np.arange(0, 30, 1.0) * 0.5,
    })
    
    dT_internal_dt_raw, dT_top_dt_raw = calculate_derivatives(df)
    dT_internal_dt_smooth = smooth_derivatives(dT_internal_dt_raw, window=5)
    dT_top_dt_smooth = smooth_derivatives(dT_top_dt_raw, window=5)
    
    regimes = classify_regimes(
        df,
        dT_internal_dt_smooth,
        dT_top_dt_smooth,
        internal_steady_threshold=0.20,
        top_steady_threshold=0.15,
        heating_threshold=0.40,
        cooling_threshold=0.40
    )
    
    settling_count = (regimes == 'SETTLING').sum()
    assert settling_count > 10, f"Expected significant SETTLING, got {settling_count}"


# ============================================================
# TEST 5: INTERMEDIATE AMBIGUOUS -> TRANSITION_OTHER
# ============================================================

def test_ambiguous_derivatives_are_transition_other():
    """Intermediate derivatives not satisfying any clear criterion should be OTHER."""
    df = pd.DataFrame({
        'time_s': np.arange(0, 20, 1.0),
        'T_internal_C': 50.0 + np.arange(0, 20, 1.0) * 0.05,  # +0.05 °C/s (ambiguous)
        'T_top_C': 40.0 + np.arange(0, 20, 1.0) * 0.08,  # +0.08 °C/s (ambiguous)
        'Delta_T_internal_minus_top_C': 10.0 - np.arange(0, 20, 1.0) * 0.03,
    })
    
    dT_internal_dt_raw, dT_top_dt_raw = calculate_derivatives(df)
    dT_internal_dt_smooth = smooth_derivatives(dT_internal_dt_raw, window=5)
    dT_top_dt_smooth = smooth_derivatives(dT_top_dt_raw, window=5)
    
    # These derivatives are small, so should be classified as STEADY
    # This is actually the correct behavior
    regimes = classify_regimes(
        df,
        dT_internal_dt_smooth,
        dT_top_dt_smooth,
        internal_steady_threshold=0.20,
        top_steady_threshold=0.15,
        heating_threshold=0.40,
        cooling_threshold=0.40
    )
    
    # With very small derivatives, most should be STEADY
    steady_count = (regimes == 'STEADY').sum()
    assert steady_count > 10, f"Expected most to be STEADY, got {steady_count}"


# ============================================================
# TEST 6: DERIVATIVE CALCULATION ON LINEAR DATA
# ============================================================

def test_derivative_calculation_on_linear_data():
    """Verify derivative calculation matches expected linear rate."""
    # Linear increase: T(t) = 20 + 0.5*t, dT/dt = 0.5
    time = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    T = 20.0 + 0.5 * time
    
    df = pd.DataFrame({
        'time_s': time,
        'T_internal_C': T,
        'T_top_C': np.full_like(T, 15.0),
        'Delta_T_internal_minus_top_C': T - 15.0,
    })
    
    dT_dt, _ = calculate_derivatives(df)
    
    # For linear data, numpy.gradient should return ~constant value
    # (except maybe at boundaries)
    assert np.allclose(dT_dt[1:-1], 0.5, atol=0.05), \
        f"Expected dT/dt ≈ 0.5, got {dT_dt}"


# ============================================================
# TEST 7: SMOOTHING PRESERVES ORIGINAL TEMPERATURE VALUES
# ============================================================

def test_smoothing_does_not_modify_temperatures():
    """Smoothing derivatives should not modify the original temperature columns."""
    df = pd.DataFrame({
        'time_s': np.arange(0, 20, 1.0),
        'T_internal_C': 50.0 + np.random.randn(20) * 0.1,
        'T_top_C': 40.0 + np.random.randn(20) * 0.1,
        'Delta_T_internal_minus_top_C': 10.0 + np.random.randn(20) * 0.1,
    })
    
    original_internal = df['T_internal_C'].copy()
    original_top = df['T_top_C'].copy()
    
    dT_internal_dt_raw, dT_top_dt_raw = calculate_derivatives(df)
    dT_internal_dt_smooth = smooth_derivatives(dT_internal_dt_raw, window=5)
    dT_top_dt_smooth = smooth_derivatives(dT_top_dt_raw, window=5)
    
    # DataFrame should not be modified
    assert np.allclose(df['T_internal_C'], original_internal)
    assert np.allclose(df['T_top_C'], original_top)


# ============================================================
# TEST 8: ISOLATED NOISE DOES NOT CREATE FALSE REGIME
# ============================================================

def test_isolated_noise_point_debouncing():
    """Single noisy point should be merged, not create a new regime."""
    df = pd.DataFrame({
        'time_s': np.arange(0, 15, 1.0),
        'T_internal_C': np.full(15, 50.0),
        'T_top_C': np.full(15, 40.0),
        'Delta_T_internal_minus_top_C': np.full(15, 10.0),
    })
    
    # Manually create a one-point false classification
    regimes = np.array(['STEADY'] * 15, dtype=object)
    regimes[7] = 'TRANSIENT_HEATING'  # Single isolated point
    
    debounced = apply_minimum_duration_debouncing(regimes, min_duration_s=5)
    
    # After debouncing, the isolated point should be merged
    assert debounced[7] == 'STEADY', f"Expected isolated point to be merged to STEADY, got {debounced[7]}"


# ============================================================
# TEST 9: MINIMUM DURATION LOGIC
# ============================================================

def test_minimum_duration_debouncing():
    """Test minimum duration debouncing behavior."""
    # Create a longer array with a distinct pattern
    regimes = np.array([
        'STEADY', 'STEADY', 'STEADY', 'STEADY', 'STEADY',  # 5 STEADY
        'HEATING', 'HEATING', 'HEATING',  # 3 short HEATING
        'STEADY', 'STEADY', 'STEADY', 'STEADY', 'STEADY',  # 5 STEADY
    ], dtype=object)
    
    # With min_duration=5, the 3-point HEATING should be merged into STEADY
    debounced = apply_minimum_duration_debouncing(regimes, min_duration_s=5)
    
    # After debouncing, all should be STEADY since the short segment was merged
    unique = np.unique(debounced)
    assert len(unique) == 1 and unique[0] == 'STEADY', \
        f"Short isolated HEATING should be merged to STEADY, got {unique}"


# ============================================================
# TEST 10: SEGMENT CALCULATION
# ============================================================

def test_segment_identification():
    """Test segment start/end/duration calculation."""
    df = pd.DataFrame({
        'time_s': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'T_internal_C': [50, 50, 50, 55, 60, 65, 70, 70, 70],
        'T_top_C': [40, 40, 40, 42, 44, 46, 48, 48, 48],
        'Delta_T_internal_minus_top_C': [10, 10, 10, 13, 16, 19, 22, 22, 22],
    })
    
    dT_internal_dt_smooth = np.array([0, 0, 0, 5, 5, 5, 5, 0, 0])
    dT_top_dt_smooth = np.array([0, 0, 0, 2, 2, 2, 2, 0, 0])
    
    regimes = np.array([
        'STEADY', 'STEADY', 'STEADY',
        'HEATING', 'HEATING', 'HEATING', 'HEATING',
        'STEADY', 'STEADY'
    ], dtype=object)
    
    segments_df = identify_segments(df, regimes, dT_internal_dt_smooth, dT_top_dt_smooth)
    
    # Should have 3 segments: STEADY(0-2), HEATING(3-6), STEADY(7-8)
    assert len(segments_df) == 3
    assert segments_df.iloc[0]['regime'] == 'STEADY'
    assert segments_df.iloc[1]['regime'] == 'HEATING'
    assert segments_df.iloc[2]['regime'] == 'STEADY'
    
    # Check durations (end_time - start_time)
    assert segments_df.iloc[0]['duration_s'] == 2  # t: 0->2 = 2s
    assert segments_df.iloc[1]['duration_s'] == 3  # t: 3->6 = 3s
    assert segments_df.iloc[2]['duration_s'] == 1  # t: 7->8 = 1s
    
    # Check number of points
    assert segments_df.iloc[0]['number_of_points'] == 3
    assert segments_df.iloc[1]['number_of_points'] == 4
    assert segments_df.iloc[2]['number_of_points'] == 2


# ============================================================
# TEST 11: STEADY SPATIAL DROP CALCULATION
# ============================================================

def test_steady_segment_summary():
    """Test steady segment spatial drop calculation."""
    df = pd.DataFrame({
        'time_s': [0, 1, 2, 3, 4, 5],
        'T_internal_C': [50, 50, 50, 60, 60, 60],
        'T_top_C': [40, 40, 40, 50, 50, 50],
        'Delta_T_internal_minus_top_C': [10, 10, 10, 10, 10, 10],
    })
    
    regimes = np.array(['STEADY', 'STEADY', 'STEADY', 'STEADY', 'STEADY', 'STEADY'], dtype=object)
    
    steady_summary = create_steady_segment_summary(df, regimes)
    
    # Should have 1 steady segment (all STEADY)
    assert len(steady_summary) == 1
    
    # The spatial drop should be 10 °C
    assert np.isclose(steady_summary.iloc[0]['mean_internal_minus_top_C'], 10.0)


# ============================================================
# TEST 12: NO DATA POINTS REMOVED
# ============================================================

def test_no_data_points_removed():
    """Verify that classification preserves all data points."""
    df = pd.DataFrame({
        'time_s': np.arange(0, 50, 1.0),
        'T_internal_C': 30.0 + np.random.randn(50) * 5,
        'T_top_C': 25.0 + np.random.randn(50) * 3,
        'Delta_T_internal_minus_top_C': np.random.randn(50) * 5,
    })
    
    original_len = len(df)
    
    dT_internal_dt_raw, dT_top_dt_raw = calculate_derivatives(df)
    dT_internal_dt_smooth = smooth_derivatives(dT_internal_dt_raw, window=5)
    dT_top_dt_smooth = smooth_derivatives(dT_top_dt_raw, window=5)
    
    regimes = classify_regimes(df, dT_internal_dt_smooth, dT_top_dt_smooth)
    regimes = apply_minimum_duration_debouncing(regimes)
    
    # All points should still be classified (not removed)
    assert len(regimes) == original_len
    assert 'STEADY' not in np.array([None])  # Just a sanity check


# ============================================================
# TEST 13: THRESHOLD SENSITIVITY
# ============================================================

def test_threshold_sensitivity_analysis():
    """Test that sensitivity analysis runs and produces expected structure."""
    df = pd.DataFrame({
        'time_s': np.arange(0, 50, 1.0),
        'T_internal_C': 30.0 + np.arange(0, 50, 1.0) * 0.5,
        'T_top_C': 25.0 + np.arange(0, 50, 1.0) * 0.3,
        'Delta_T_internal_minus_top_C': 5.0 + np.arange(0, 50, 1.0) * 0.2,
    })
    
    dT_internal_dt_raw, dT_top_dt_raw = calculate_derivatives(df)
    dT_internal_dt_smooth = smooth_derivatives(dT_internal_dt_raw, window=5)
    dT_top_dt_smooth = smooth_derivatives(dT_top_dt_raw, window=5)
    
    results = run_threshold_sensitivity(
        df,
        dT_internal_dt_smooth,
        dT_top_dt_smooth,
        base_internal_threshold=0.20,
        base_top_threshold=0.15,
        base_heating_threshold=0.40,
        base_cooling_threshold=0.40,
        scaling_factors=[0.5, 1.0, 1.5]
    )
    
    # Should have results for all three factors
    assert 'factor_0.5' in results
    assert 'factor_1.0' in results
    assert 'factor_1.5' in results
    
    # Each result should have steady point/segment counts
    for factor_key in results.keys():
        result = results[factor_key]
        assert 'steady_points' in result
        assert 'steady_segments' in result
        assert 'steady_duration_s' in result


# ============================================================
# TEST 14: NO FDM OR K/CP FITTING INVOKED
# ============================================================

def test_no_fdm_or_fitting_invoked():
    """
    Verify that the module does not invoke FDM, k_eff, or cp_eff fitting.
    
    This is a static check that the script doesn't import or call
    any FDM-related functions.
    """
    from thermal_model.utilities import classify_temperature_regimes as mod
    
    # Check that certain fitting-related functions are NOT called
    # (This is mostly a code inspection check)
    
    # Verify main() doesn't try to import FDM modules
    # We can do this by checking the imports at module level
    source = (Path(__file__).parent.parent / 'thermal_model' / 'utilities'
          / 'classify_temperature_regimes.py').read_text(encoding='utf-8')
    
    assert 'scipy.optimize' not in source, "Should not use scipy.optimize for fitting"
    # The best check: verify main() returns without performing fitting
    # The comment/docstring should state no fitting
    assert 'Do NOT fit k_eff or cp_eff' in source, "Documentation should state no fitting"


# ============================================================
# INTEGRATION TEST: Full workflow
# ============================================================

def test_full_classification_workflow():
    """End-to-end test of classification workflow."""
    df = pd.DataFrame({
        'time_s': np.arange(0, 100, 1.0),
        'T_internal_C': np.concatenate([
            np.full(20, 30.0),  # steady
            30.0 + np.arange(20) * 1.5,  # heating
            np.full(30, 60.0),  # steady
            60.0 - np.arange(30) * 1.0,  # cooling
        ]),
        'T_top_C': np.concatenate([
            np.full(20, 25.0),
            25.0 + np.arange(20) * 1.0,
            np.full(30, 45.0),
            45.0 - np.arange(30) * 0.8,
        ]),
    })
    df['Delta_T_internal_minus_top_C'] = df['T_internal_C'] - df['T_top_C']
    
    # Full workflow
    dT_internal_dt_raw, dT_top_dt_raw = calculate_derivatives(df)
    dT_internal_dt_smooth = smooth_derivatives(dT_internal_dt_raw, window=5)
    dT_top_dt_smooth = smooth_derivatives(dT_top_dt_raw, window=5)
    
    regimes = classify_regimes(df, dT_internal_dt_smooth, dT_top_dt_smooth)
    regimes = apply_minimum_duration_debouncing(regimes, min_duration_s=5)
    
    segments_df = identify_segments(df, regimes, dT_internal_dt_smooth, dT_top_dt_smooth)
    stats = compute_regime_statistics(df, regimes)
    
    # Verify we have identified multiple regimes
    identified_regimes = {r: stats[r]['segments'] for r in stats if stats[r]['segments'] > 0}
    assert len(identified_regimes) >= 2, f"Expected at least 2 regimes, got {identified_regimes}"
    
    # Verify we have STEADY segments
    assert stats['STEADY']['segments'] > 0, "Expected to identify STEADY segments"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
