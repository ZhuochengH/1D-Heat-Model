#!/usr/bin/env python3
"""
TEMPERATURE REGIME CLASSIFICATION FOR ALIGNED 72°C EXPERIMENT

Purpose:
    Classify already time-aligned experimental data into physically 
    interpretable temperature regimes: STEADY, TRANSIENT_HEATING, 
    TRANSIENT_COOLING, SETTLING, TRANSITION_OTHER.

    Classification is for interpretation and diagnostics ONLY.
    All aligned data points are preserved (no removal).
    No fitting of k_eff or cp_eff is performed.

Input:
    Aligned CSV from temperature_alignment_output/72C/aligned_internal_top_temperature.csv

Output:
    - temperature_regime_labeled.csv (pointwise labels)
    - temperature_regime_segments.csv (segment summary)
    - steady_segment_summary.csv (steady segments only)
    - regime_classification_metadata.json
    - temperature_regime_classification.png (temperature + regime visualization)
    - temperature_derivative_diagnostic.png (derivatives + thresholds)
    - steady_spatial_temperature_drop.png (steady spatial drop analysis)

Do NOT run FDM.
Do NOT fit k_eff or cp_eff.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# SETUP LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# DATA LOADING
# ============================================================

def load_aligned_data(csv_path: str) -> pd.DataFrame:
    """Load aligned internal-top temperature CSV."""
    df = pd.read_csv(csv_path)
    
    required_cols = ['time_s', 'T_internal_interpolated_C', 'T_top_measured_C']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Rename for convenience
    df = df.rename(columns={
        'T_internal_interpolated_C': 'T_internal_C',
        'T_top_measured_C': 'T_top_C'
    })
    
    return df


# ============================================================
# DERIVATIVE CALCULATION
# ============================================================

def calculate_derivatives(
    df: pd.DataFrame,
    time_col: str = 'time_s',
    T_internal_col: str = 'T_internal_C',
    T_top_col: str = 'T_top_C'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate raw temperature derivatives using numpy.gradient.
    
    Returns:
        (dT_internal/dt, dT_top/dt) in °C/s
    """
    time = df[time_col].values
    T_internal = df[T_internal_col].values
    T_top = df[T_top_col].values
    
    # numpy.gradient handles non-uniform spacing and boundary conditions
    dT_internal_dt = np.gradient(T_internal, time)
    dT_top_dt = np.gradient(T_top, time)
    
    return dT_internal_dt, dT_top_dt


# ============================================================
# SMOOTHING
# ============================================================

def smooth_derivatives(
    derivatives: np.ndarray,
    window: int = 5
) -> np.ndarray:
    """Apply light centered rolling mean to derivatives."""
    s = pd.Series(derivatives)
    smoothed = s.rolling(window=window, center=True, min_periods=1).mean().values
    return smoothed


# ============================================================
# REGIME CLASSIFICATION
# ============================================================

def classify_regimes(
    df: pd.DataFrame,
    dT_internal_dt_smooth: np.ndarray,
    dT_top_dt_smooth: np.ndarray,
    internal_steady_threshold: float = 0.20,
    top_steady_threshold: float = 0.15,
    heating_threshold: float = 0.40,
    cooling_threshold: float = 0.40
) -> np.ndarray:
    """
    Pointwise regime classification based on smoothed derivatives.
    
    Returns:
        Array of regime labels
    """
    n = len(df)
    regimes = np.array(['TRANSITION_OTHER'] * n, dtype=object)
    
    for i in range(n):
        dT_int = dT_internal_dt_smooth[i]
        dT_top = dT_top_dt_smooth[i]
        
        # Priority-based classification
        if dT_int >= heating_threshold:
            regimes[i] = 'TRANSIENT_HEATING'
        elif dT_int <= -cooling_threshold:
            regimes[i] = 'TRANSIENT_COOLING'
        elif abs(dT_int) <= internal_steady_threshold and \
             abs(dT_top) <= top_steady_threshold:
            regimes[i] = 'STEADY'
        elif abs(dT_int) <= internal_steady_threshold and \
             abs(dT_top) > top_steady_threshold:
            regimes[i] = 'SETTLING'
        # else: TRANSITION_OTHER (already set)
    
    return regimes


# ============================================================
# DEBOUNCING: MINIMUM DURATION
# ============================================================

def apply_minimum_duration_debouncing(
    regimes: np.ndarray,
    min_duration_s: int = 5
) -> np.ndarray:
    """
    Merge short regime segments (< min_duration_s) into neighboring regimes
    or mark as TRANSITION_OTHER.
    
    Time spacing is assumed to be ~1 s, so min_duration_s ≈ min_duration_points.
    """
    min_points = min_duration_s
    debounced = regimes.copy()
    
    i = 0
    while i < len(debounced):
        regime_i = debounced[i]
        
        # Skip if already TRANSITION_OTHER
        if regime_i == 'TRANSITION_OTHER':
            i += 1
            continue
        
        # Count consecutive points with same regime
        j = i + 1
        while j < len(debounced) and debounced[j] == regime_i:
            j += 1
        
        duration = j - i
        
        # If too short, try to merge into neighbor
        if duration < min_points:
            if i > 0 and i + duration < len(debounced):
                # Merge into the more "active" neighbor
                left_regime = debounced[i - 1]
                right_regime = debounced[i + duration]
                
                # Prefer merging with non-TRANSITION regimes
                if left_regime != 'TRANSITION_OTHER':
                    debounced[i:j] = left_regime
                elif right_regime != 'TRANSITION_OTHER':
                    debounced[i:j] = right_regime
                # else mark as TRANSITION_OTHER (already done)
            elif i > 0:
                # Only left neighbor
                debounced[i:j] = debounced[i - 1]
            elif i + duration < len(debounced):
                # Only right neighbor
                debounced[i:j] = debounced[i + duration]
        
        i = j
    
    return debounced


# ============================================================
# SEGMENT ANALYSIS
# ============================================================

def identify_segments(
    df: pd.DataFrame,
    regimes: np.ndarray,
    dT_internal_dt_smooth: np.ndarray,
    dT_top_dt_smooth: np.ndarray
) -> pd.DataFrame:
    """
    Convert pointwise regimes into segment-level summary.
    """
    segments = []
    segment_id = 0
    
    i = 0
    while i < len(df):
        regime = regimes[i]
        start_idx = i
        
        # Find end of this regime
        while i < len(df) and regimes[i] == regime:
            i += 1
        
        end_idx = i - 1
        segment_data = df.iloc[start_idx:end_idx+1]
        
        seg = {
            'segment_id': segment_id,
            'regime': regime,
            'start_time_s': segment_data['time_s'].iloc[0],
            'end_time_s': segment_data['time_s'].iloc[-1],
            'duration_s': segment_data['time_s'].iloc[-1] - segment_data['time_s'].iloc[0],
            'number_of_points': len(segment_data),
            
            'mean_T_internal_C': segment_data['T_internal_C'].mean(),
            'mean_T_top_C': segment_data['T_top_C'].mean(),
            'mean_Delta_T_C': segment_data['Delta_T_internal_minus_top_C'].mean(),
            
            'min_T_internal_C': segment_data['T_internal_C'].min(),
            'max_T_internal_C': segment_data['T_internal_C'].max(),
            
            'min_T_top_C': segment_data['T_top_C'].min(),
            'max_T_top_C': segment_data['T_top_C'].max(),
            
            'mean_dT_internal_dt_C_per_s': dT_internal_dt_smooth[start_idx:end_idx+1].mean(),
            'mean_dT_top_dt_C_per_s': dT_top_dt_smooth[start_idx:end_idx+1].mean(),
        }
        
        # For STEADY segments, explicitly report steady spatial drop
        if regime == 'STEADY':
            seg['steady_spatial_drop_C'] = seg['mean_Delta_T_C']
        
        segments.append(seg)
        segment_id += 1
    
    return pd.DataFrame(segments)


# ============================================================
# STEADY SEGMENT SUMMARY
# ============================================================

def create_steady_segment_summary(
    df: pd.DataFrame,
    regimes: np.ndarray
) -> pd.DataFrame:
    """Extract STEADY segment details for reference."""
    steady_mask = regimes == 'STEADY'
    
    summaries = []
    segment_id = 0
    
    i = 0
    while i < len(df):
        if regimes[i] == 'STEADY':
            start_idx = i
            while i < len(df) and regimes[i] == 'STEADY':
                i += 1
            end_idx = i - 1
            
            segment_data = df.iloc[start_idx:end_idx+1]
            
            summaries.append({
                'segment_id': segment_id,
                'start_time_s': segment_data['time_s'].iloc[0],
                'end_time_s': segment_data['time_s'].iloc[-1],
                'duration_s': segment_data['time_s'].iloc[-1] - segment_data['time_s'].iloc[0],
                
                'mean_T_internal_C': segment_data['T_internal_C'].mean(),
                'SD_T_internal_C': segment_data['T_internal_C'].std(),
                
                'mean_T_top_C': segment_data['T_top_C'].mean(),
                'SD_T_top_C': segment_data['T_top_C'].std(),
                
                'mean_internal_minus_top_C': segment_data['Delta_T_internal_minus_top_C'].mean(),
                'SD_internal_minus_top_C': segment_data['Delta_T_internal_minus_top_C'].std(),
            })
            segment_id += 1
        else:
            i += 1
    
    return pd.DataFrame(summaries)


# ============================================================
# REGIME STATISTICS
# ============================================================

def compute_regime_statistics(df: pd.DataFrame, regimes: np.ndarray) -> Dict:
    """
    Count points and segments by regime.
    
    Duration is computed using interval-based accounting:
    for each point i in regime, sum the interval from time[i] to time[i+1].
    
    This ensures:
        sum(regime_interval_durations) = time[-1] - time[0]
    
    and avoids double-counting or miscounting gaps between non-contiguous
    segments of the same regime.
    """
    time_arr = df['time_s'].values
    
    stats = {}
    for regime_name in ['STEADY', 'TRANSIENT_HEATING', 'TRANSIENT_COOLING', 
                        'SETTLING', 'TRANSITION_OTHER']:
        mask = regimes == regime_name
        points = mask.sum()
        
        # Calculate interval-based duration: sum of intervals [time[i], time[i+1])
        # for every point i that belongs to this regime
        interval_duration = 0.0
        if points > 0:
            for i in range(len(time_arr) - 1):
                if mask[i]:
                    interval_duration += time_arr[i + 1] - time_arr[i]
        
        # Count segments (contiguous runs of same regime)
        segments = 0
        in_regime = False
        for i in range(len(regimes)):
            if regimes[i] == regime_name:
                if not in_regime:
                    segments += 1
                    in_regime = True
            else:
                in_regime = False
        
        stats[regime_name] = {
            'points': int(points),
            'interval_duration_s': float(interval_duration),
            'segments': int(segments)
        }
    
    return stats


# ============================================================
# THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

def run_threshold_sensitivity(
    df: pd.DataFrame,
    dT_internal_dt_smooth: np.ndarray,
    dT_top_dt_smooth: np.ndarray,
    base_internal_threshold: float = 0.20,
    base_top_threshold: float = 0.15,
    base_heating_threshold: float = 0.40,
    base_cooling_threshold: float = 0.40,
    scaling_factors: List[float] = [0.5, 1.0, 1.5]
) -> Dict:
    """Run sensitivity analysis with scaled thresholds."""
    results = {}
    
    for factor in scaling_factors:
        label = f'factor_{factor}'
        
        regimes = classify_regimes(
            df,
            dT_internal_dt_smooth,
            dT_top_dt_smooth,
            internal_steady_threshold=base_internal_threshold * factor,
            top_steady_threshold=base_top_threshold * factor,
            heating_threshold=base_heating_threshold * factor,
            cooling_threshold=base_cooling_threshold * factor
        )
        
        regimes = apply_minimum_duration_debouncing(regimes)
        stats = compute_regime_statistics(df, regimes)
        
        results[label] = {
            'steady_points': stats['STEADY']['points'],
            'steady_segments': stats['STEADY']['segments'],
            'steady_duration_s': stats['STEADY']['interval_duration_s'],
        }
    
    return results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_regime_classification(
    df: pd.DataFrame,
    regimes: np.ndarray,
    output_path: str
):
    """
    Plot temperature curves with regime background shading.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    time = df['time_s'].values
    T_internal = df['T_internal_C'].values
    T_top = df['T_top_C'].values
    
    # Define colors for regimes
    regime_colors = {
        'STEADY': '#90EE90',  # light green
        'TRANSIENT_HEATING': '#FFB6C1',  # light red
        'TRANSIENT_COOLING': '#87CEEB',  # light blue
        'SETTLING': '#FFE4B5',  # moccasin
        'TRANSITION_OTHER': '#D3D3D3',  # light gray
    }
    
    # Add background shading for each regime
    current_regime = regimes[0]
    start_time = time[0]
    
    for i in range(1, len(time)):
        if regimes[i] != current_regime:
            # Shade from start_time to time[i-1]
            ax.axvspan(start_time, time[i-1], 
                       alpha=0.2, color=regime_colors.get(current_regime, 'gray'))
            
            current_regime = regimes[i]
            start_time = time[i]
    
    # Final regime
    ax.axvspan(start_time, time[-1], 
               alpha=0.2, color=regime_colors.get(current_regime, 'gray'))
    
    # Plot temperature curves
    ax.plot(time, T_internal, 'r-', linewidth=2, label='T_internal (interpolated)')
    ax.plot(time, T_top, 'b-', linewidth=2, label='T_top (measured)')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Temperature Regime Classification (72°C Experiment)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # Add custom legend for regimes
    from matplotlib.patches import Patch
    regime_patches = [
        Patch(facecolor=regime_colors['STEADY'], alpha=0.5, label='STEADY'),
        Patch(facecolor=regime_colors['TRANSIENT_HEATING'], alpha=0.5, label='TRANSIENT_HEATING'),
        Patch(facecolor=regime_colors['TRANSIENT_COOLING'], alpha=0.5, label='TRANSIENT_COOLING'),
        Patch(facecolor=regime_colors['SETTLING'], alpha=0.5, label='SETTLING'),
        Patch(facecolor=regime_colors['TRANSITION_OTHER'], alpha=0.5, label='TRANSITION_OTHER'),
    ]
    ax.legend(handles=regime_patches, loc='upper left', fontsize=9, title='Regimes')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved regime classification plot: {output_path}")


def plot_derivative_diagnostic(
    df: pd.DataFrame,
    dT_internal_dt_smooth: np.ndarray,
    dT_top_dt_smooth: np.ndarray,
    internal_steady_threshold: float = 0.20,
    top_steady_threshold: float = 0.15,
    heating_threshold: float = 0.40,
    cooling_threshold: float = 0.40,
    output_path: str = None
):
    """Plot derivatives with threshold reference lines."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    time = df['time_s'].values
    
    ax.plot(time, dT_internal_dt_smooth, 'r-', linewidth=2, label='dT_internal/dt (smoothed)')
    ax.plot(time, dT_top_dt_smooth, 'b-', linewidth=2, label='dT_top/dt (smoothed)')
    
    # Threshold lines
    ax.axhline(y=internal_steady_threshold, color='r', linestyle='--', 
               linewidth=1, alpha=0.7, label=f'Internal steady threshold ({internal_steady_threshold})')
    ax.axhline(y=-internal_steady_threshold, color='r', linestyle='--', 
               linewidth=1, alpha=0.7)
    
    ax.axhline(y=top_steady_threshold, color='b', linestyle='--', 
               linewidth=1, alpha=0.7, label=f'Top steady threshold ({top_steady_threshold})')
    ax.axhline(y=-top_steady_threshold, color='b', linestyle='--', 
               linewidth=1, alpha=0.7)
    
    ax.axhline(y=heating_threshold, color='orange', linestyle=':', 
               linewidth=1, alpha=0.7, label=f'Heating threshold ({heating_threshold})')
    ax.axhline(y=-cooling_threshold, color='cyan', linestyle=':', 
               linewidth=1, alpha=0.7, label=f'Cooling threshold ({-cooling_threshold})')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('dT/dt (°C/s)', fontsize=12)
    ax.set_title('Temperature Derivative Diagnostic', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved derivative diagnostic plot: {output_path}")


def plot_steady_spatial_drop(
    segments_df: pd.DataFrame,
    output_path: str
):
    """Plot steady spatial temperature drop vs internal temperature."""
    steady_segments = segments_df[segments_df['regime'] == 'STEADY']
    
    if len(steady_segments) == 0:
        logger.warning("No STEADY segments found; skipping spatial drop plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    T_internal = steady_segments['mean_T_internal_C'].values
    spatial_drop = steady_segments['steady_spatial_drop_C'].values
    
    ax.scatter(T_internal, spatial_drop, s=100, alpha=0.6, color='green', edgecolor='darkgreen', linewidth=2)
    
    # Add segment labels
    for idx, row in steady_segments.iterrows():
        ax.annotate(f"seg {row['segment_id']}", 
                   xy=(row['mean_T_internal_C'], row['steady_spatial_drop_C']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Mean T_internal (°C)', fontsize=12)
    ax.set_ylabel('Steady Spatial Drop ΔT = T_internal - T_top (°C)', fontsize=12)
    ax.set_title('Steady State Spatial Temperature Drop Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved spatial drop plot: {output_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Classify temperature regimes for aligned 72°C experiment.'
    )
    
    parser.add_argument(
        'input_csv',
        help='Path to aligned_internal_top_temperature.csv'
    )
    
    parser.add_argument(
        '--output-dir',
        default='temperature_regime_output/72C',
        help='Output directory (default: temperature_regime_output/72C)'
    )
    
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=5,
        help='Smoothing window for derivatives (seconds, default: 5)'
    )
    
    parser.add_argument(
        '--internal-steady-threshold',
        type=float,
        default=0.20,
        help='Internal temperature steady threshold (°C/s, default: 0.20)'
    )
    
    parser.add_argument(
        '--top-steady-threshold',
        type=float,
        default=0.15,
        help='Top temperature steady threshold (°C/s, default: 0.15)'
    )
    
    parser.add_argument(
        '--heating-threshold',
        type=float,
        default=0.40,
        help='Heating threshold (°C/s, default: 0.40)'
    )
    
    parser.add_argument(
        '--cooling-threshold',
        type=float,
        default=0.40,
        help='Cooling threshold (°C/s, default: 0.40)'
    )
    
    parser.add_argument(
        '--min-duration',
        type=int,
        default=5,
        help='Minimum regime duration (seconds, default: 5)'
    )
    
    args = parser.parse_args()
    
    # ========================================================
    # Load data
    # ========================================================
    
    logger.info(f"Loading aligned data from: {args.input_csv}")
    df = load_aligned_data(args.input_csv)
    logger.info(f"Loaded {len(df)} data points")
    
    # ========================================================
    # Calculate derivatives
    # ========================================================
    
    logger.info("Calculating temperature derivatives...")
    dT_internal_dt_raw, dT_top_dt_raw = calculate_derivatives(df)
    
    logger.info(f"Smoothing derivatives with window={args.smooth_window}s...")
    dT_internal_dt_smooth = smooth_derivatives(dT_internal_dt_raw, window=args.smooth_window)
    dT_top_dt_smooth = smooth_derivatives(dT_top_dt_raw, window=args.smooth_window)
    
    # ========================================================
    # Classify regimes
    # ========================================================
    
    logger.info("Classifying temperature regimes (preliminary)...")
    regimes = classify_regimes(
        df,
        dT_internal_dt_smooth,
        dT_top_dt_smooth,
        internal_steady_threshold=args.internal_steady_threshold,
        top_steady_threshold=args.top_steady_threshold,
        heating_threshold=args.heating_threshold,
        cooling_threshold=args.cooling_threshold
    )
    
    logger.info("Applying minimum duration debouncing...")
    regimes = apply_minimum_duration_debouncing(regimes, min_duration_s=args.min_duration)
    
    # ========================================================
    # Create output directory
    # ========================================================
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # ========================================================
    # Create labeled output CSV
    # ========================================================
    
    output_df = df.copy()
    output_df['dT_internal_dt_raw_C_per_s'] = dT_internal_dt_raw
    output_df['dT_top_dt_raw_C_per_s'] = dT_top_dt_raw
    output_df['dT_internal_dt_smooth_C_per_s'] = dT_internal_dt_smooth
    output_df['dT_top_dt_smooth_C_per_s'] = dT_top_dt_smooth
    output_df['regime'] = regimes
    
    labeled_csv_path = output_dir / 'temperature_regime_labeled.csv'
    output_df.to_csv(labeled_csv_path, index=False)
    logger.info(f"Saved labeled data: {labeled_csv_path}")
    
    # ========================================================
    # Segment analysis
    # ========================================================
    
    logger.info("Analyzing segments...")
    segments_df = identify_segments(df, regimes, dT_internal_dt_smooth, dT_top_dt_smooth)
    
    segments_csv_path = output_dir / 'temperature_regime_segments.csv'
    segments_df.to_csv(segments_csv_path, index=False)
    logger.info(f"Saved segment summary: {segments_csv_path}")
    
    # ========================================================
    # Steady segment summary
    # ========================================================
    
    steady_summary_df = create_steady_segment_summary(df, regimes)
    steady_summary_path = output_dir / 'steady_segment_summary.csv'
    steady_summary_df.to_csv(steady_summary_path, index=False)
    logger.info(f"Saved steady segment summary: {steady_summary_path}")
    
    # ========================================================
    # Regime statistics
    # ========================================================
    
    stats = compute_regime_statistics(df, regimes)
    logger.info("Regime statistics:")
    for regime, counts in stats.items():
        logger.info(f"  {regime}: {counts['points']} points, "
                   f"{counts['interval_duration_s']:.1f} s, {counts['segments']} segments")
    
    # ========================================================
    # Threshold sensitivity
    # ========================================================
    
    logger.info("Running threshold sensitivity analysis...")
    sensitivity_results = run_threshold_sensitivity(
        df,
        dT_internal_dt_smooth,
        dT_top_dt_smooth,
        base_internal_threshold=args.internal_steady_threshold,
        base_top_threshold=args.top_steady_threshold,
        base_heating_threshold=args.heating_threshold,
        base_cooling_threshold=args.cooling_threshold,
        scaling_factors=[0.5, 1.0, 1.5]
    )
    
    logger.info("Sensitivity results (STEADY regime):")
    for factor_label, result in sensitivity_results.items():
        logger.info(f"  {factor_label}: {result['steady_points']} points, "
                   f"{result['steady_segments']} segments, "
                   f"{result['steady_duration_s']:.1f} s")
    
    # ========================================================
    # Visualization
    # ========================================================
    
    logger.info("Generating visualizations...")
    
    plot_regime_classification(
        df,
        regimes,
        str(output_dir / 'temperature_regime_classification.png')
    )
    
    plot_derivative_diagnostic(
        df,
        dT_internal_dt_smooth,
        dT_top_dt_smooth,
        internal_steady_threshold=args.internal_steady_threshold,
        top_steady_threshold=args.top_steady_threshold,
        heating_threshold=args.heating_threshold,
        cooling_threshold=args.cooling_threshold,
        output_path=str(output_dir / 'temperature_derivative_diagnostic.png')
    )
    
    if len(steady_summary_df) > 0:
        plot_steady_spatial_drop(
            segments_df,
            str(output_dir / 'steady_spatial_temperature_drop.png')
        )
    
    # ========================================================
    # Metadata
    # ========================================================
    
    metadata = {
        'source_aligned_csv': str(Path(args.input_csv).resolve()),
        'output_directory': str(output_dir.resolve()),
        'smoothing_method': 'centered rolling mean',
        'smoothing_window_s': args.smooth_window,
        
        'internal_steady_threshold': args.internal_steady_threshold,
        'top_steady_threshold': args.top_steady_threshold,
        'heating_threshold': args.heating_threshold,
        'cooling_threshold': args.cooling_threshold,
        'minimum_duration_s': args.min_duration,
        
        'total_points': int(len(df)),
        
        'steady_points': int(stats['STEADY']['points']),
        'heating_points': int(stats['TRANSIENT_HEATING']['points']),
        'cooling_points': int(stats['TRANSIENT_COOLING']['points']),
        'settling_points': int(stats['SETTLING']['points']),
        'other_points': int(stats['TRANSITION_OTHER']['points']),
        
        'steady_segments': int(stats['STEADY']['segments']),
        'heating_segments': int(stats['TRANSIENT_HEATING']['segments']),
        'cooling_segments': int(stats['TRANSIENT_COOLING']['segments']),
        'settling_segments': int(stats['SETTLING']['segments']),
        'other_segments': int(stats['TRANSITION_OTHER']['segments']),
        
        'steady_total_duration': float(stats['STEADY']['interval_duration_s']),
        'heating_total_duration': float(stats['TRANSIENT_HEATING']['interval_duration_s']),
        'cooling_total_duration': float(stats['TRANSIENT_COOLING']['interval_duration_s']),
        'settling_total_duration': float(stats['SETTLING']['interval_duration_s']),
        'other_total_duration': float(stats['TRANSITION_OTHER']['interval_duration_s']),
        
        'threshold_sensitivity_results': sensitivity_results,
        
        'regime_labels_are_diagnostic_only': True,
        'data_removed_based_on_regime': False,
    }
    
    metadata_path = output_dir / 'regime_classification_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_path}")
    
    logger.info("=" * 60)
    logger.info("REGIME CLASSIFICATION COMPLETE")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
