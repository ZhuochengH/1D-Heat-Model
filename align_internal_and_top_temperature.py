#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal / top-surface temperature alignment (top as reference time axis)
=========================================================================

科学目的 (仅数据准备, 不做 FDM / 不拟合 k, cp):
------------------------------------------------
同一加热实验中两个独立温度测量:

  Dataset A: Peltier Zone 1 内置传感器温度  T_internal(t)
             (将来作为 FDM 输入)
  Dataset B: 实测顶面温度                  T_top_measured(t)
             (将来作为拟合 k_eff / cp_eff 的实验目标)

本任务只做: 读取 -> 校验 -> 插值对齐 -> 导出 -> 绘图。
不运行 FDM, 不优化任何热物性参数。

v3 修改 (以顶部测量为参考时间轴):
  - 不再使用绝对时间戳同步 (不要求内部绝对开始时间);
  - 不再按行索引对齐;
  - 不做互相关 / 峰值匹配 / 加热起点匹配 / 自动信号移位;
  - 顶部测量作为 MASTER 参考: 首个有效 T Avg 样本 = t=0,
    后续样本按 1 s 间隔 (t = 0,1,2,...);
  - 内部测量使用其 Time(s) 列作为实验时间坐标 (非均匀可接受);
  - 科学假设: 顶部 t=0 与内部 Time(s) 属于同一实验时间参考;
  - 内部温度经线性插值 (np.interp) 到顶部时间网格;
  - 无外推: 顶部时间超出内部 Time(s) 范围的点被排除并计数报告。

对齐输出将用于后续 k_eff / cp_eff 拟合:
  FDM 输入       : T_internal(t)
  实验拟合目标   : T_top_measured(t)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "temperature_alignment_output" / "72C"

DEFAULT_INTERNAL_XLSX = (
    PROJECT_ROOT.parent / "Calibration"
    / "08.06 COC_top 72°C_zone1_temperature_analysis.xlsx"
)
DEFAULT_TOP_XLSX = (
    PROJECT_ROOT.parent / "Calibration" / "extension 72°C.xls"
)

INTERNAL_COLUMN = "Zone 1 Avg (°C)"
TOP_COLUMN = "T Avg"
ASSUMED_DT = 1.0
INTERNAL_TIME_COL = "Time(s)"      # 内部时间坐标 (非均匀可接受)
TOP_TIME_COL = "RECTime"           # 仅诊断用 (确认 1 Hz), 不决定对齐原点
TOP_DIAG_TIME_COL = "RECTime"      # 顶部诊断时间列


# ============================================================
# 列名匹配 (空格折叠容错)
# ============================================================

def _find_column(df, column):
    """列名查找: 先精确, 再空格折叠容错。"""
    if column in df.columns:
        return column
    col_norm = re.sub(r"\s+", " ", str(column).strip())
    for c in df.columns:
        if re.sub(r"\s+", " ", str(c).strip()) == col_norm:
            return c
    return None


# ============================================================
# 数据加载
# ============================================================

def _validate_time_axis(t, source_desc):
    """校验时间轴: 有穷、严格递增。"""
    t = np.asarray(t, dtype=float)
    if len(t) < 2 or np.any(~np.isfinite(t)):
        raise ValueError(f"时间轴无效: {source_desc}")
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"时间轴必须严格递增: {source_desc}")
    diffs = np.diff(t)
    return {
        "median_dt": float(np.median(diffs)),
        "min_dt": float(np.min(diffs)),
        "max_dt": float(np.max(diffs)),
    }


def load_top_series_top_reference(
    xlsx_path,
    column=TOP_COLUMN,
    sheet="Data",
    diag_time_col=TOP_DIAG_TIME_COL,
):
    """
    加载顶部表面温度, 以顶部测量为 MASTER 参考时间轴。

    - 首个有效 T Avg 样本 = t=0;
    - 后续样本按 1 s 间隔: t_top = 0, 1, 2, ...
    - RECTime (若有) 仅用于诊断确认 1 Hz 采样, 不决定对齐原点。

    返回 dict:
        t_top:   参考时间轴 (0,1,2,...), 与温度一一对应
        T:       顶部温度序列
        t_diag:  RECTime 诊断时间 (相对秒) 或 None
        n_original / n_valid / resolved_column
        median_dt / min_dt / max_dt (RECTime 诊断)
    """
    path = Path(xlsx_path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    col = _find_column(df, column)
    if col is None:
        raise KeyError(
            f"找不到温度列 {column!r} (含空格折叠容错); 可用列: {list(df.columns)}"
        )
    T = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    n_original = len(T)
    valid = np.isfinite(T)
    T = T[valid]
    n_valid = int(T.size)

    # 顶部参考时间轴: 0,1,2,...
    t_top = np.arange(n_valid, dtype=float) * ASSUMED_DT

    # RECTime 诊断 (确认 1 Hz)
    t_diag = None
    diag_stats = {"median_dt": ASSUMED_DT, "min_dt": ASSUMED_DT,
                  "max_dt": ASSUMED_DT}
    if diag_time_col is not None and diag_time_col in df.columns:
        raw = df[diag_time_col]
        t_dt = pd.to_datetime(raw, errors="coerce")
        ok = t_dt.notna().to_numpy()
        if ok.any():
            base = t_dt[ok].iloc[0]
            t_diag_all = np.full(len(df), np.nan, dtype=float)
            t_diag_all[ok] = (
                (t_dt[ok] - base).dt.total_seconds().to_numpy(dtype=float)
            )
            t_diag_all = t_diag_all[valid]
            fin = np.isfinite(t_diag_all)
            if fin.sum() >= 2:
                t_diag = t_diag_all[fin]
                diffs = np.diff(t_diag)
                diffs = diffs[diffs > 0]
                if len(diffs):
                    diag_stats = {
                        "median_dt": float(np.median(diffs)),
                        "min_dt": float(np.min(diffs)),
                        "max_dt": float(np.max(diffs)),
                    }

    return {
        "t_top": t_top,
        "T": T,
        "t_diag": t_diag,
        "n_original": n_original,
        "n_valid": n_valid,
        "resolved_column": col,
        "time_source": (
            "reference grid: first valid T Avg sample = 0 s; "
            "subsequent samples at 1 s intervals"
        ),
        "median_dt": diag_stats["median_dt"],
        "min_dt": diag_stats["min_dt"],
        "max_dt": diag_stats["max_dt"],
    }


def load_internal_series_time_col(
    xlsx_path,
    column=INTERNAL_COLUMN,
    sheet="Extracted_Data",
    time_col=INTERNAL_TIME_COL,
):
    """
    加载内部传感器温度, 使用其 Time(s) 列作为实验时间坐标。

    - Time(s) 必须数值/有穷/严格递增;
    - 允许非均匀采样间隔;
    - 温度与时间一一配对。

    返回 dict:
        t_internal:  Time(s) 值 (秒)
        T:           内部温度序列
        n_original / n_valid / resolved_column
        time_source / first_time / last_time
        median_dt / min_dt / max_dt
    """
    path = Path(xlsx_path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    col = _find_column(df, column)
    if col is None:
        raise KeyError(
            f"找不到温度列 {column!r} (含空格折叠容错); 可用列: {list(df.columns)}"
        )
    T = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    n_original = len(T)
    valid = np.isfinite(T)
    n_valid = int(valid.sum())

    if time_col is None or time_col not in df.columns:
        raise ValueError(
            f"内部数据需要时间列 (期望 {time_col!r}); 当前列: {list(df.columns)}"
        )
    raw = df[time_col]
    is_dt = pd.api.types.is_datetime64_any_dtype(raw)
    t_num = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    if is_dt or np.isfinite(t_num).sum() < 0.5 * len(t_num):
        raise ValueError(
            f"内部时间列 {time_col!r} 应为数值秒 (Time(s)); "
            f"收到非数值内容: {raw.dtype}"
        )

    t_all = t_num[valid]
    T = T[valid]
    fin = np.isfinite(t_all)
    t_internal = t_all[fin]
    T = T[fin]

    dt_stats = _validate_time_axis(t_internal, f"internal {time_col!r}")

    return {
        "t_internal": t_internal,
        "T": T,
        "n_original": n_original,
        "n_valid": int(T.size),
        "resolved_column": col,
        "time_source": f"numeric column {time_col!r}",
        "first_time": float(t_internal[0]),
        "last_time": float(t_internal[-1]),
        "median_dt": dt_stats["median_dt"],
        "min_dt": dt_stats["min_dt"],
        "max_dt": dt_stats["max_dt"],
    }


# ============================================================
# 顶部参考对齐与差异
# ============================================================

def align_to_top_reference(internal, top, max_top_rows=300):
    """
    核心对齐: 以顶部测量为 MASTER 参考时间轴。

    - 顶部参考时间 t_top = 0,1,2,... (前 max_top_rows 个有效样本);
    - 内部温度用 np.interp 线性插值到顶部时间;
    - 无外推: 只保留内部 Time(s) 范围内的顶部时间点,
      超出部分被排除并计数 (excluded_early / excluded_late)。

    返回 dict:
        time_s:            对齐时间轴 (相对顶部 t=0)
        T_internal:        插值后的内部温度
        T_top:             顶部温度
        delta:             T_internal - T_top
        n_requested / n_aligned
        n_excluded_early / n_excluded_late
        diag_before / diag_after: 插值源点诊断数组
    """
    t_top_all = top["t_top"]
    T_top_all = top["T"]

    n_requested = min(int(max_top_rows), int(len(t_top_all)))
    t_top = t_top_all[:n_requested]
    T_top_sel = T_top_all[:n_requested]

    t_int = internal["t_internal"]
    T_int = internal["T"]
    t_int_min = float(t_int[0])
    t_int_max = float(t_int[-1])

    # 只保留内部时间范围内的顶部时间点 (无外推)
    inside = (t_top >= t_int_min - 1e-9) & (t_top <= t_int_max + 1e-9)
    n_excluded_early = int(np.sum(t_top < t_int_min - 1e-9))
    n_excluded_late = int(np.sum(t_top > t_int_max + 1e-9))
    t_top_use = t_top[inside]
    T_top_use = T_top_sel[inside]

    if len(t_top_use) == 0:
        raise ValueError(
            "没有顶部时间点落在内部 Time(s) 范围内 "
            f"[{t_int_min:.3f}, {t_int_max:.3f}] s; 无法对齐。"
        )

    # 线性插值 (np.interp): 对每个目标时间找到相邻内部样本
    T_int_interp = np.interp(t_top_use, t_int, T_int)

    # 插值诊断: 每个对齐点使用的内部源点区间
    diag_before = np.full(len(t_top_use), np.nan, dtype=float)
    diag_after = np.full(len(t_top_use), np.nan, dtype=float)
    t_before = np.full(len(t_top_use), np.nan, dtype=float)
    t_after = np.full(len(t_top_use), np.nan, dtype=float)
    T_before = np.full(len(t_top_use), np.nan, dtype=float)
    T_after = np.full(len(t_top_use), np.nan, dtype=float)
    for i, tg in enumerate(t_top_use):
        j = int(np.searchsorted(t_int, tg, side="right")) - 1
        j = max(0, min(j, len(t_int) - 2))
        t_before[i] = t_int[j]
        t_after[i] = t_int[j + 1]
        T_before[i] = T_int[j]
        T_after[i] = T_int[j + 1]
        diag_before[i] = t_int[j]
        diag_after[i] = t_int[j + 1]

    delta = T_int_interp - T_top_use

    return {
        "time_s": t_top_use,
        "T_internal": T_int_interp,
        "T_top": T_top_use,
        "delta": delta,
        "n_requested": n_requested,
        "n_aligned": int(len(t_top_use)),
        "n_excluded_early": n_excluded_early,
        "n_excluded_late": n_excluded_late,
        "internal_t_before": t_before,
        "internal_t_after": t_after,
        "internal_T_before": T_before,
        "internal_T_after": T_after,
        "interpolation_method": "linear (np.interp)",
    }


def describe_series(T):
    """基本描述统计: min/max/initial/final。"""
    return {
        "min": float(np.min(T)),
        "max": float(np.max(T)),
        "initial": float(T[0]),
        "final": float(T[-1]),
    }


def compute_delta_metrics(delta):
    """点对点 Delta_T 描述统计 (仅描述, 不拟合)。"""
    d = np.asarray(delta, dtype=float)
    return {
        "mean": float(np.mean(d)),
        "mean_abs": float(np.mean(np.abs(d))),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "max_abs": float(np.max(np.abs(d))),
    }


# ============================================================
# 绘图
# ============================================================

def plot_internal_vs_top(aligned, output_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(aligned["time_s"], aligned["T_internal"], color="#d62728",
            linewidth=1.8, label="Peltier Internal Sensor Temperature (Interpolated)")
    ax.plot(aligned["time_s"], aligned["T_top"], color="#1f77b4",
            linewidth=1.8, label="Measured Top-Surface Temperature")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"Temperature ($^\circ$C)", fontsize=12)
    ax.set_title("Internal vs Top-Surface Measured Temperature\n"
                 "(top-reference aligned — no FDM, no calibration)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_delta(aligned, output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(aligned["time_s"], aligned["delta"], color="#2ca02c",
            linewidth=1.5)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"Internal - Top Temperature Difference ($^\circ$C)",
                  fontsize=12)
    ax.set_title("Measured Internal minus Top-Surface Temperature Difference\n"
                 "(top-reference aligned, descriptive only — no fit)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_interpolation_check(internal, aligned, output_path):
    """原始内部测量点 + 插值轨迹 (验证插值跟随原始数据、无过冲/漂移)。"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(internal["t_internal"], internal["T"], color="#999999",
            marker="o", markersize=3, linewidth=0, alpha=0.6,
            label="Raw internal measurements (Time(s))")
    ax.plot(aligned["time_s"], aligned["T_internal"], color="#d62728",
            linewidth=1.5, label="Linear interpolated T_internal (top 1 s grid)")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"Temperature ($^\circ$C)", fontsize=12)
    ax.set_title("Internal-Temperature Interpolation Check\n"
                 "(linear np.interp — follows raw data, no overshoot)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# CLI 与主程序
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Align internal & top-surface temperature; "
            "top measurement is the reference time axis"
        )
    )
    p.add_argument("--internal-xlsx", default=str(DEFAULT_INTERNAL_XLSX),
                   help="内部传感器温度 Excel 路径")
    p.add_argument("--internal-sheet", default="Extracted_Data",
                   help="内部传感器 sheet 名或索引")
    p.add_argument("--internal-col", default=INTERNAL_COLUMN,
                   help="内部传感器温度列 (空格折叠容错)")
    p.add_argument("--internal-time-col", default=INTERNAL_TIME_COL,
                   help="内部传感器时间列 (默认 Time(s))")
    p.add_argument("--top-xlsx", default=str(DEFAULT_TOP_XLSX),
                   help="顶部表面温度 Excel 路径")
    p.add_argument("--top-sheet", default="Data",
                   help="顶部表面 sheet 名或索引")
    p.add_argument("--top-col", default=TOP_COLUMN,
                   help="顶部表面温度列 (空格折叠容错)")
    p.add_argument("--top-diag-time-col", default=TOP_DIAG_TIME_COL,
                   help="顶部诊断时间列 (RECTime, 仅确认 1 Hz)")
    p.add_argument("--max-top-rows", type=int, default=300,
                   help="使用前 N 个有效顶部样本 (默认 300)")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                   help="输出目录")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ---- 顶部表面 (MASTER 参考时间轴) ----
    top = load_top_series_top_reference(
        args.top_xlsx, column=args.top_col, sheet=args.top_sheet,
        diag_time_col=args.top_diag_time_col,
    )
    if top["n_valid"] < args.max_top_rows:
        print(f"WARN: 顶部有效行 {top['n_valid']} < "
              f"--max-top-rows {args.max_top_rows}, 使用全部有效行。")

    # ---- 内部传感器 (Time(s) 时间坐标) ----
    internal = load_internal_series_time_col(
        args.internal_xlsx, column=args.internal_col, sheet=args.internal_sheet,
        time_col=args.internal_time_col,
    )

    # ---- 顶部参考对齐 + 线性插值 (无外推) ----
    aligned = align_to_top_reference(
        internal, top, max_top_rows=args.max_top_rows
    )
    if aligned["n_aligned"] == 0:
        print("\n[STOP] 没有可对齐的时间点 (内部 Time(s) 范围与顶部网格无交集)。")
        return 2

    int_stats = describe_series(aligned["T_internal"])
    top_stats = describe_series(aligned["T_top"])
    delta_metrics = compute_delta_metrics(aligned["delta"])

    # ---- 输出目录 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 对齐 CSV ----
    out_df = pd.DataFrame({
        "time_s": aligned["time_s"],
        "T_internal_interpolated_C": aligned["T_internal"],
        "T_top_measured_C": aligned["T_top"],
        "Delta_T_internal_minus_top_C": aligned["delta"],
        "internal_source_time_before_s": aligned["internal_t_before"],
        "internal_source_time_after_s": aligned["internal_t_after"],
    })
    csv_path = output_dir / "aligned_internal_top_temperature.csv"
    out_df.to_csv(csv_path, index=False)

    # ---- 插值诊断 CSV ----
    diag_df = pd.DataFrame({
        "target_top_time_s": aligned["time_s"],
        "internal_t_before_s": aligned["internal_t_before"],
        "internal_T_before_C": aligned["internal_T_before"],
        "internal_t_after_s": aligned["internal_t_after"],
        "internal_T_after_C": aligned["internal_T_after"],
        "interpolated_internal_T_C": aligned["T_internal"],
        "top_measured_T_C": aligned["T_top"],
    })
    diag_csv = output_dir / "interpolation_diagnostic.csv"
    diag_df.to_csv(diag_csv, index=False)

    # ---- 图 ----
    fig1 = output_dir / "internal_vs_top_temperature.png"
    fig2 = output_dir / "internal_minus_top_temperature.png"
    fig3 = output_dir / "internal_interpolation_check.png"
    plot_internal_vs_top(aligned, fig1)
    plot_delta(aligned, fig2)
    plot_interpolation_check(internal, aligned, fig3)

    # ---- 元数据 ----
    metadata = {
        "alignment_strategy": (
            "Top measurement defines t=0 and 1-s reference grid; "
            "internal temperature is linearly interpolated from its Time(s) "
            "values onto top measurement times."
        ),
        "timing_assumption": (
            "Top t=0 and internal Time(s) are treated as belonging to the "
            "same experimental time reference."
        ),
        "top_file": str(Path(args.top_xlsx).resolve()),
        "top_sheet": str(args.top_sheet),
        "top_temperature_column": top["resolved_column"],
        "top_time_definition": (
            "first valid T Avg sample = 0 s; subsequent samples at 1 s intervals"
        ),
        "top_valid_rows": top["n_valid"],
        "top_diag_RECTime_median_dt": top["median_dt"],
        "top_diag_RECTime_min_dt": top["min_dt"],
        "top_diag_RECTime_max_dt": top["max_dt"],
        "internal_file": str(Path(args.internal_xlsx).resolve()),
        "internal_sheet": str(args.internal_sheet),
        "internal_temperature_column": internal["resolved_column"],
        "internal_time_column": args.internal_time_col,
        "internal_first_time": internal["first_time"],
        "internal_last_time": internal["last_time"],
        "internal_median_dt": internal["median_dt"],
        "internal_min_dt": internal["min_dt"],
        "internal_max_dt": internal["max_dt"],
        "internal_valid_rows": internal["n_valid"],
        "requested_top_points": int(aligned["n_requested"]),
        "aligned_points": int(aligned["n_aligned"]),
        "excluded_early_top_points": int(aligned["n_excluded_early"]),
        "excluded_late_top_points": int(aligned["n_excluded_late"]),
        "interpolation_method": aligned["interpolation_method"],
        "extrapolation_used": False,
        "signal_based_time_shift_used": False,
        "row_index_alignment_used": False,
        "note": (
            "Data alignment only. No FDM run, no k/cp fitting, "
            "no steady/transient classification."
        ),
    }
    meta_path = output_dir / "alignment_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ---- 控制台诊断 ----
    print("\nTOP DATA (MASTER reference):")
    print(f"  file: {Path(args.top_xlsx).resolve()}")
    print(f"  sheet: {args.top_sheet}")
    print(f"  temperature column: {top['resolved_column']!r}")
    print(f"  valid rows: {top['n_valid']}")
    print(f"  points requested: {aligned['n_requested']}")
    print(f"  reference time range: 0 → {aligned['n_requested'] - 1:.0f} s")
    print(f"  (RECTime diagnostic median dt: {top['median_dt']:.4f} s, "
          f"min {top['min_dt']:.4f}, max {top['max_dt']:.4f})")
    print(f"  first temperature: {top_stats['initial']:.3f} °C")
    print(f"  min temperature: {top_stats['min']:.3f} °C")
    print(f"  max temperature: {top_stats['max']:.3f} °C")
    print(f"  final temperature: {top_stats['final']:.3f} °C")

    print("\nINTERNAL DATA:")
    print(f"  file: {Path(args.internal_xlsx).resolve()}")
    print(f"  sheet: {args.internal_sheet}")
    print(f"  temperature column: {internal['resolved_column']!r}")
    print(f"  time column: {args.internal_time_col}")
    print(f"  valid rows: {internal['n_valid']}")
    print(f"  time range: {internal['first_time']:.3f} → "
          f"{internal['last_time']:.3f} s")
    print(f"  median dt: {internal['median_dt']:.4f} s "
          f"(min {internal['min_dt']:.4f}, max {internal['max_dt']:.4f})")
    print(f"  first temperature: {int_stats['initial']:.3f} °C")
    print(f"  min temperature: {int_stats['min']:.3f} °C")
    print(f"  max temperature: {int_stats['max']:.3f} °C")
    print(f"  final temperature: {int_stats['final']:.3f} °C")

    print("\nALIGNMENT:")
    print(f"  strategy: top reference time axis; internal linearly interpolated")
    print(f"  requested top points: {aligned['n_requested']}")
    print(f"  successfully aligned: {aligned['n_aligned']}")
    print(f"  excluded before internal range: {aligned['n_excluded_early']}")
    print(f"  excluded after internal range: {aligned['n_excluded_late']}")
    print(f"  aligned time range: {aligned['time_s'][0]:.1f} → "
          f"{aligned['time_s'][-1]:.1f} s")
    print(f"  interpolation: {aligned['interpolation_method']}, "
          f"no extrapolation")

    print("\nMEASURED DIFFERENCE (interpolated internal - top):")
    print(f"  mean: {delta_metrics['mean']:.3f} °C")
    print(f"  mean absolute: {delta_metrics['mean_abs']:.3f} °C")
    print(f"  min: {delta_metrics['min']:.3f} °C")
    print(f"  max: {delta_metrics['max']:.3f} °C")
    print(f"  max absolute: {delta_metrics['max_abs']:.3f} °C")

    print("\nOUTPUT:")
    print(f"  aligned CSV: {csv_path.resolve()}")
    print(f"  interpolation diagnostic CSV: {diag_csv.resolve()}")
    print(f"  comparison figure: {fig1.resolve()}")
    print(f"  difference figure: {fig2.resolve()}")
    print(f"  interpolation check figure: {fig3.resolve()}")
    print(f"  metadata: {meta_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
