#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅含内部温度的样品层温度重建 (FROZEN calibrated model)
========================================================

科学目的
--------
用当前冻结的裸顶标定模型 (NOMINAL_BARE_TOP_CALIBRATION_V1:
k_eff = 0.0165 W/(m K), cp_eff = 900 J/(kg K), rho = 1020 kg/m3) 从
Rhonda 内置传感器内部温度日志重建样品层 (180 -> 200 um) 温度剖面。

NOT calibration / NOT validation / NOT fitting / NOT scanning:
    - 无实测 Top COC 温度参与;
    - 无任何参数优化 / 网格搜索 / 时间平移 / 平滑;
    - 内部温度仅作为底部 Dirichlet 边界输入。

流程:
    实测内部温度 (Time(s) / Zone 1 Avg (°C))
        -> 冻结 1D FDM 模型 (BARE_TOP_COC_LAYERS, 850 um)
        -> T_sample_predicted (控制体积加权空间平均)
        -> T_top_predicted (次级输出, 供未来与实测比较)

复用 (不重复实现):
    - heat_model.run_simulation                      : 唯一权威 FDM;
    - calibrated_model_config                        : 冻结名义标定;
    - scan_effective_thermal_parameters.
        sample_prediction_at_measurement_times       : 修正时间采样
          (查询轴 = 实测时间坐标, 绝不用温度值作查询轴)。

时间约定
--------
    source_time_s  = 原始 Time(s) 值 (保留原样)
    elapsed_time_s = Time(s) - Time(s)[0]  (模型输出/绘图的标准时间轴)
求解器内部已把协议起点映射到 t=0 (验证过的既有行为), 因此
    measurement time (查询轴) = elapsed_time_s。

初始条件
--------
    T_initial = 第一个有效内部温度 (从本文件动态解析, 不硬编码)。
文档化为模型假设 (本数据集无同时顶部测量, 无法实验校验)。

用法
----
    uv run python predict_sample_from_internal_temperature.py \
        --input "<xlsx path>" \
        --experiment-name "08.12_pm_DOE11_faster" \
        --output-dir "calibrated_model_output/08.12_pm_DOE11_faster_sample_prediction_v1"
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from thermal_model.core import heat_model
from thermal_model.config.calibrated_model_config import (
    NOMINAL_BARE_TOP_CALIBRATION_V1,
    make_nominal_calibrated_materials,
    nominal_layer_stack,
)
from thermal_model.utilities.scan_effective_thermal_parameters import (
    sample_prediction_at_measurement_times,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT.parent / "Calibration"
    / "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx"
)
DEFAULT_EXPERIMENT_NAME = "08.12_pm_DOE11_faster"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "calibrated_model_output"
    / "08.12_pm_DOE11_faster_sample_prediction_v1"
)

H_CONV = 5.0
T_AMB = 25.0
SAVE_DT = 0.1

TIME_COL = "Time(s)"
TEMP_COL = "Zone 1 Avg (°C)"
SHEET = "Extracted_Data"


# ============================================================
# Git 元数据
# ============================================================

def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_describe():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ============================================================
# 列名查找 (空格折叠容错, 与既有脚本一致)
# ============================================================

def _find_column(df, column):
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

def load_internal_data(path, sheet=SHEET, time_col=TIME_COL,
                       temp_col=TEMP_COL):
    """加载内部温度日志。

    返回 dict:
        source_time_s / T_internal_C / elapsed_time_s / n_valid
        resolved_time_col / resolved_temp_col / dt 统计
        first_time / last_time
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在: {p}")
    df = pd.read_excel(p, sheet_name=sheet)

    tc = _find_column(df, time_col)
    if tc is None:
        raise KeyError(f"找不到时间列 {time_col!r}; 可用列: {list(df.columns)}")
    gc = _find_column(df, temp_col)
    if gc is None:
        raise KeyError(f"找不到温度列 {temp_col!r}; 可用列: {list(df.columns)}")

    t = pd.to_numeric(df[tc], errors="coerce").to_numpy(dtype=float)
    T = pd.to_numeric(df[gc], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(T)
    t = t[ok]
    T = T[ok]
    if len(t) < 2:
        raise ValueError("有效内部数据点不足。")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time(s) 必须严格递增。")

    dt = np.diff(t)
    elapsed = t - t[0]
    return {
        "source_time_s": t,
        "T_internal_C": T,
        "elapsed_time_s": elapsed,
        "n_valid": int(len(t)),
        "resolved_time_col": tc,
        "resolved_temp_col": gc,
        "median_dt": float(np.median(dt)),
        "min_dt": float(np.min(dt)),
        "max_dt": float(np.max(dt)),
        "first_time": float(t[0]),
        "last_time": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "strictly_increasing": True,
        "T_min_C": float(np.min(T)),
        "T_max_C": float(np.max(T)),
        "T_initial_C": float(T[0]),
        "T_final_C": float(T[-1]),
    }


# ============================================================
# 冻结配置 / 初始条件
# ============================================================

def build_frozen_config():
    """返回冻结名义标定 (只读)。"""
    cal = NOMINAL_BARE_TOP_CALIBRATION_V1
    layers = nominal_layer_stack(cal)
    mats = make_nominal_calibrated_materials(cal)
    return cal, layers, mats, H_CONV, T_AMB


def compute_initial_condition(t_internal):
    """初始条件 = 第一个有效内部温度 (动态解析, 不硬编码)。"""
    return float(np.asarray(t_internal, dtype=float)[0])


# ============================================================
# 模型执行 (单一固定参数 FDM)
# ============================================================

def run_frozen_simulation(elapsed_time_s, t_internal, cal, layers, mats):
    """一次固定参数 FDM。返回 (t_arr, T_sample_arr, T_top_surface_arr)。

    注意: 求解器把协议起点映射到 t=0 (既有行为), 因此传入 elapsed 时间轴。
    """
    T_initial = compute_initial_condition(t_internal)
    result = heat_model.run_simulation(
        time_s=elapsed_time_s,
        bottom_temperature_C=t_internal,
        materials=mats,
        layers=layers,
        h_conv=H_CONV,
        T_air_ambient=T_AMB,
        save_dt=SAVE_DT,
        T_initial_C=T_initial,
    )
    return (result["t_array"], result["T_sample_arr"],
            result["T_top_surface_arr"])


def predict_sample_temperature(source_time_s, elapsed_time_s, t_internal,
                               output_dir, experiment_name,
                               input_path=None):
    """完整预测流程 (冻结模型, 无拟合)。

    返回 (summary_dict, trace_df)。
    """
    cal, layers, mats, h_conv, t_amb = build_frozen_config()

    t_arr, T_sample, T_top = run_frozen_simulation(
        elapsed_time_s, t_internal, cal, layers, mats
    )

    # 修正时间采样: 查询轴 = elapsed 实测时间坐标 (绝不用温度值)
    T_sample_pred = sample_prediction_at_measurement_times(
        elapsed_time_s, t_arr, T_sample
    )
    T_top_pred = sample_prediction_at_measurement_times(
        elapsed_time_s, t_arr, T_top
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 主迹线 CSV ----
    trace = pd.DataFrame({
        "source_time_s": source_time_s,
        "elapsed_time_s": elapsed_time_s,
        "T_internal_C": t_internal,
        "T_sample_predicted_C": T_sample_pred,
        "T_top_predicted_C": T_top_pred,
    })
    trace.to_csv(output_dir / "sample_temperature_prediction.csv", index=False)

    # ---- 汇总计算 ----
    summary = compute_summary(elapsed_time_s, t_internal, T_sample_pred,
                              T_top_pred)

    # ---- 周期检测 -------
    cycles = detect_cycles(elapsed_time_s, t_internal, T_sample_pred)
    summary["cycles"] = cycles
    if cycles:
        # 每周期高温停留时间 + cycle_summary.csv
        cyc_rows = per_cycle_dwell(cycles, elapsed_time_s, T_sample_pred)
        summary["per_cycle_dwell_s"] = cyc_rows
        pd.DataFrame(cyc_rows).to_csv(
            output_dir / "cycle_summary.csv", index=False
        )

    # ---- 图形 ----
    _plot_main(elapsed_time_s, t_internal, T_sample_pred, T_top_pred,
               experiment_name, output_dir)
    _plot_internal_sample_diff(elapsed_time_s, t_internal, T_sample_pred,
                               output_dir)

    # ---- 汇总 txt + 元数据 ----
    _write_summary_txt(input_path, cal, summary, output_dir, experiment_name)
    _write_metadata(input_path, cal, summary, output_dir, experiment_name,
                    source_time_s, t_internal, t_arr)

    return summary, trace


# ============================================================
# 汇总计算
# ============================================================

def compute_summary(elapsed, t_int, t_sample, t_top):
    e = np.asarray(elapsed, dtype=float)
    ti = np.asarray(t_int, dtype=float)
    ts = np.asarray(t_sample, dtype=float)
    tt = np.asarray(t_top, dtype=float)

    d = ts - ti
    return {
        "time_range_s": [float(e[0]), float(e[-1])],
        "internal": {
            "min_C": float(np.min(ti)),
            "max_C": float(np.max(ti)),
            "initial_C": float(ti[0]),
        },
        "sample_predicted": {
            "min_C": float(np.min(ts)),
            "max_C": float(np.max(ts)),
            "time_of_min_s": float(e[int(np.argmin(ts))]),
            "time_of_max_s": float(e[int(np.argmax(ts))]),
        },
        "top_predicted": {
            "min_C": float(np.min(tt)),
            "max_C": float(np.max(tt)),
        },
        "sample_minus_internal": {
            "max_C": float(np.max(d)),
            "min_C": float(np.min(d)),
            "mean_abs_C": float(np.mean(np.abs(d))),
            "time_of_max_s": float(e[int(np.argmax(d))]),
            "time_of_min_s": float(e[int(np.argmin(d))]),
        },
        "ramp_rates": {
            "internal": ramp_summary(e, ti),
            "sample_predicted": ramp_summary(e, ts),
        },
        "dwell_times_s": dwell_times(e, ts),
    }


def ramp_summary(t, T):
    """基于 numpy.gradient(实测时间坐标) 的斜坡率描述。

    区分点级极值 (max/min) 与稳健估计 (p95 正/负幅值)。
    """
    dT = np.gradient(T, t)
    pos = dT[dT > 0]
    neg = dT[dT < 0]
    return {
        "max_positive_C_per_s": float(np.max(dT)),
        "max_negative_C_per_s": float(np.min(dT)),
        "p95_positive_C_per_s": float(np.percentile(pos, 95)) if pos.size else 0.0,
        "p95_negative_magnitude_C_per_s": float(-np.percentile(-neg, 95))
        if neg.size else 0.0,
    }


def dwell_times(t, T, thresholds=(90.0, 92.0, 94.0, 95.0),
                ranges=((55, 60), (60, 65), (65, 70), (70, 75))):
    """基于实际时间戳的区间积分停留时间 (非均匀 dt 安全)。

    对每个采样点 i, 贡献区间 [t_i, t_{i+1}) 内满足条件的时间长度
    (用区间端点温度线性判定)。
    """
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    n = len(t)
    out = {}

    def _dwell_bool(cond):
        total = 0.0
        for i in range(n - 1):
            a, b = t[i], t[i + 1]
            ca = cond(T[i])
            cb = cond(T[i + 1])
            if ca and cb:
                total += (b - a)
            elif ca != cb:
                # 线性跨越: 解 T(t) = threshold 交点
                span = b - a
                total += span * ca  # 粗略: 取起点侧半区间
        return total

    # 阈值 (>=)
    for th in thresholds:
        if np.max(T) < th - 5.0:
            continue  # 远低于阈值, 跳过
        total = 0.0
        for i in range(n - 1):
            a, b = t[i], t[i + 1]
            Ta, Tb = T[i], T[i + 1]
            if Ta >= th and Tb >= th:
                total += (b - a)
            elif Ta >= th or Tb >= th:
                # 线性插值交点
                if Tb != Ta:
                    frac = (th - Ta) / (Tb - Ta)
                    if Ta >= th:
                        total += (b - a) * (1.0 - frac)
                    else:
                        total += (b - a) * frac
        out[f"sample_ge_{th:.0f}C_s"] = float(total)

    for lo, hi in ranges:
        total = 0.0
        for i in range(n - 1):
            a, b = t[i], t[i + 1]
            Ta, Tb = T[i], T[i + 1]
            if lo <= Ta < hi and lo <= Tb < hi:
                total += (b - a)
            elif (lo <= Ta < hi) or (lo <= Tb < hi):
                total += (b - a) * 0.5
        out[f"sample_{lo:.0f}_{hi:.0f}C_s"] = float(total)

    return out


# ============================================================
# 周期检测 (仅基于内部温度, 稳健)
# ============================================================

def detect_cycles(t, t_internal, t_sample, peak_threshold=88.0,
                  dip_threshold=60.0, min_peak_separation_s=15.0):
    """从内部温度检测稳健的 PCR 周期 (峰-谷结构)。

    规则:
        1) 局部最大峰 (T_internal >= peak_threshold, 平台感知);
        2) 若两个相邻峰之间没有真正的低谷 (min < dip_threshold),
           则它们属于同一高温相 (如初始 90C 平台上的波动) -> 合并,
           只保留更高的峰;
        3) 每个 (合并后) 峰的周期起点 = 其前方最近的低谷 (< dip_threshold)
           所在时间 (无低谷时 = t=0, 即初始冷启动);
        4) 相邻峰最小时间间隔 min_peak_separation_s。

    返回 list[dict] 或空列表 (无法稳健检测时)。
    """
    t = np.asarray(t, dtype=float)
    ti = np.asarray(t_internal, dtype=float)
    ts = np.asarray(t_sample, dtype=float)
    n = len(t)

    # ---- 峰 (局部最大, 平台感知) ----
    peaks = []
    i = 1
    while i < n - 1:
        if ti[i] >= peak_threshold and ti[i] >= ti[i - 1]:
            j = i
            while j + 1 < n and ti[j + 1] == ti[j]:
                j += 1
            if j == n - 1 or ti[j] >= ti[j + 1]:
                peaks.append(j)
            i = j + 1
        else:
            i += 1

    # ---- 低谷 (局部最小 < dip_threshold, 平台感知) ----
    troughs = []
    i = 1
    while i < n - 1:
        if ti[i] < dip_threshold and ti[i] <= ti[i - 1]:
            j = i
            while j + 1 < n and ti[j + 1] == ti[j]:
                j += 1
            if j == n - 1 or ti[j] <= ti[j + 1]:
                troughs.append(j)
            i = j + 1
        else:
            i += 1

    # ---- 合并无低谷分隔的相邻峰 (同一高温相) ----
    merged = []
    for idx in peaks:
        if merged:
            prev = merged[-1]
            between = ti[prev + 1:idx]
            min_between = float(np.min(between)) if between.size else \
                min(float(ti[prev]), float(ti[idx]))
            if min_between < dip_threshold:
                merged.append(idx)
            else:
                # 同一高温相: 保留更高的峰
                if ti[idx] > ti[prev]:
                    merged[-1] = idx
        else:
            merged.append(idx)

    # ---- 最小间隔去重 ----
    selected = []
    for idx in merged:
        if selected and (t[idx] - t[selected[-1]]) < min_peak_separation_s:
            continue
        selected.append(idx)

    if len(selected) < 2:
        return []

    # ---- 每个周期: 起点 = 前方最近低谷 ----
    cycles = []
    for k, pidx in enumerate(selected):
        # 该峰之前的低谷 (时间上早于峰; 取最近一个)
        prior = [tr for tr in troughs if tr < pidx]
        trough = prior[-1] if prior else 0
        # 样品峰: 在内部峰后的窗口内 (样品滞后), 取样品温度最大值
        hi = min(n - 1, pidx + 20)
        sw = ts[pidx:hi + 1]
        spi = pidx + int(np.argmax(sw))
        c = {
            "cycle_number": k + 1,
            "cycle_start_time_s": float(t[trough]),
            "internal_peak_time_s": float(t[pidx]),
            "internal_high_peak_C": float(ti[pidx]),
            "sample_high_peak_C": float(ts[spi]),
            "sample_peak_time_s": float(t[spi]),
            "internal_low_trough_C": float(ti[trough]),
            "sample_low_trough_C": float(ts[trough]),
            "sample_trough_time_s": float(t[trough]),
        }
        if k > 0:
            c["cycle_duration_s"] = float(t[trough] - t[selected[k - 1]])
        else:
            c["cycle_duration_s"] = None
        cycles.append(c)

    return cycles


def per_cycle_dwell(cycles, t, t_sample, thresholds=(72.0, 75.0, 80.0)):
    """每个周期内 T_sample >= th 的停留时间 (基于时间戳区间积分)。

    阈值按观测样品温度范围自适应给定 (本数据样品峰 ~78 C, 故用 72/75/80,
    而非 90+ 的不可达阈值)。
    """
    out = []
    for c in cycles:
        start = c["cycle_start_time_s"]
        end = c["internal_peak_time_s"] + 8.0
        mask = (t >= start) & (t <= end)
        if not mask.any():
            continue
        tc = t[mask]
        sc = t_sample[mask]
        dw = dwell_times(tc, sc, thresholds=thresholds, ranges=())
        row = {"cycle_number": c["cycle_number"]}
        row.update({k: v for k, v in dw.items()
                    if k.startswith("sample_ge_")})
        out.append(row)
    return out


# ============================================================
# 图形
# ============================================================

def _plot_main(e, t_int, t_sample, t_top, experiment_name, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(e, t_int, color="#7f7f7f", lw=1.4, ls=":",
            label="Internal sensor input (measured)")
    ax.plot(e, t_sample, color="#2ca02c", lw=2.2,
            label="Sample predicted (model estimate)")
    ax.plot(e, t_top, color="#1f77b4", lw=1.6, ls="--",
            label="Top COC predicted (model estimate)")
    ax.set_xlabel("Time [s] (elapsed)")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"{experiment_name} — Predicted Sample-Layer Thermal Profile\n"
                 "(frozen 72C-calibrated model; sample/top are model estimates)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "sample_temperature_prediction.png", dpi=200)
    fig.savefig(output_dir / "sample_temperature_prediction.pdf")
    plt.close(fig)


def _plot_internal_sample_diff(e, t_int, t_sample, output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(e, t_sample - t_int, color="#9467bd", lw=1.2)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Time [s] (elapsed)")
    ax.set_ylabel(r"$\Delta T$ sample - internal [°C]")
    ax.set_title("Model-Estimated Internal-to-Sample Temperature Difference\n"
                 "(thermal lag descriptive; NOT model error)")
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "internal_vs_sample_difference.png", dpi=200)
    plt.close(fig)


# ============================================================
# 输出文件
# ============================================================

def _write_summary_txt(input_path, cal, summary, output_dir, experiment_name):
    s = summary
    lines = [
        f"{experiment_name.upper()} SAMPLE-LAYER TEMPERATURE PREDICTION SUMMARY",
        "=" * 72,
        f"input file: {input_path}",
        f"nominal model: {cal.name}",
        f"  k_eff = {cal.k_eff_W_mK} W/(m K), cp_eff = {cal.cp_eff_J_kgK} "
        f"J/(kg K), rho = {cal.rho_COC_kg_m3} kg/m3",
        f"initial condition (T_initial): {s['internal']['initial_C']:.3f} C "
        f"(= first measured internal temperature; model assumption)",
        f"time range: {s['time_range_s'][0]:.3f} -> {s['time_range_s'][1]:.3f} s",
        f"internal range: {s['internal']['min_C']:.3f} -> "
        f"{s['internal']['max_C']:.3f} C",
        f"predicted sample range: {s['sample_predicted']['min_C']:.3f} "
        f"({s['sample_predicted']['time_of_min_s']:.1f} s) -> "
        f"{s['sample_predicted']['max_C']:.3f} "
        f"({s['sample_predicted']['time_of_max_s']:.1f} s) C",
        f"predicted top range: {s['top_predicted']['min_C']:.3f} -> "
        f"{s['top_predicted']['max_C']:.3f} C",
        "",
        "INTERNAL VS SAMPLE:",
        f"  max (sample - internal): {s['sample_minus_internal']['max_C']:.3f} C",
        f"  min (sample - internal): {s['sample_minus_internal']['min_C']:.3f} C",
        f"  mean |sample - internal|: "
        f"{s['sample_minus_internal']['mean_abs_C']:.3f} C",
        "",
        "RAMP RATES (numpy.gradient on actual time):",
        f"  internal max positive: "
        f"{s['ramp_rates']['internal']['max_positive_C_per_s']:.4f} C/s",
        f"  internal max negative: "
        f"{s['ramp_rates']['internal']['max_negative_C_per_s']:.4f} C/s",
        f"  internal robust heating (p95 pos): "
        f"{s['ramp_rates']['internal']['p95_positive_C_per_s']:.4f} C/s",
        f"  internal robust cooling (p95 neg mag): "
        f"{s['ramp_rates']['internal']['p95_negative_magnitude_C_per_s']:.4f} C/s",
        f"  sample max positive: "
        f"{s['ramp_rates']['sample_predicted']['max_positive_C_per_s']:.4f} C/s",
        f"  sample max negative: "
        f"{s['ramp_rates']['sample_predicted']['max_negative_C_per_s']:.4f} C/s",
        f"  sample robust heating (p95 pos): "
        f"{s['ramp_rates']['sample_predicted']['p95_positive_C_per_s']:.4f} C/s",
        f"  sample robust cooling (p95 neg mag): "
        f"{s['ramp_rates']['sample_predicted']['p95_negative_magnitude_C_per_s']:.4f} C/s",
        "",
        "SAMPLE DWELL TIMES (interval-integrated over actual timestamps):",
    ]
    for k, v in s["dwell_times_s"].items():
        lines.append(f"  {k}: {v:.2f} s")
    lines.append("")
    if s.get("cycles"):
        lines.append(f"CYCLE ANALYSIS: robust detection YES — "
                     f"{len(s['cycles'])} cycles")
        for c in s["cycles"]:
            lines.append(
                f"  cycle {c['cycle_number']}: start t={c['cycle_start_time_s']:.1f} s "
                f"| internal peak {c['internal_high_peak_C']:.2f} C "
                f"(t={c['internal_peak_time_s']:.1f}) | sample peak "
                f"{c['sample_high_peak_C']:.2f} C (t={c['sample_peak_time_s']:.1f}) "
                f"| trough internal "
                f"{c['internal_low_trough_C']:.2f} / sample "
                f"{c['sample_low_trough_C']:.2f} C | duration "
                f"{c['cycle_duration_s'] if c['cycle_duration_s'] else 'n/a'} s"
            )
    else:
        lines.append("CYCLE ANALYSIS: robust automatic detection NO — "
                     "continuous trace only.")
    lines.append("")
    lines.append("NOTE: sample temperature is estimated by the calibrated "
                 "phenomenological model and is NOT directly measured.")
    (output_dir / "sample_temperature_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_metadata(input_path, cal, summary, output_dir, experiment_name,
                    source_time_s, t_internal, t_arr):
    metadata = {
        "analysis_id": f"{experiment_name}_sample_prediction_v1",
        "source_internal_file": str(Path(input_path).resolve()),
        "nominal_calibration": {
            "k_eff_W_mK": cal.k_eff_W_mK,
            "cp_eff_J_kgK": cal.cp_eff_J_kgK,
            "rho_COC_kg_m3": cal.rho_COC_kg_m3,
        },
        "calibration_source": f"{cal.source_analysis} / 72C",
        "model": "BARE_TOP_COC_LAYERS (850 um total)",
        "geometry_um": {
            "bottom_COC": [0, 180],
            "sample": [180, 200],
            "oil": [200, 250],
            "top_COC": [250, 850],
        },
        "h_conv_W_m2K": H_CONV,
        "T_air_ambient_C": T_AMB,
        "initial_condition": {
            "mode": "first measured internal temperature",
            "value_C": float(t_internal[0]),
            "independently_validated_this_run": False,
        },
        "no_refitting": True,
        "no_top_COC_measurement_used": True,
        "source_point_count": int(len(source_time_s)),
        "time_range_s": summary["time_range_s"],
        "internal_range_C": [summary["internal"]["min_C"],
                             summary["internal"]["max_C"]],
        "predicted_sample_range_C": [summary["sample_predicted"]["min_C"],
                                     summary["sample_predicted"]["max_C"]],
        "predicted_top_range_C": [summary["top_predicted"]["min_C"],
                                  summary["top_predicted"]["max_C"]],
        "summary": summary,
        "git_commit": git_head(),
        "git_tag": git_describe(),
        "note": (
            "T_sample and T_top are phenomenological model estimates, NOT "
            "measured values. The model was calibrated on the 72C dataset "
            "(corrected measurement-time objective). No Top COC measurement "
            "was used in this run; no parameter was fitted. The 60C transfer "
            "validation is treated as temporarily inconclusive and was not "
            "used to adjust confidence here."
        ),
    }
    (output_dir / "sample_prediction_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="内部温度日志 xlsx 路径")
    p.add_argument("--sheet", default=SHEET)
    p.add_argument("--time-col", default=TIME_COL)
    p.add_argument("--temp-col", default=TEMP_COL)
    p.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data = load_internal_data(args.input, sheet=args.sheet,
                              time_col=args.time_col, temp_col=args.temp_col)
    print(f"[data] {data['n_valid']} 点, t [{data['first_time']:.3f}, "
          f"{data['last_time']:.3f}] s, dt median {data['median_dt']:.4f} s")
    print(f"  T_internal: {data['T_min_C']:.3f} -> {data['T_max_C']:.3f} C, "
          f"initial {data['T_initial_C']:.3f} C")
    cal, layers, mats, h_conv, t_amb = build_frozen_config()
    print(f"[frozen] {cal.name}: k_eff={cal.k_eff_W_mK} "
          f"cp_eff={cal.cp_eff_J_kgK} rho={cal.rho_COC_kg_m3} — NO refit")
    print(f"[initial] T_initial = {data['T_initial_C']:.3f} C "
          f"(first internal; model assumption)")

    summary, trace = predict_sample_temperature(
        data["source_time_s"], data["elapsed_time_s"],
        data["T_internal_C"], args.output_dir,
        experiment_name=args.experiment_name,
        input_path=args.input,
    )
    print(f"[sample] min {summary['sample_predicted']['min_C']:.3f} C "
          f"(t={summary['sample_predicted']['time_of_min_s']:.1f} s), "
          f"max {summary['sample_predicted']['max_C']:.3f} C "
          f"(t={summary['sample_predicted']['time_of_max_s']:.1f} s)")
    print(f"[cycles] {len(summary.get('cycles', []))} detected")
    print(f"[output] {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
