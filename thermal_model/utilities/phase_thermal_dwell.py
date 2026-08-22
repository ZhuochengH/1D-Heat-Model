#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正的相特定热停留时间 (dwell) 计算模块
========================================

问题:
    旧分析把「全协议 dwell / 重复周期数」当作每周期 dwell, 可能混入
    激活相 (初始长时间高温/预处理相) 与非 PCR 时段。

本模块为每个温度阈值计算四种互斥的停留时间:

    1. TOTAL_PROTOCOL_DWELL      : 整个有效协议区间积分;
    2. ACTIVATION_DWELL          : 仅激活/预周期区间;
    3. REPEATED_CYCLE_TOTAL_DWELL: 所有重复周期区间之和;
    4. PER_CYCLE_DWELL           : 每个重复周期单独计算
                                   (mean/median/min/max/std/正值周期数)。

相定义 (显式区间, 时间戳区间积分, 非均匀 dt 安全):

    activation_interval      = [t[0], 第一个重复周期起点]
    repeated_cycle_intervals = [[start_i, end_i]] (end_i = 下一周期起点
                              或最后峰后 +30 s 窗口或迹线末尾)
    post_cycle_interval      = [最后一个重复周期结束, t[-1]] (若存在)

严禁:
    total_protocol_dwell / cycle_count
    作为重复周期平均 dwell。
"""
from typing import Dict, List, Optional, Tuple

import numpy as np


# 标准阈值
DWELL_THRESHOLDS = (75.0, 80.0, 85.0, 90.0, 92.0, 94.0, 95.0)


# ============================================================
# 区间内阈值 dwell (时间戳线性插值交点积分)
# ============================================================

def dwell_ge_in_interval(t, T, threshold, t_start, t_end):
    """在 [t_start, t_end] 内积分 I(T(t) >= threshold) dt。

    实际时间戳区间 [t_i, t_{i+1}] 线性判定跨越交点;
    区间裁剪到数据范围; 非均匀 dt 安全。
    """
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    lo, hi = float(t_start), float(t_end)
    if hi <= lo:
        return 0.0
    n = len(t)
    total = 0.0
    for i in range(n - 1):
        a, b = t[i], t[i + 1]
        if b <= lo or a >= hi:
            continue
        a = max(a, lo)
        b = min(b, hi)
        if b <= a:
            continue
        Ta = T[i]
        Tb = T[i + 1]
        if Ta >= threshold and Tb >= threshold:
            total += (b - a)
        elif Ta >= threshold or Tb >= threshold:
            if Tb != Ta:
                frac = (threshold - Ta) / (Tb - Ta)
                if Ta >= threshold:
                    total += (b - a) * (1.0 - frac)
                else:
                    total += (b - a) * frac
    return float(total)


def dwell_full_range(t, T, thresholds=DWELL_THRESHOLDS):
    """整个有效协议的 dwell (每个阈值)。"""
    t = np.asarray(t, dtype=float)
    lo, hi = float(t[0]), float(t[-1])
    return {f"sample_ge_{th:.0f}C_s": dwell_ge_in_interval(
        t, T, th, lo, hi) for th in thresholds}


# ============================================================
# 相区间构建
# ============================================================

def build_phase_intervals(activation, repeated_cycles, t):
    """构建显式相区间。

    参数:
        activation       : detect_activation_and_repeated_cycles 返回的
                           activation dict 或 None;
        repeated_cycles  : 重复周期 list[dict] (含 cycle_start_time_s,
                           internal_peak_time_s);
        t                : 时间轴 (用于末尾裁剪)。

    返回 dict:
        activation_interval      : [start, end] 或 None;
        repeated_cycle_intervals: list[[start, end]];
        post_cycle_interval      : [start, end] 或 None;
        t_start / t_end          : 数据范围。
    """
    t = np.asarray(t, dtype=float)
    t0, tN = float(t[0]), float(t[-1])

    repeated_intervals: List[Tuple[float, float]] = []
    n_cyc = len(repeated_cycles)
    for k, c in enumerate(repeated_cycles):
        s = float(c["cycle_start_time_s"])
        if k + 1 < n_cyc:
            e = min(float(repeated_cycles[k + 1]["cycle_start_time_s"]), tN)
        else:
            e = min(float(c["internal_peak_time_s"]) + 30.0, tN)
        if e > s:
            repeated_intervals.append((s, e))

    # 激活区间 = [t0, 第一个重复周期起点] (若存在激活相)
    activation_interval: Optional[Tuple[float, float]] = None
    if activation is not None and repeated_intervals:
        activation_interval = (t0, repeated_intervals[0][0])

    # 末尾非周期相
    post_cycle_interval: Optional[Tuple[float, float]] = None
    if repeated_intervals:
        last_end = repeated_intervals[-1][1]
        if last_end < tN - 1e-9:
            post_cycle_interval = (last_end, tN)

    return {
        "activation_interval": activation_interval,
        "repeated_cycle_intervals": repeated_intervals,
        "post_cycle_interval": post_cycle_interval,
        "t_start": t0,
        "t_end": tN,
    }


# ============================================================
# 相特定 dwell 表
# ============================================================

def phase_specific_dwell_table(t, T_sample, intervals,
                               thresholds=DWELL_THRESHOLDS):
    """每个阈值的相特定 dwell 表。

    返回 (phase_rows, per_cycle_rows):
        phase_rows: list[dict]
            threshold_C
            total_protocol_dwell_s
            activation_dwell_s
            repeated_cycle_total_dwell_s
            repeated_cycle_mean_dwell_s
            repeated_cycle_median_dwell_s
            repeated_cycle_min_dwell_s
            repeated_cycle_max_dwell_s
            repeated_cycle_std_dwell_s
            cycles_with_positive_dwell
            repeated_cycle_count
        per_cycle_rows: list[dict]  (cycle_number, threshold_C, dwell_s)
    """
    t = np.asarray(t, dtype=float)
    T = np.asarray(T_sample, dtype=float)
    act = intervals["activation_interval"]
    rep = intervals["repeated_cycle_intervals"]
    post = intervals["post_cycle_interval"]
    t0, tN = intervals["t_start"], intervals["t_end"]
    n_cyc = len(rep)

    phase_rows = []
    per_cycle_rows = []
    for th in thresholds:
        total = dwell_ge_in_interval(t, T, th, t0, tN)
        act_dw = dwell_ge_in_interval(t, T, th, *act) if act else 0.0
        rep_total = sum(dwell_ge_in_interval(t, T, th, s, e)
                        for s, e in rep)
        per_cyc = [dwell_ge_in_interval(t, T, th, s, e) for s, e in rep]
        if per_cyc:
            mean = float(np.mean(per_cyc))
            median = float(np.median(per_cyc))
            mn = float(np.min(per_cyc))
            mx = float(np.max(per_cyc))
            sd = float(np.std(per_cyc)) if len(per_cyc) > 1 else 0.0
            n_pos = int(np.sum(np.array(per_cyc) > 0))
        else:
            mean = median = mn = mx = sd = np.nan
            n_pos = 0
        phase_rows.append({
            "threshold_C": th,
            "total_protocol_dwell_s": total,
            "activation_dwell_s": act_dw,
            "repeated_cycle_total_dwell_s": rep_total,
            "repeated_cycle_mean_dwell_s": mean,
            "repeated_cycle_median_dwell_s": median,
            "repeated_cycle_min_dwell_s": mn,
            "repeated_cycle_max_dwell_s": mx,
            "repeated_cycle_std_dwell_s": sd,
            "cycles_with_positive_dwell": n_pos,
            "repeated_cycle_count": n_cyc,
        })
        for ci, dw in enumerate(per_cyc, start=1):
            per_cycle_rows.append({
                "cycle_number": ci,
                "threshold_C": th,
                "dwell_s": dw,
            })
    return phase_rows, per_cycle_rows


# ============================================================
# 相特定峰值统计 (激活 vs 重复周期分离)
# ============================================================

def phase_peak_stats(repeated_metrics, activation=None, t_sample=None):
    """分离激活峰与重复周期峰的样品峰值统计。

    返回 dict:
        activation_sample_max: float|None
        repeated_cycle_sample_peak_min/max/mean/median
        (重复周期峰严格来自 repeated_metrics, 不含激活)
    """
    peaks = [float(c["sample_peak_C"]) for c in repeated_metrics]
    act_max = None
    if activation is not None and t_sample is not None:
        # 激活相样品最大值: 激活峰后窗口 (与检测一致)
        t = np.asarray(t_sample["t"], dtype=float)
        ts = np.asarray(t_sample["T_sample"], dtype=float)
        pidx = int(np.searchsorted(t, float(activation["internal_peak_time_s"])))
        hi = min(len(ts) - 1, pidx + 20)
        act_max = float(np.max(ts[pidx:hi + 1])) if hi >= pidx else None
    out = {"activation_sample_max": act_max}
    if peaks:
        out.update({
            "repeated_cycle_sample_peak_min_C": float(np.min(peaks)),
            "repeated_cycle_sample_peak_max_C": float(np.max(peaks)),
            "repeated_cycle_sample_peak_mean_C": float(np.mean(peaks)),
            "repeated_cycle_sample_peak_median_C": float(np.median(peaks)),
        })
    else:
        out.update({
            "repeated_cycle_sample_peak_min_C": None,
            "repeated_cycle_sample_peak_max_C": None,
            "repeated_cycle_sample_peak_mean_C": None,
            "repeated_cycle_sample_peak_median_C": None,
        })
    return out
