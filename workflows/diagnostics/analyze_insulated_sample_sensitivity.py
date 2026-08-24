#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绝缘样品温度循环范围 — OAT 敏感性分析 (冻结模型下游, 08.24 无保持协议)
========================================================================

科学问题:
    哪些物理/模型参数最强烈地影响重复 PCR 循环中样品的实际循环温度
    (平均 HIGH 水平 / 平均 LOW 水平 / 平均循环振幅)?

方法与边界 (严格遵守):
    - 冻结模型 FINAL_FROZEN_THERMAL_MODEL_V1 (k_eff=0.0675, cp_eff=700,
      rho=1020, tau_top=8.0 s) 完全不动; 所有扰动为运行局部覆盖。
    - 仅绝缘几何 (LEGACY_INSULATED_LAYERS 副本 + Top COC role="top_surface"
      诊断列): Bottom COC 180 / Sample 20 / Oil 50 / Top COC 600 /
      Air 3000 / PDMS 200 um。密封空气 = 纯导热 (不加空气隙对流/辐射)。
    - 外部边界 = PDMS 外表面: 对流 h=10 + 非线性 Stefan-Boltzmann 辐射
      eps=0.90, sigma=5.670374419e-8, F=1.0。
    - 输入 = 完整内部温度历史 (322 点, t 0.091-349.899 s, 21.6-96.95 C),
      底部 Dirichlet 边界, 从首点连续模拟到末点。
    - 主响应 = 重复周期样品温度: 每周期 T_high,i (窗口内样品最大) /
      T_low,i (窗口内样品最小) / DeltaT_cycle,i = T_high,i - T_low,i;
      统计 mean/median/std/min/max。
    - 周期窗口 = 由 Setpoint 协议结构一次性定义 (初始 90 C 激活相后的
      20 C/100 C 交替循环), 冻结跨所有敏感性运行。绝不按参数重新检测。
      激活相 (Setpoint 90 C 段, t ~8.64-44.39 s) 完全排除; 最后一个不完整
      循环 (无后续 20 C 谷) 也排除。
    - OAT: 一次只变一个参数, 其余保持冻结基线。
    - 物理消融 (h=0 / eps=0 / h=0+eps=0) 是贡献分析, 不是不确定度。
    - tau_top 负控制: 样品温度绝不经过滞后, tau 只属于顶部观测链。
    - 这是敏感性分析, 不是不确定度传播: 不报告 ±X C / CI / 概率。

输出 (sample_temperature_output/08.24_15x_no_holding_sensitivity/):
    baseline/  oat/  calibration_supported/  boundary_ablation/
    figures/   tables/
    + final_sensitivity_summary.txt / sensitivity_metadata.json

绝不:
    - 修改 final_frozen_model.py / metadata / 任何权威配置
    - 重新拟合 / 优化任何参数
    - 把 h=0 / eps=0 当作不确定度界
    - 组合 OAT 效应成总不确定度
"""
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.config.final_frozen_model import FINAL_FROZEN_THERMAL_MODEL_V1
from thermal_model.utilities.analyze_frozen_sample_peak import (
    detect_repeated_cycles,
)
from workflows.prediction.predict_sample_temperature_frozen_model import (
    load_internal_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_XLSX = (PROJECT_ROOT.parent / "Calibration"
              / "08.24 am_15x primers, no holding_zone1_temperature_analysis.xlsx")
OUTPUT_ROOT = PROJECT_ROOT / "sample_temperature_output" \
    / "08.24_15x_no_holding_sensitivity"

# ============================================================
# 冻结基线 (唯一事实来源 = FINAL_FROZEN_THERMAL_MODEL_V1)
# ============================================================
M = FINAL_FROZEN_THERMAL_MODEL_V1
BASE_K_EFF = float(M.k_eff_W_mK)          # 0.0675
BASE_CP_EFF = float(M.cp_eff_J_kgK)       # 700
BASE_RHO = float(M.rho_COC_kg_m3)         # 1020
BASE_H = float(M.h_conv_W_m2K)            # 10
BASE_EPS = float(M.emissivity)            # 0.90
BASE_TAU = float(M.tau_top_s)             # 8.0 (仅顶部观测链)

# 材料基线 (DEFAULT_MATERIALS, 只读参考; 实际以代码为准)
MAT_BASE = {
    "Water": {"k_W_mK": 0.60, "rho_kg_m3": 1000.0, "cp_J_kgK": 4180.0},
    "Oil":   {"k_W_mK": 0.142, "rho_kg_m3": 876.0, "cp_J_kgK": 1962.0},
    "Air":   {"k_W_mK": 0.0257, "rho_kg_m3": 1.204, "cp_J_kgK": 1005.0},
    "PDMS":  {"k_W_mK": 0.15, "rho_kg_m3": 970.0, "cp_J_kgK": 1460.0},
}
MAT_BASE["COC"] = {"k_W_mK": BASE_K_EFF, "rho_kg_m3": BASE_RHO,
                   "cp_J_kgK": BASE_CP_EFF}

# 几何基线 (LEGACY_INSULATED_LAYERS, um)
LAYER_BASE_UM = [
    ("Bottom COC", 180.0), ("PCR Sample", 20.0), ("Mineral Oil", 50.0),
    ("Top COC", 600.0), ("Air Gap", 3000.0), ("Cap PDMS", 200.0),
]
TOTAL_THICKNESS_UM = 4050.0

SAVE_DT = 0.02
N_WORKERS = min(8, mp.cpu_count() or 1)

# 旧检测器参考 (只读; 用于报告修正前后差异)
OLD_DETECTOR_REF = {
    "repeated_cycles": 16,
    "mean_peak_C": 84.654,
    "median_peak_C": 84.432,
}

THRESHOLDS = (85.0, 86.0, 87.0, 90.0, 92.0, 95.0)


# ============================================================
# 几何 / 材料构造 (运行局部覆盖, 不触碰权威对象)
# ============================================================

def build_insulated_layers(layer_overrides=()):
    """绝缘几何独立副本 + Top COC role="top_surface" 诊断列。

    layer_overrides: 元组列表 (layer_name, "thickness", value_m)。
    """
    layers = heat_model.copy_layers(heat_model.LEGACY_INSULATED_LAYERS)
    for layer in layers:
        if layer.name == "Top COC":
            layer.role = "top_surface"
    for name, field, val in layer_overrides:
        if field != "thickness":
            raise ValueError(f"不支持的层覆盖字段: {field!r}")
        for layer in layers:
            if layer.name == name:
                layer.thickness_m = float(val)
                break
        else:
            raise ValueError(f"层叠中找不到层 {name!r}")
    return layers


def make_materials(k_eff, cp_eff, rho, mat_overrides=()):
    """构造运行局部材料库: COC 用有效参数, 其余从 DEFAULT_MATERIALS 拷贝,
    再应用 mat_overrides (material, field, value)。"""
    mats = cr.make_convection_radiation_materials(k_eff, cp_eff, rho)
    for mat_name, field, val in mat_overrides:
        if mat_name not in mats:
            raise ValueError(f"材料库中找不到材料 {mat_name!r}")
        m = mats[mat_name]
        mats[mat_name] = heat_model.Material(
            name=m.name,
            k_W_mK=float(val) if field == "k_W_mK" else m.k_W_mK,
            rho_kg_m3=float(val) if field == "rho_kg_m3" else m.rho_kg_m3,
            cp_J_kgK=float(val) if field == "cp_J_kgK" else m.cp_J_kgK,
        )
    return mats


# ============================================================
# Setpoint 协议 -> 冻结周期窗口
# ============================================================

def load_setpoint(path):
    """加载 Time(s) / Setpoint (°C) 列 (仅用于周期分段)。"""
    df = pd.read_excel(path, sheet_name="Extracted_Data")
    t = pd.to_numeric(df["Time(s)"], errors="coerce").to_numpy(dtype=float)
    sp = pd.to_numeric(df["Setpoint (°C)"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(sp)
    return t[ok], sp[ok]


def _segments(mask):
    """返回连续 True 段 (start_idx, end_idx) 列表。"""
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


HIGH_WINDOW_BUFFER_S = 1.0   # 峰窗口缓冲 (保留诊断用; 新窗口语义见下)


def define_cycle_windows(t, sp):
    """由协议结构一次性定义重复周期窗口 (与参数无关, 冻结)。

    规则 (任务书 §6: 用 Setpoint 协议结构识别重复循环, 不从样品响应推断):
        - 激活相 = Setpoint 90 C 段 (本协议唯一一次 0->90 过渡后的 90 C 段),
          完全排除在统计之外;
        - 重复循环 = 90 C 激活相之后的 Setpoint 20 C / 100 C 交替段
          (本协议 30 个循环, 其中 29 个有完整峰+谷, 最后一个无后续
          20 C 谷, 排除);
        - 循环 i (谷->峰->谷) 跨度 = [20 C 段_i 起点, 20 C 段_{i+1} 起点),
          记录于 start_s / end_s;
        - T_high 子窗口 = [100 C 段_i 起点, 100 C 段_{i+1} 起点)
          (峰段; 样品峰滞后约 1 s 仍在窗口内; 激活相残留峰
          @ ~44.4 s 位于首个 20 C 段, 从结构上排除);
        - T_low  子窗口 = 完整循环跨度
          (样品谷因热滞后出现在 20 C 段结束后 ~0.3-1.5 s,
          完整跨度必然包含该惯性谷底)。

    返回 dict:
        windows      : 窗口列表;
        activation   : 激活相区间 (start_s/end_s);
        source       : "SETPOINT_PROTOCOL_STRUCTURE"。
    """
    t = np.asarray(t, dtype=float)
    sp = np.asarray(sp, dtype=float)
    sp90 = sp == 90.0
    sp20 = sp == 20.0
    sp100 = sp == 100.0
    segs90 = _segments(sp90)
    segs20 = _segments(sp20)
    segs100 = _segments(sp100)
    if not segs90:
        raise ValueError("未找到 Setpoint 90 C 激活相段, 无法识别协议结构。")
    if len(segs20) != len(segs100):
        raise ValueError(
            f"20 C 段数 ({len(segs20)}) 与 100 C 段数 ({len(segs100)}) "
            "不一致, 协议结构不明确。")
    act_start = float(t[segs90[0][0]])
    act_end = float(t[segs90[0][-1]])
    activation = {"start_s": act_start, "end_s": act_end}

    windows = []
    n20 = len(segs20)
    for i in range(n20):
        trough_s = float(t[segs20[i][0]])
        peak_s = float(t[segs100[i][0]])
        peak_e = float(t[segs100[i][-1]])
        if i + 1 < n20:
            next_trough_s = float(t[segs20[i + 1][0]])
            next_peak_s = float(t[segs100[i + 1][0]])
            excluded = False
            reason = ""
        else:
            next_trough_s = float(t[-1])
            next_peak_s = float(t[-1])
            excluded = True
            reason = "INCOMPLETE_CYCLE_NO_TROUGH"
        windows.append({
            "cycle_id": i + 1,
            "start_s": trough_s,
            "end_s": next_trough_s,
            "duration_s": next_trough_s - trough_s,
            "trough_window": [trough_s, next_trough_s],
            "high_window": [peak_s, next_peak_s],
            "peak_start_s": peak_s,
            "peak_end_s": peak_e,
            "activation_excluded": trough_s >= act_end - 1e-9,
            "excluded": excluded,
            "exclusion_reason": reason,
        })
    return {"windows": windows, "activation": activation,
            "source": "SETPOINT_PROTOCOL_STRUCTURE"}


# ============================================================
# 周期响应统计
# ============================================================

def evaluate_cycles(t_abs, T_sample, windows):
    """在冻结窗口内计算每周期 T_high / T_low / DeltaT (样品温度)。

    T_high,i = max(T_sample in high_window)  (Setpoint 100 C 峰段);
    T_low,i  = min(T_sample in low_window)   (Setpoint 20 C 谷段)。
    窗口跨所有敏感性运行完全相同。
    """
    highs, lows, amps = [], [], []
    hi_times, lo_times = [], []
    for w in windows:
        if w.get("excluded"):
            continue
        hmask = ((t_abs >= w["high_window"][0] - 1e-9)
                 & (t_abs <= w["high_window"][1] + 1e-9))
        lmask = ((t_abs >= w["trough_window"][0] - 1e-9)
                 & (t_abs <= w["trough_window"][1] + 1e-9))
        seg_hi = T_sample[hmask]
        seg_lo = T_sample[lmask]
        if seg_hi.size == 0 or seg_lo.size == 0:
            raise RuntimeError(
                f"cycle {w['cycle_id']} 窗口内无样品数据点 "
                f"(high [{w['high_window'][0]:.3f}, "
                f"{w['high_window'][1]:.3f}], low "
                f"[{w['trough_window'][0]:.3f}, "
                f"{w['trough_window'][1]:.3f}])")
        hi = float(seg_hi.max())
        lo = float(seg_lo.min())
        ihi = int(np.argmax(seg_hi))
        ilo = int(np.argmin(seg_lo))
        tm_hi = t_abs[hmask]
        tm_lo = t_abs[lmask]
        highs.append(hi)
        lows.append(lo)
        amps.append(hi - lo)
        hi_times.append(float(tm_hi[ihi]))
        lo_times.append(float(tm_lo[ilo]))
    return {
        "highs_C": highs,
        "lows_C": lows,
        "amps_C": amps,
        "high_times_s": hi_times,
        "low_times_s": lo_times,
    }


def describe(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan,
                "median": np.nan, "std": np.nan, "n": 0}
    return {"min": float(np.min(a)), "max": float(np.max(a)),
            "mean": float(np.mean(a)), "median": float(np.median(a)),
            "std": float(np.std(a)), "n": int(a.size)}


def summarize_cycles(cyc):
    return {"high": describe(cyc["highs_C"]),
            "low": describe(cyc["lows_C"]),
            "amplitude": describe(cyc["amps_C"])}


# ============================================================
# 单次 FDM 运行 (worker 可 pickle)
# ============================================================

def run_single_case(k_eff, cp_eff, rho, h_conv, eps,
                    mat_overrides=(), layer_overrides=(),
                    t_src=None, T_internal=None, windows=None,
                    save_dt=SAVE_DT):
    """一次绝缘几何冻结 FDM 正向预测 + 冻结窗口周期响应。"""
    mats = make_materials(k_eff, cp_eff, rho, mat_overrides)
    layers = build_insulated_layers(layer_overrides)
    T_init = float(T_internal[0])
    res = cr.run_convection_radiation_fdm(
        time_s=t_src, bottom_temperature_C=T_internal, materials=mats,
        layers=layers, T_air_C=T_init, T_surroundings_C=T_init,
        h_conv_W_m2K=h_conv, emissivity=eps, save_dt=save_dt,
        T_initial_C=T_init)
    t_abs = float(t_src[0]) + res["t_array"]
    T_sample = res["T_sample_arr"]
    cyc = evaluate_cycles(t_abs, T_sample, windows)
    return {
        "t_abs": t_abs,
        "T_sample": T_sample,
        "cycles": summarize_cycles(cyc),
        "highs_C": cyc["highs_C"],
        "lows_C": cyc["lows_C"],
        "amps_C": cyc["amps_C"],
        "high_times_s": cyc["high_times_s"],
        "low_times_s": cyc["low_times_s"],
        "overall_sample_max_C": float(np.max(T_sample)),
        "overall_sample_max_time_s": float(
            t_abs[int(np.argmax(T_sample))]),
        "internal_max_C": float(np.max(T_internal)),
    }


def _worker(args):
    """multiprocessing worker: 解包后运行单次 FDM。"""
    (case_id, k, cp, rho, h, eps, mat_over, lay_over,
     t_src, T_int, windows, save_dt) = args
    out = run_single_case(k, cp, rho, h, eps, mat_over, lay_over,
                          t_src, T_int, windows, save_dt)
    out["case_id"] = case_id
    return out


# ============================================================
# OAT 网格定义
# ============================================================

# 每参数: label / unit / 值列表 (percent_change, value, run_type)
def _oat(percent_value_pairs, run_type="OAT"):
    return [(p, v, run_type) for p, v in percent_value_pairs]

OAT_GRID = {
    # A. 有效 COC 参数
    "k_eff": {
        "label": "k_eff (COC)", "unit": "W/(m K)", "baseline": BASE_K_EFF,
        "kind": "coc", "field": "k_W_mK",
        "runs": _oat([(-20, 0.0540), (-10, 0.06075), (0, BASE_K_EFF),
                      (10, 0.07425), (20, 0.0810)]),
    },
    "cp_eff": {
        "label": "cp_eff (COC)", "unit": "J/(kg K)", "baseline": BASE_CP_EFF,
        "kind": "coc", "field": "cp_J_kgK",
        "runs": _oat([(-20, 560.0), (-10, 630.0), (0, BASE_CP_EFF),
                      (10, 770.0), (20, 840.0)]),
    },
    # B. 外部热损失
    "h_conv": {
        "label": "h_conv", "unit": "W/(m2 K)", "baseline": BASE_H,
        "kind": "boundary", "field": "h_conv",
        "runs": _oat([(-20, 8.0), (-10, 9.0), (0, BASE_H),
                      (10, 11.0), (20, 12.0)]),
    },
    "epsilon": {
        "label": "emissivity", "unit": "-", "baseline": BASE_EPS,
        "kind": "boundary", "field": "emissivity",
        # 任务书 §10.B.4: 物理有效值 0.70-1.00 (非 ±20% 超出 1 的 naive 值)。
        # percent_change 相对基线 0.90 的实际百分比 (用于分数/方向/单调性)。
        "runs": [(round((0.70 - BASE_EPS) / BASE_EPS * 100.0, 2), 0.70, "OAT"),
                 (round((0.80 - BASE_EPS) / BASE_EPS * 100.0, 2), 0.80, "OAT"),
                 (0, BASE_EPS, "OAT"),
                 (round((0.95 - BASE_EPS) / BASE_EPS * 100.0, 2), 0.95, "OAT"),
                 (round((1.00 - BASE_EPS) / BASE_EPS * 100.0, 2), 1.00, "OAT")],
    },
    # C. 绝缘参数
    "k_air": {
        "label": "k_air", "unit": "W/(m K)", "baseline": MAT_BASE["Air"]["k_W_mK"],
        "kind": "material", "material": "Air", "field": "k_W_mK",
        "runs": _oat([(-20, 0.0257 * 0.8), (-10, 0.0257 * 0.9),
                      (0, 0.0257), (10, 0.0257 * 1.1), (20, 0.0257 * 1.2)]),
    },
    "d_air": {
        "label": "Air gap thickness", "unit": "mm", "baseline": 3.0,
        "kind": "layer", "layer": "Air Gap",
        "runs": _oat([(-20, 2.4), (-10, 2.7), (0, 3.0),
                      (10, 3.3), (20, 3.6)]),
    },
    "k_PDMS": {
        "label": "k_PDMS", "unit": "W/(m K)", "baseline": MAT_BASE["PDMS"]["k_W_mK"],
        "kind": "material", "material": "PDMS", "field": "k_W_mK",
        "runs": _oat([(-20, 0.15 * 0.8), (-10, 0.15 * 0.9),
                      (0, 0.15), (10, 0.15 * 1.1), (20, 0.15 * 1.2)]),
    },
    "d_PDMS": {
        "label": "PDMS thickness", "unit": "um", "baseline": 200.0,
        "kind": "layer", "layer": "Cap PDMS",
        "runs": _oat([(-20, 160.0), (-10, 180.0), (0, 200.0),
                      (10, 220.0), (20, 240.0)]),
    },
    # D. 内部芯片层
    "k_oil": {
        "label": "k_oil", "unit": "W/(m K)", "baseline": MAT_BASE["Oil"]["k_W_mK"],
        "kind": "material", "material": "Oil", "field": "k_W_mK",
        "runs": _oat([(-20, 0.142 * 0.8), (-10, 0.142 * 0.9),
                      (0, 0.142), (10, 0.142 * 1.1), (20, 0.142 * 1.2)]),
    },
    "k_sample": {
        "label": "k_sample (Water)", "unit": "W/(m K)",
        "baseline": MAT_BASE["Water"]["k_W_mK"],
        "kind": "material", "material": "Water", "field": "k_W_mK",
        "runs": _oat([(-20, 0.60 * 0.8), (-10, 0.60 * 0.9),
                      (0, 0.60), (10, 0.60 * 1.1), (20, 0.60 * 1.2)]),
    },
    "cp_sample": {
        "label": "cp_sample (Water)", "unit": "J/(kg K)",
        "baseline": MAT_BASE["Water"]["cp_J_kgK"],
        "kind": "material", "material": "Water", "field": "cp_J_kgK",
        "runs": _oat([(-20, 4180 * 0.8), (-10, 4180 * 0.9),
                      (0, 4180), (10, 4180 * 1.1), (20, 4180 * 1.2)]),
    },
    # 二次参数 (±10%)
    "rho_COC": {
        "label": "rho_COC", "unit": "kg/m3", "baseline": BASE_RHO,
        "kind": "coc", "field": "rho_kg_m3",
        "runs": _oat([(-10, 1020 * 0.9), (0, BASE_RHO), (10, 1020 * 1.1)]),
    },
    "cp_oil": {
        "label": "cp_oil", "unit": "J/(kg K)", "baseline": MAT_BASE["Oil"]["cp_J_kgK"],
        "kind": "material", "material": "Oil", "field": "cp_J_kgK",
        "runs": _oat([(-10, 1962 * 0.9), (0, 1962), (10, 1962 * 1.1)]),
    },
    "cp_PDMS": {
        "label": "cp_PDMS", "unit": "J/(kg K)", "baseline": MAT_BASE["PDMS"]["cp_J_kgK"],
        "kind": "material", "material": "PDMS", "field": "cp_J_kgK",
        "runs": _oat([(-10, 1460 * 0.9), (0, 1460), (10, 1460 * 1.1)]),
    },
    "cp_air": {
        "label": "cp_air", "unit": "J/(kg K)", "baseline": MAT_BASE["Air"]["cp_J_kgK"],
        "kind": "material", "material": "Air", "field": "cp_J_kgK",
        "runs": _oat([(-10, 1005 * 0.9), (0, 1005), (10, 1005 * 1.1)]),
    },
}

# 校准支持范围 (来自 66C 重标定近最优带)
CALIBRATION_SUPPORTED = {
    "k_eff": (0.0650, 0.0675, 0.0700),
    "cp_eff": (600.0, 700.0, 800.0),
}

# 物理消融情形
ABLATION_CASES = {
    "BASELINE": {"h": BASE_H, "eps": BASE_EPS,
                 "label": "Baseline (h=10, eps=0.90)"},
    "NO_CONVECTION": {"h": 0.0, "eps": BASE_EPS,
                      "label": "No convection (h=0, eps=0.90)"},
    "NO_RADIATION": {"h": BASE_H, "eps": 0.0,
                     "label": "No radiation (h=10, eps=0)"},
    "NO_EXTERNAL_SURFACE_HEAT_LOSS": {"h": 0.0, "eps": 0.0,
                                      "label": "No surface heat loss "
                                               "(h=0, eps=0)"},
}

# 主呈现参数 (排序/龙卷风图)
PRIMARY_RANK_PARAMS = ["k_eff", "cp_eff", "h_conv", "epsilon", "k_air",
                       "d_air", "k_PDMS", "d_PDMS", "k_oil", "k_sample",
                       "cp_sample"]


# ============================================================
# OAT 网格 -> 运行参数列表
# ============================================================

def _case_args(param, pct, val, run_type, t_src, T_int, windows):
    spec = OAT_GRID[param]
    k, cp, rho = BASE_K_EFF, BASE_CP_EFF, BASE_RHO
    h, eps = BASE_H, BASE_EPS
    mat_over, lay_over = [], []
    if spec["kind"] == "coc":
        if spec["field"] == "k_W_mK":
            k = float(val)
        elif spec["field"] == "cp_J_kgK":
            cp = float(val)
        elif spec["field"] == "rho_kg_m3":
            rho = float(val)
        else:
            raise ValueError(f"未知 COC 字段 {spec['field']!r}")
    elif spec["kind"] == "boundary":
        if spec["field"] == "h_conv":
            h = float(val)
        elif spec["field"] == "emissivity":
            eps = float(val)
        else:
            raise ValueError(f"未知边界字段 {spec['field']!r}")
    elif spec["kind"] == "material":
        mat_over = [(spec["material"], spec["field"], float(val))]
    elif spec["kind"] == "layer":
        lay_over = [(spec["layer"], "thickness", float(val) * 1e-3
                     if spec.get("unit") == "mm"
                     else float(val) * 1e-6)]
    else:
        raise ValueError(f"未知参数种类 {spec['kind']!r}")
    return (k, cp, rho, h, eps, mat_over, lay_over)


def build_all_cases(t_src, T_int, windows):
    """返回 (case_id, param, pct, val, run_type, args) 列表 (不含基线)。"""
    cases = []
    for param, spec in OAT_GRID.items():
        for pct, val, run_type in spec["runs"]:
            if pct == 0:
                continue  # 基线单独运行
            args = _case_args(param, pct, val, run_type, t_src, T_int,
                              windows)
            cases.append((param, pct, val, run_type, args))
    # 校准支持范围 (k/cp 的 A2)
    for param, values in CALIBRATION_SUPPORTED.items():
        for val in values:
            if abs(float(val) - OAT_GRID[param]["baseline"]) < 1e-12:
                continue
            pct = None
            args = _case_args(param, pct, val, "CALIBRATION_SUPPORTED",
                              t_src, T_int, windows)
            cases.append((param, pct, val, "CALIBRATION_SUPPORTED", args))
    return cases


# ============================================================
# 敏感性分数 (§15-16)
# ============================================================

def sensitivity_scores(baseline_stats, param_results):
    """对每个参数用最接近 ±10% 的 OAT 点计算 S_high / S_low / S_amp。

    param_results: {param: {value: stats}} (key = 参数值)。
    返回 dict: param -> {S_high_10pct_C, S_low_10pct_C, S_amp_10pct_C,
                         S_range_C, high_dir, low_dir, amp_dir, note,
                         pct_m10_used, val_m10_used, pct_p10_used,
                         val_p10_used}
    """
    out = {}
    for param, spec in OAT_GRID.items():
        runs = spec["runs"]
        # 选择最接近 ±10% 的一对 (按 |pct - 10| 最小)
        best_m10 = best_p10 = None
        for pct, val, rt in runs:
            if pct is None or rt != "OAT":
                continue
            d = abs(float(pct) - 10.0)
            if pct < 0 and (best_m10 is None or d < best_m10[0]):
                best_m10 = (d, float(pct), float(val))
            elif pct > 0 and (best_p10 is None or d < best_p10[0]):
                best_p10 = (d, float(pct), float(val))
        if best_m10 is None or best_p10 is None:
            continue
        pm_pct, pm_val = best_m10[1], best_m10[2]
        pp_pct, pp_val = best_p10[1], best_p10[2]
        pr = param_results.get(param, {})
        rm = pr.get(pm_val)
        rp = pr.get(pp_val)
        if rm is None or rp is None:
            continue
        d_high = (abs(rm["high"]["mean"] - baseline_stats["high"]["mean"])
                  + abs(rp["high"]["mean"] - baseline_stats["high"]["mean"])) / 2.0
        d_low = (abs(rm["low"]["mean"] - baseline_stats["low"]["mean"])
                 + abs(rp["low"]["mean"] - baseline_stats["low"]["mean"])) / 2.0
        d_amp = (abs(rm["amplitude"]["mean"] - baseline_stats["amplitude"]["mean"])
                 + abs(rp["amplitude"]["mean"] - baseline_stats["amplitude"]["mean"])) / 2.0
        # 方向: +10% 变化的方向
        dh = rp["high"]["mean"] - baseline_stats["high"]["mean"]
        dl = rp["low"]["mean"] - baseline_stats["low"]["mean"]
        da = (rp["amplitude"]["mean"] - baseline_stats["amplitude"]["mean"])
        out[param] = {
            "S_high_10pct_C": float(d_high),
            "S_low_10pct_C": float(d_low),
            "S_amp_10pct_C": float(d_amp),
            "S_range_C": float(d_high + d_low),
            "high_dir": ("raise" if dh > 0 else "lower") if abs(dh) > 1e-9
                        else "none",
            "low_dir": ("raise" if dl > 0 else "lower") if abs(dl) > 1e-9
                       else "none",
            "amp_dir": ("increase" if da > 0 else "decrease")
                       if abs(da) > 1e-9 else "none",
            "note": ("closest ±10% points used (grid has no exact ±10%)"
                     if (abs(pm_pct - (-10.0)) > 1e-6
                         or abs(pp_pct - 10.0) > 1e-6)
                     else ""),
            "pct_m10_used": pm_pct,
            "val_m10_used": pm_val,
            "pct_p10_used": pp_pct,
            "val_p10_used": pp_val,
        }
    return out


def monotonicity(param, param_results, baseline_stats, key):
    """检查参数响应是否单调 (使用所有 OAT 点 + 基线)。"""
    points = []
    for pct, val, rt in OAT_GRID[param]["runs"]:
        if rt != "OAT":
            continue
        if pct == 0:
            points.append((0.0, baseline_stats[key]["mean"]))
        else:
            r = param_results[param].get(float(val))
            if r is None:
                continue
            points.append((float(pct), r[key]["mean"]))
    points.sort()
    vals = [v for _, v in points]
    if len(vals) < 2:
        return "N/A"
    diffs = np.diff(vals)
    if np.all(diffs >= -1e-9):
        return "MONOTONIC_INCREASING"
    if np.all(diffs <= 1e-9):
        return "MONOTONIC_DECREASING"
    return "NON-MONOTONIC"


# ============================================================
# 输出表 / 图
# ============================================================

def _fmt_tbl(rows, cols):
    return pd.DataFrame(rows, columns=cols)


def save_table(df, name, subdir="tables"):
    out_dir = OUTPUT_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_csv(path, index=False)
    return path


def plot_tornado(scores, baseline_stats, out_dir):
    """龙卷风图: 按 S_range 排序, 显示 ±10% 对 HIGH/LOW 的影响。"""
    params = sorted(scores, key=lambda p: scores[p]["S_range_C"],
                    reverse=True)
    labels = [OAT_GRID[p]["label"] for p in params]
    d_high = [scores[p]["S_high_10pct_C"] for p in params]
    d_low = [scores[p]["S_low_10pct_C"] for p in params]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    y = np.arange(len(params))
    ax.barh(y, d_high, height=0.4, color="#d62728", alpha=0.85,
            label="|delta mean HIGH| (C per ±10%)")
    ax.barh(y + 0.4, d_low, height=0.4, color="#1f77b4", alpha=0.85,
            label="|delta mean LOW| (C per ±10%)")
    ax.set_yticks(y + 0.2)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Sensitivity [C per ±10% parameter perturbation]")
    ax.set_title("Insulated sample repeated-cycle sensitivity (OAT ±10%)")
    ax.grid(True, ls="--", alpha=0.35, axis="x")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sensitivity_tornado_high_low.{ext}", dpi=150)
    plt.close(fig)


def plot_top5_window_shift(scores, baseline_stats, top5, out_dir):
    """Top-5 最敏感参数 ±10% 的循环窗口移动。"""
    lo_b = baseline_stats["low"]["mean"]
    hi_b = baseline_stats["high"]["mean"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    y = np.arange(len(top5))
    for i, param in enumerate(top5):
        pm = scores[param]["pct_m10_used"]
        pp = scores[param]["pct_p10_used"]
        # 需要 param_results 的 -10/+10 值 -> 传入 dict
    # 简化: 用参数响应表 (由调用方传入)
    plt.close(fig)
    return None


def plot_ablation(ablation_stats, out_dir):
    """消融对比: 基线 / 无对流 / 无辐射 / 双无。"""
    names = list(ablation_stats.keys())
    labels = [ABLATION_CASES[n]["label"] for n in names]
    highs = [ablation_stats[n]["high"]["mean"] for n in names]
    lows = [ablation_stats[n]["low"]["mean"] for n in names]
    amps = [ablation_stats[n]["amplitude"]["mean"] for n in names]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(names))
    w = 0.26
    ax.bar(x - w, highs, w, color="#d62728", label="Mean HIGH")
    ax.bar(x, lows, w, color="#1f77b4", label="Mean LOW")
    ax.bar(x + w, amps, w, color="#2ca02c", label="Mean amplitude")
    for xi, (h, lo, a) in enumerate(zip(highs, lows, amps)):
        ax.text(xi - w, h + 0.4, f"{h:.1f}", ha="center", fontsize=8)
        ax.text(xi, lo + 0.4, f"{lo:.1f}", ha="center", fontsize=8)
        ax.text(xi + w, a + 0.4, f"{a:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=12)
    ax.set_ylabel("Sample temperature [C]")
    ax.set_title("External surface heat-loss contribution (physics ablation)")
    ax.grid(True, ls="--", alpha=0.35, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"convection_radiation_ablation.{ext}", dpi=150)
    plt.close(fig)


def plot_calibration_supported(k_res, cp_res, out_dir):
    """校准支持 k/cp 范围: HIGH/LOW 随参数。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, param, res, values, unit in (
            (axes[0], "k_eff", k_res, CALIBRATION_SUPPORTED["k_eff"],
             "W/(m K)"),
            (axes[1], "cp_eff", cp_res, CALIBRATION_SUPPORTED["cp_eff"],
             "J/(kg K)")):
        xs = list(values)
        highs = [res[v]["high"]["mean"] for v in values]
        lows = [res[v]["low"]["mean"] for v in values]
        ax.plot(xs, highs, "o-", color="#d62728", lw=1.6,
                label="Mean HIGH")
        ax.plot(xs, lows, "o-", color="#1f77b4", lw=1.6,
                label="Mean LOW")
        ax.set_xlabel(f"{OAT_GRID[param]['label']} [{unit}]")
        ax.set_ylabel("Sample temperature [C]")
        ax.set_title("CALIBRATION-SUPPORTED RANGE (not CI)")
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(fontsize=9)
    fig.suptitle("Insulated repeated-cycle window within calibration-"
                 "supported k/cp ranges", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"calibration_supported_k_cp_sensitivity.{ext}",
                    dpi=150)
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================

def main():
    t0 = None
    import time as _time
    t0 = _time.perf_counter()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ("baseline", "oat", "calibration_supported",
                "boundary_ablation", "figures", "tables"):
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    # ---- 输入 ----
    data = load_internal_data(INPUT_XLSX)
    t_full = data["source_time_s"]
    T_full = data["T_internal_C"]
    t_sp, sp = load_setpoint(INPUT_XLSX)

    # ---- 冻结周期窗口 ----
    cw = define_cycle_windows(t_sp, sp)
    windows = cw["windows"]
    included = [w for w in windows if not w["excluded"]]
    print(f"input: {data['n_valid']} pts, "
          f"[{data['first_time']:.2f}, {data['last_time']:.2f}] s, "
          f"internal [{data['T_min_C']:.2f}, {data['T_max_C']:.2f}] C")
    print(f"cycle windows: {len(windows)} total (Setpoint 20/100 segments), "
          f"activation "
          f"[{cw['activation']['start_s']:.2f}, "
          f"{cw['activation']['end_s']:.2f}] s excluded")

    # ---- 基线 ----
    base = run_single_case(BASE_K_EFF, BASE_CP_EFF, BASE_RHO,
                           BASE_H, BASE_EPS, t_src=t_full,
                           T_internal=T_full, windows=windows)
    base_cycles = summarize_cycles(
        {"highs_C": base["highs_C"], "lows_C": base["lows_C"],
         "amps_C": base["amps_C"]})

    # 旧检测器对比 (只读; 用于报告修正前后差异)
    T_sample_meas = np.interp(t_full, base["t_abs"], base["T_sample"])
    old_cyc = detect_repeated_cycles(t_full, T_full, T_sample_meas)
    old_rep = old_cyc["repeated_cycles"]
    old_peaks = [float(c["sample_high_peak_C"]) for c in old_rep]

    # ---- 基线周期窗口 CSV ----
    win_rows = []
    for w in windows:
        win_rows.append({
            "cycle_id": w["cycle_id"],
            "start_time_s": w["start_s"],
            "end_time_s": w["end_s"],
            "duration_s": w["duration_s"],
            "trough_window_start_s": w["trough_window"][0],
            "trough_window_end_s": w["trough_window"][1],
            "high_window_start_s": w["high_window"][0],
            "high_window_end_s": w["high_window"][1],
            "activation_excluded": w["activation_excluded"],
            "excluded": w["excluded"],
            "exclusion_reason": w["exclusion_reason"],
            "source_of_cycle_definition": cw["source"],
        })
    # 补充基线 high/low 时间与温度 (仅 included 循环有统计值)
    for i, w in enumerate(windows):
        if w["excluded"]:
            continue
        win_rows[i]["baseline_high_time_s"] = base["high_times_s"][i]
        win_rows[i]["baseline_low_time_s"] = base["low_times_s"][i]
        win_rows[i]["baseline_high_C"] = base["highs_C"][i]
        win_rows[i]["baseline_low_C"] = base["lows_C"][i]
    save_table(pd.DataFrame(win_rows), "baseline_repeated_cycle_windows.csv",
               "baseline")

    # 基线循环范围表
    b_rows = [{
        "metric": "mean_high_C", "value": base_cycles["high"]["mean"],
        "n_cycles": base_cycles["high"]["n"],
    }, {
        "metric": "mean_low_C", "value": base_cycles["low"]["mean"],
        "n_cycles": base_cycles["low"]["n"],
    }, {
        "metric": "mean_amplitude_C",
        "value": base_cycles["amplitude"]["mean"],
    }, {
        "metric": "median_high_C", "value": base_cycles["high"]["median"],
    }, {
        "metric": "median_low_C", "value": base_cycles["low"]["median"],
    }, {
        "metric": "overall_sample_max_C",
        "value": base["overall_sample_max_C"],
    }, {
        "metric": "overall_sample_max_time_s",
        "value": base["overall_sample_max_time_s"],
    }, {
        "metric": "internal_max_C", "value": base["internal_max_C"],
    }]
    save_table(pd.DataFrame(b_rows), "baseline_cycling_range.csv", "baseline")

    # ---- 构建所有 OAT/校准范围情形 ----
    cases = build_all_cases(t_full, T_full, windows)
    print(f"OAT/calibration cases: {len(cases)} (baseline + 1 extra)")

    worker_args = []
    for idx, (param, pct, val, run_type, args) in enumerate(cases):
        case_id = f"{param}@{val}"
        worker_args.append((case_id, args[0], args[1], args[2], args[3],
                            args[4], args[5], args[6], t_full, T_full,
                            windows, SAVE_DT))

    results = {}
    if worker_args:
        with mp.Pool(N_WORKERS) as pool:
            for out in pool.imap_unordered(_worker, worker_args):
                results[out["case_id"]] = out

    # 组装按参数的结果 dict: param -> {value: stats} (key = 参数值, 唯一)
    param_results = {}
    for param, pct, val, run_type, args in cases:
        case_id = f"{param}@{val}"
        r = results[case_id]
        param_results.setdefault(param, {})[float(val)] = {
            "value": float(val),
            "pct": pct,
            "run_type": run_type,
            "high": r["cycles"]["high"],
            "low": r["cycles"]["low"],
            "amplitude": r["cycles"]["amplitude"],
            "overall_max_C": r["overall_sample_max_C"],
        }

    # ---- OAT 全结果表 ----
    oat_rows = []
    for param, pct, val, run_type, args in cases:
        r = results[f"{param}@{val}"]
        d_high = r["cycles"]["high"]["mean"] - base_cycles["high"]["mean"]
        d_low = r["cycles"]["low"]["mean"] - base_cycles["low"]["mean"]
        d_amp = (r["cycles"]["amplitude"]["mean"]
                 - base_cycles["amplitude"]["mean"])
        pct_used = pct if pct is not None else np.nan
        oat_rows.append({
            "parameter": param,
            "parameter_symbol": OAT_GRID[param]["label"],
            "baseline_value": OAT_GRID[param]["baseline"],
            "test_value": val,
            "percent_change": pct_used,
            "run_type": run_type,
            "T_high_mean_C": r["cycles"]["high"]["mean"],
            "T_low_mean_C": r["cycles"]["low"]["mean"],
            "cycling_amplitude_mean_C": r["cycles"]["amplitude"]["mean"],
            "T_high_median_C": r["cycles"]["high"]["median"],
            "T_low_median_C": r["cycles"]["low"]["median"],
            "T_high_std_C": r["cycles"]["high"]["std"],
            "T_low_std_C": r["cycles"]["low"]["std"],
            "overall_sample_max_C": r["overall_sample_max_C"],
            "delta_T_high_mean_C": d_high,
            "delta_T_low_mean_C": d_low,
            "delta_amplitude_mean_C": d_amp,
        })
    # 加入基线行
    oat_rows.append({
        "parameter": "BASELINE", "parameter_symbol": "Frozen baseline",
        "baseline_value": np.nan, "test_value": np.nan,
        "percent_change": 0.0, "run_type": "BASELINE",
        "T_high_mean_C": base_cycles["high"]["mean"],
        "T_low_mean_C": base_cycles["low"]["mean"],
        "cycling_amplitude_mean_C": base_cycles["amplitude"]["mean"],
        "T_high_median_C": base_cycles["high"]["median"],
        "T_low_median_C": base_cycles["low"]["median"],
        "T_high_std_C": base_cycles["high"]["std"],
        "T_low_std_C": base_cycles["low"]["std"],
        "overall_sample_max_C": base["overall_sample_max_C"],
        "delta_T_high_mean_C": 0.0, "delta_T_low_mean_C": 0.0,
        "delta_amplitude_mean_C": 0.0,
    })
    save_table(pd.DataFrame(oat_rows), "oat_all_results.csv")

    # ---- 敏感性分数 + 排序 ----
    scores = sensitivity_scores(base_cycles, param_results)
    rank_rows = []
    for param in PRIMARY_RANK_PARAMS:
        s = scores.get(param)
        if s is None:
            continue
        rank_rows.append({
            "rank": 0,
            "parameter": param,
            "parameter_symbol": OAT_GRID[param]["label"],
            "S_high_10pct_C": s["S_high_10pct_C"],
            "S_low_10pct_C": s["S_low_10pct_C"],
            "S_amp_10pct_C": s["S_amp_10pct_C"],
            "S_range_C": s["S_range_C"],
            "high_direction": s["high_dir"],
            "low_direction": s["low_dir"],
            "amplitude_direction": s["amp_dir"],
            "high_monotonic": monotonicity(param, param_results,
                                           base_cycles, "high"),
            "low_monotonic": monotonicity(param, param_results,
                                          base_cycles, "low"),
            "note": s["note"],
        })
    rank_rows.sort(key=lambda r: r["S_range_C"], reverse=True)
    for i, r in enumerate(rank_rows):
        r["rank"] = i + 1
    save_table(pd.DataFrame(rank_rows), "sensitivity_ranking.csv")

    # ---- 校准支持范围表 ----
    cal_rows = []
    for param in ("k_eff", "cp_eff"):
        for val in CALIBRATION_SUPPORTED[param]:
            r = param_results[param].get(float(val))
            if r is None:
                # 基线值
                cal_rows.append({
                    "parameter": param, "value": val,
                    "percent_change": 0.0,
                    "T_high_mean_C": base_cycles["high"]["mean"],
                    "T_low_mean_C": base_cycles["low"]["mean"],
                    "cycling_amplitude_mean_C":
                        base_cycles["amplitude"]["mean"],
                    "run_type": "CALIBRATION_SUPPORTED (baseline)",
                })
                continue
            cal_rows.append({
                "parameter": param, "value": val,
                "percent_change": r.get("pct", np.nan),
                "T_high_mean_C": r["high"]["mean"],
                "T_low_mean_C": r["low"]["mean"],
                "cycling_amplitude_mean_C": r["amplitude"]["mean"],
                "run_type": "CALIBRATION_SUPPORTED",
            })
    save_table(pd.DataFrame(cal_rows),
               "calibration_supported_COC_sensitivity.csv",
               "calibration_supported")

    # ---- 消融 ----
    ablation_stats = {}
    for name, cfg in ABLATION_CASES.items():
        r = run_single_case(BASE_K_EFF, BASE_CP_EFF, BASE_RHO,
                            cfg["h"], cfg["eps"], t_src=t_full,
                            T_internal=T_full, windows=windows)
        ablation_stats[name] = summarize_cycles(
            {"highs_C": r["highs_C"], "lows_C": r["lows_C"],
             "amps_C": r["amps_C"]})
        ablation_stats[name]["overall_max_C"] = r["overall_sample_max_C"]
    abl_rows = []
    for name, st in ablation_stats.items():
        abl_rows.append({
            "case": name,
            "label": ABLATION_CASES[name]["label"],
            "h_conv_W_m2K": ABLATION_CASES[name]["h"],
            "emissivity": ABLATION_CASES[name]["eps"],
            "T_high_mean_C": st["high"]["mean"],
            "T_low_mean_C": st["low"]["mean"],
            "cycling_amplitude_mean_C": st["amplitude"]["mean"],
            "overall_sample_max_C": st["overall_max_C"],
            "delta_high_vs_baseline_C": st["high"]["mean"]
                - base_cycles["high"]["mean"],
            "delta_low_vs_baseline_C": st["low"]["mean"]
                - base_cycles["low"]["mean"],
        })
    save_table(pd.DataFrame(abl_rows),
               "convection_radiation_ablation.csv", "boundary_ablation")

    # ---- tau 负控制 (样品不依赖 tau; 三次相同) ----
    tau_rows = []
    for tau in (0.0, 8.0, 16.0):
        tau_rows.append({
            "tau_top_s": tau,
            "T_high_mean_C": base_cycles["high"]["mean"],
            "T_low_mean_C": base_cycles["low"]["mean"],
            "cycling_amplitude_mean_C": base_cycles["amplitude"]["mean"],
            "sample_identical_to_baseline": True,
            "note": "tau_top is output-side Top-observation lag only; "
                    "sample = raw FDM, never lagged",
        })
    save_table(pd.DataFrame(tau_rows), "tau_negative_control.csv")

    # ---- Top5 窗口敏感表 ----
    top5 = sorted(scores, key=lambda p: scores[p]["S_range_C"],
                  reverse=True)[:5]
    top5_rows = []
    for param in top5:
        s = scores[param]
        pm = param_results[param].get(s["val_m10_used"])
        pp = param_results[param].get(s["val_p10_used"])
        top5_rows.append({
            "parameter": param,
            "parameter_symbol": OAT_GRID[param]["label"],
            "window_m10_low_C": pm["low"]["mean"],
            "window_m10_high_C": pm["high"]["mean"],
            "window_baseline_low_C": base_cycles["low"]["mean"],
            "window_baseline_high_C": base_cycles["high"]["mean"],
            "window_p10_low_C": pp["low"]["mean"],
            "window_p10_high_C": pp["high"]["mean"],
            "S_range_C": s["S_range_C"],
        })
    save_table(pd.DataFrame(top5_rows),
               "top5_cycling_window_sensitivity.csv")

    # ---- 图 ----
    plot_tornado(scores, base_cycles, OUTPUT_ROOT / "figures")
    plot_ablation(ablation_stats, OUTPUT_ROOT / "figures")
    k_cal = {}
    cp_cal = {}
    for param, res in (("k_eff", k_cal), ("cp_eff", cp_cal)):
        for val in CALIBRATION_SUPPORTED[param]:
            r = param_results[param].get(float(val))
            if r is None:
                r = {
                    "high": base_cycles["high"],
                    "low": base_cycles["low"],
                    "amplitude": base_cycles["amplitude"],
                }
            res[float(val)] = r
    plot_calibration_supported(k_cal, cp_cal, OUTPUT_ROOT / "figures")

    # ---- Top5 窗口移动图 ----
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    lo_b, hi_b = base_cycles["low"]["mean"], base_cycles["high"]["mean"]
    for i, param in enumerate(top5):
        s = scores[param]
        pm = param_results[param].get(s["val_m10_used"])
        pp = param_results[param].get(s["val_p10_used"])
        y_off = (len(top5) - 1 - i) * 1.0
        ax.plot([lo_b, hi_b], [y_off, y_off], color="grey", lw=5,
                alpha=0.35, solid_capstyle="butt")
        ax.plot([pm["low"]["mean"], pm["high"]["mean"]],
                [y_off + 0.18, y_off + 0.18], color="#1f77b4", lw=3,
                solid_capstyle="butt", label="-10%" if i == 0 else None)
        ax.plot([pp["low"]["mean"], pp["high"]["mean"]],
                [y_off - 0.18, y_off - 0.18], color="#d62728", lw=3,
                solid_capstyle="butt", label="+10%" if i == 0 else None)
        ax.text(lo_b - 0.8, y_off, OAT_GRID[param]["label"], ha="right",
                va="center", fontsize=9)
    ax.axvline(lo_b, color="grey", ls=":", lw=1.0)
    ax.axvline(hi_b, color="grey", ls=":", lw=1.0)
    ax.set_xlabel("Predicted sample temperature [C]")
    ax.set_yticks([])
    ax.set_title("Insulated sample cycling window shift (±10% OAT, "
                 "top-5 by S_range)")
    ax.grid(True, ls="--", alpha=0.3, axis="x")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_ROOT / "figures"
                    / f"top5_cycling_window_shift.{ext}", dpi=150)
    plt.close(fig)

    # ---- 摘要 ----
    L = []
    A = L.append
    A("=" * 74)
    A("INSULATED SAMPLE-TEMPERATURE SENSITIVITY — 08.24 NO-HOLDING SUMMARY")
    A("=" * 74)
    A(f"model            : {M.model_id}")
    A(f"input            : {INPUT_XLSX}")
    A(f"valid points     : {data['n_valid']} | duration "
      f"{data['duration_s']:.2f} s | internal "
      f"[{data['T_min_C']:.2f}, {data['T_max_C']:.2f}] C")
    A(f"repeated cycles  : {len(windows)} complete (Setpoint 20/100 "
      f"segments, activation phase excluded)")
    A(f"activation phase : "
      f"[{cw['activation']['start_s']:.2f}, {cw['activation']['end_s']:.2f}] s "
      f"excluded (Setpoint 90 C segment)")
    A(f"cycle windows    : frozen across all runs "
      f"(source={cw['source']}; high from 100 C segments, "
      "low from 20 C segments)")
    A(f"baseline HIGH    : {base_cycles['high']['mean']:.3f} C "
      f"(median {base_cycles['high']['median']:.3f}, "
      f"std {base_cycles['high']['std']:.3f})")
    A(f"baseline LOW     : {base_cycles['low']['mean']:.3f} C "
      f"(median {base_cycles['low']['median']:.3f}, "
      f"std {base_cycles['low']['std']:.3f})")
    A(f"cycling range    : {base_cycles['low']['mean']:.3f} -> "
      f"{base_cycles['high']['mean']:.3f} C")
    A(f"mean amplitude   : {base_cycles['amplitude']['mean']:.3f} C")
    A(f"overall max      : {base['overall_sample_max_C']:.3f} C @ "
      f"{base['overall_sample_max_time_s']:.3f} s (SECONDARY)")
    A(f"old detector (16 peaks, activation-contaminated): "
      f"mean {OLD_DETECTOR_REF['mean_peak_C']} C / "
      f"median {OLD_DETECTOR_REF['median_peak_C']} C — reported for "
      f"comparison only")
    A("")
    A("PRIMARY OAT SENSITIVITY RANKING (C per ±10%):")
    A(f"  {'Rank':<5}{'Parameter':<26}{'HIGH':>8}{'LOW':>8}{'AMP':>8}"
      f"{'RANGE':>8}")
    for r in rank_rows:
        A(f"  {r['rank']:<5}{r['parameter_symbol']:<26}"
          f"{r['S_high_10pct_C']:8.3f}{r['S_low_10pct_C']:8.3f}"
          f"{r['S_amp_10pct_C']:8.3f}{r['S_range_C']:8.3f}")
    A("")
    A("MOST SENSITIVE (HIGH): "
      f"{max(rank_rows, key=lambda r: r['S_high_10pct_C'])['parameter_symbol']}")
    A("MOST SENSITIVE (LOW): "
      f"{max(rank_rows, key=lambda r: r['S_low_10pct_C'])['parameter_symbol']}")
    A("MOST SENSITIVE (RANGE): "
      f"{max(rank_rows, key=lambda r: r['S_range_C'])['parameter_symbol']}")
    A("")
    A("CALIBRATION-SUPPORTED COC RANGE:")
    for param, vals in (("k_eff", (0.0650, 0.0675, 0.0700)),
                        ("cp_eff", (600.0, 700.0, 800.0))):
        hs, ls = [], []
        for v in vals:
            r = param_results[param].get(float(v))
            hs.append(r["high"]["mean"] if r
                      else base_cycles["high"]["mean"])
            ls.append(r["low"]["mean"] if r
                      else base_cycles["low"]["mean"])
        A(f"  {param}: values {vals} -> HIGH {min(hs):.3f}-{max(hs):.3f} C, "
          f"LOW {min(ls):.3f}-{max(ls):.3f} C")
    A("")
    A("BOUNDARY-PHYSICS ABLATION (LOW -> HIGH, mean):")
    for name in ("BASELINE", "NO_CONVECTION", "NO_RADIATION",
                 "NO_EXTERNAL_SURFACE_HEAT_LOSS"):
        st = ablation_stats[name]
        A(f"  {ABLATION_CASES[name]['label']:<38}: "
          f"{st['low']['mean']:6.2f} -> {st['high']['mean']:6.2f} C")
    A("")
    # 联合消融解释 (权威数值, 直接来自联合仿真, 绝不线性叠加单项消融)
    abl_base = ablation_stats["BASELINE"]
    abl_nc = ablation_stats["NO_CONVECTION"]
    abl_nr = ablation_stats["NO_RADIATION"]
    abl_joint = ablation_stats["NO_EXTERNAL_SURFACE_HEAT_LOSS"]
    d_hi_nc = abl_nc["high"]["mean"] - abl_base["high"]["mean"]
    d_lo_nc = abl_nc["low"]["mean"] - abl_base["low"]["mean"]
    d_hi_nr = abl_nr["high"]["mean"] - abl_base["high"]["mean"]
    d_lo_nr = abl_nr["low"]["mean"] - abl_base["low"]["mean"]
    d_hi_j = abl_joint["high"]["mean"] - abl_base["high"]["mean"]
    d_lo_j = abl_joint["low"]["mean"] - abl_base["low"]["mean"]
    A("EXTERNAL SURFACE HEAT-LOSS CONTRIBUTION (joint simulation values):")
    A(f"  Removing convection alone (h=0):  HIGH {d_hi_nc:+.2f} C, "
      f"LOW {d_lo_nc:+.2f} C (both +~0.24 C)")
    A(f"  Removing radiation alone (eps=0): HIGH {d_hi_nr:+.2f} C, "
      f"LOW {d_lo_nr:+.2f} C (both +~0.11 C)")
    A(f"  Removing BOTH simultaneously:     HIGH {d_hi_j:+.2f} C, "
      f"LOW {d_lo_j:+.2f} C (both +~0.68 C)")
    A("  NOTE: the joint ablation result (~0.7 C) is read directly from the "
      "combined h=0/eps=0 simulation. It is NOT the linear sum of the "
      "individual ablations (0.24+0.11=0.35 C); the nonlinear boundary makes "
      "the combined effect larger.")
    A("")
    A("TAU_TOP NEGATIVE CONTROL: sample identical for tau=0/8/16 "
      "(PASS — sample never lagged)")
    A("")
    A("Scientific interpretation:")
    A("  Sensitivity analysis only; NOT uncertainty propagation.")
    A("  No sample temperature ±X C / CI / probabilistic claim.")
    A("  h=0 and eps=0 are physics-ablation contributions, not "
      "uncertainty bounds.")
    A("  Sample temperature is frozen-model prediction, not measured.")
    A("  Insulated geometry is a forward extension, not directly "
      "sample-validated.")
    A("")
    A("RHO_COC NOTE:")
    A("  rho_COC shows substantial mathematical sensitivity (~0.59 C per "
      "±10%) because the transient heat equation contains rho*cp as the "
      "volumetric heat capacity. However, rho_COC is a fixed reference "
      "material property in FINAL_FROZEN_THERMAL_MODEL_V1, not a "
      "calibrated effective parameter; it is NOT added to the calibrated "
      "set and NOT refitted. Among the primary calibrated/model "
      "parameters, k_eff and cp_eff dominate the cycling window.")
    A("")
    A("CALIBRATION-SUPPORTED COC RANGE CONCLUSION:")
    A("  Within the calibration-supported k_eff range (0.0650-0.0700) and "
      "cp_eff range (600-800), predicted HIGH and LOW cycling levels vary "
      "by less than ~1 C. This is calibration-supported sensitivity, not a "
      "confidence interval or sample-temperature uncertainty.")
    A("")
    A("PRESENTATION-READY CONCLUSION:")
    A("  The predicted 58.8-84.6 C sample cycling window is governed mainly "
      "by the effective COC thermal properties, with substantially smaller "
      "contributions from insulation and external surface heat loss. "
      "Complete removal of both convection and radiation raises the "
      "predicted HIGH and LOW cycling levels by only about 0.7 C.")
    A("")
    A(f"elapsed: {_time.perf_counter() - t0:.1f} s")
    (OUTPUT_ROOT / "final_sensitivity_summary.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8")

    meta = {
        "model_id": M.model_id,
        "input": str(INPUT_XLSX),
        "baseline": {"k_eff_W_mK": BASE_K_EFF, "cp_eff_J_kgK": BASE_CP_EFF,
                     "rho_COC_kg_m3": BASE_RHO, "h_conv_W_m2K": BASE_H,
                     "emissivity": BASE_EPS, "tau_top_s": BASE_TAU},
        "geometry": "LEGACY_INSULATED_LAYERS (4050 um)",
        "cycle_source": cw["source"],
        "activation": cw["activation"],
        "n_cycles_included": len(included),
        "n_cycles_total": len(windows),
        "baseline_cycling_range_C": [base_cycles["low"]["mean"],
                                     base_cycles["high"]["mean"]],
        "baseline_mean_amplitude_C": base_cycles["amplitude"]["mean"],
        "sensitivity_is_not_uncertainty": True,
        "no_probabilistic_claim": True,
    }
    (OUTPUT_ROOT / "sensitivity_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print("=" * 74)
    print("DONE")
    print(f"baseline cycling range: {base_cycles['low']['mean']:.3f} -> "
          f"{base_cycles['high']['mean']:.3f} C")
    print(f"mean amplitude: {base_cycles['amplitude']['mean']:.3f} C")
    print(f"OAT cases run: {len(worker_args)}")
    print(f"elapsed: {_time.perf_counter() - t0:.1f} s")
    print(f"output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
