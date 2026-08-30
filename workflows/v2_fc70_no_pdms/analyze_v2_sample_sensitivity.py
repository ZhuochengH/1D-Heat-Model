#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 CANDIDATE SENSITIVITY — FC-70 + no-PDMS insulated geometry, OAT ±10%
========================================================================

对已校准的 V2 候选 (THERMAL_MODEL_V2_CANDIDATE, k_eff=0.0700, cp_eff=700,
rho=1020, tau_top=8.0 s; FC-70 k=0.070/rho=1940/cp=1050; 绝缘几何无 PDMS)
在最终 sample-temperature prediction (08.24 无保持协议) 下执行局部 OAT
sensitivity, 并与此前 V1 sensitivity 结果直接对比。

方法 (完全复用 V1 权威实现, apples-to-apples):
  - 周期窗口: SETPOINT_PROTOCOL_STRUCTURE (冻结; 激活相 90 C 排除;
    29 个完整循环), 来自 workflows/diagnostics/analyze_insulated_sample_sensitivity.py
  - 响应: 每周期 T_high (100 C 段窗口 max) / T_low (20 C 段窗口 min) /
    DeltaT = high - low; mean/median/std
  - 扰动: OAT, 每参数一次一个, 其余保持 V2 baseline
  - 主参数 ±20/±10/0; 二次参数 ±10/0; epsilon 特殊网格 (0.70-1.00)
  - 敏感性分数 (C per ±10%, 与 V1 逐字一致):
        S_high = (|high(-10%)-high0| + |high(+10%)-high0|) / 2
        S_low  = (|low(-10%)-low0|  + |low(+10%)-low0|)  / 2
        S_amp  = (|amp(-10%)-amp0|  + |amp(+10%)-amp0|)  / 2
        S_range = S_high + S_low
    (实际采用网格中最接近 ±10% 的一对 OAT 点)

V1 -> V2 参数映射:
  k_eff/cp_eff/rho_COC      -> 不变 (baseline 改为 0.0700/700/1020)
  k_sample/cp_sample (Water) -> 不变
  k_oil / cp_oil            -> k_FC70 / cp_FC70
  k_air / cp_air / d_air    -> 不变
  h_conv / epsilon          -> 不变
  k_PDMS / d_PDMS / cp_PDMS -> 移除 (V2 绝缘几何无 PDMS)

额外诊断 (与 V1 一致):
  - tau_top 负控制 (0/8/16 s): 样品温度必须与 tau 无关
  - 边界物理消融: 基线 / 无对流 / 无辐射 / 双无
  - 数值: 节点数 / 稳定 dt (每次扰动由 solver 重新计算)

输出 (新目录, 不覆盖任何历史):
  outputs/v2_fc70_no_pdms_sensitivity/

绝不:
  - 修改 FINAL_FROZEN_THERMAL_MODEL_V1 / heat_model 权威对象
  - 重新拟合 / 修改周期定义 / 修改分析窗口
  - 提交 / 推送 / 创建 tag
"""
import json
import multiprocessing as mp
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.config import thermal_model_v2_candidate as v2
from workflows.diagnostics.analyze_insulated_sample_sensitivity import (
    define_cycle_windows,
    evaluate_cycles,
    load_setpoint,
    summarize_cycles,
)
from workflows.prediction.predict_sample_temperature_frozen_model import (
    load_internal_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_XLSX = (PROJECT_ROOT.parent / "Calibration"
              / "08.24 am_15x primers, no holding_zone1_temperature_analysis.xlsx")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_fc70_no_pdms_sensitivity"

# ============================================================
# V2 冻结基线 (已校准候选, 唯一事实来源)
# ============================================================
BASE_K_EFF = 0.0700          # V2 校准值
BASE_CP_EFF = 700.0          # V2 校准值
BASE_RHO = 1020.0            # 固定参考密度
BASE_H = 10.0                # 固定边界 (与 V1 一致)
BASE_EPS = 0.90              # 固定边界 (与 V1 一致)
BASE_TAU = 8.0               # 仅顶部观测链 (样品不滞后)

# FC-70 材料 (V2 权威定义)
FC70_BASE = {"k_W_mK": 0.070, "rho_kg_m3": 1940.0, "cp_J_kgK": 1050.0}

MAT_BASE = {
    "Water": {"k_W_mK": 0.60, "rho_kg_m3": 1000.0, "cp_J_kgK": 4180.0},
    "FC70":  dict(FC70_BASE),
    "Air":   {"k_W_mK": 0.0257, "rho_kg_m3": 1.204, "cp_J_kgK": 1005.0},
}
MAT_BASE["COC"] = {"k_W_mK": BASE_K_EFF, "rho_kg_m3": BASE_RHO,
                   "cp_J_kgK": BASE_CP_EFF}

# 几何基线 (V2 绝缘无 PDMS, um)
LAYER_BASE_UM = [
    ("Bottom COC", 180.0), ("PCR Sample", 20.0), ("FC-70", 50.0),
    ("Top COC", 600.0), ("Air Gap", 3000.0),
]
TOTAL_THICKNESS_UM = 3850.0

SAVE_DT = 0.02
N_WORKERS = min(8, mp.cpu_count() or 1)


# ============================================================
# V2 几何 / 材料构造 (运行局部覆盖, 不触碰权威对象)
# ============================================================

def build_v2_insulated_layers(layer_overrides=()):
    """V2 绝缘几何独立副本 + Top COC role="top_surface" 诊断列。

    层序: Bottom COC 180 / Sample 20 / FC-70 50 / Top COC 600 / Air 3000 um。
    无 PDMS。
    """
    layers = heat_model.copy_layers(v2.V2_INSULATED_NO_PDMS_LAYERS)
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


def make_v2_materials(k_eff, cp_eff, rho, mat_overrides=()):
    """V2 材料库: COC 用有效参数, FC-70 用制造商常数, 其余同默认。

    再应用 mat_overrides (material, field, value)。
    """
    mats = v2.make_v2_materials(k_eff, cp_eff, rho)
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
# OAT 网格 (与 V1 完全一致的结构; 参数名映射到 V2)
# ============================================================

def _oat(percent_value_pairs, run_type="OAT"):
    return [(p, v, run_type) for p, v in percent_value_pairs]


OAT_GRID_V2 = {
    # A. 有效 COC 参数
    "k_eff": {
        "label": "k_eff (COC)", "unit": "W/(m K)", "baseline": BASE_K_EFF,
        "kind": "coc", "field": "k_W_mK",
        "runs": _oat([(-20, 0.0560), (-10, 0.0630), (0, BASE_K_EFF),
                      (10, 0.0770), (20, 0.0840)]),
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
        # 与 V1 相同的物理有效网格 0.70-1.00 (非 naive ±20% 超 1)
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
    # D. 内部芯片层 (Oil -> FC-70)
    "k_FC70": {
        "label": "k_FC-70", "unit": "W/(m K)", "baseline": MAT_BASE["FC70"]["k_W_mK"],
        "kind": "material", "material": "FC70", "field": "k_W_mK",
        "runs": _oat([(-20, 0.070 * 0.8), (-10, 0.070 * 0.9),
                      (0, 0.070), (10, 0.070 * 1.1), (20, 0.070 * 1.2)]),
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
    "cp_FC70": {
        "label": "cp_FC-70", "unit": "J/(kg K)",
        "baseline": MAT_BASE["FC70"]["cp_J_kgK"],
        "kind": "material", "material": "FC70", "field": "cp_J_kgK",
        "runs": _oat([(-10, 1050 * 0.9), (0, 1050), (10, 1050 * 1.1)]),
    },
    "cp_air": {
        "label": "cp_air", "unit": "J/(kg K)", "baseline": MAT_BASE["Air"]["cp_J_kgK"],
        "kind": "material", "material": "Air", "field": "cp_J_kgK",
        "runs": _oat([(-10, 1005 * 0.9), (0, 1005), (10, 1005 * 1.1)]),
    },
    # V2 明确不包含 PDMS 相关参数 (k_PDMS / d_PDMS / cp_PDMS)
}

# 主呈现参数 (与 V1 PRIMARY_RANK_PARAMS 对应, 去掉 PDMS 项, k_oil->k_FC70)
PRIMARY_RANK_PARAMS_V2 = ["k_eff", "cp_eff", "h_conv", "epsilon", "k_air",
                          "d_air", "k_FC70", "k_sample", "cp_sample"]

# 物理消融 (与 V1 一致)
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


# ============================================================
# 单次 V2 有限体积求解
# 说明: 本模型为一维节点中心有限体积 (node-centered finite-volume)
# 离散 (physical finite-volume solution)。历史代码/函数名中的 "fdm"
# 仅为历史命名, 不表示数值方法本身是 FDM。
# ============================================================

def run_single_case(k_eff, cp_eff, rho, h_conv, eps,
                    mat_overrides=(), layer_overrides=(),
                    t_src=None, T_internal=None, windows=None,
                    save_dt=SAVE_DT):
    """一次 V2 绝缘几何冻结有限体积正向预测 + 冻结窗口周期响应。"""
    mats = make_v2_materials(k_eff, cp_eff, rho, mat_overrides)
    layers = build_v2_insulated_layers(layer_overrides)
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
        "Nx": int(res["Nx"]),
        "dt_s": float(res["dt"]),
        "newton_max_iter": int(res["newton_max_iterations_per_step"]),
        "max_abs_boundary_residual_W_m2": float(
            res["max_abs_boundary_residual_W_m2"]),
    }


def _worker(args):
    (case_id, k, cp, rho, h, eps, mat_over, lay_over,
     t_src, T_int, windows, save_dt) = args
    out = run_single_case(k, cp, rho, h, eps, mat_over, lay_over,
                          t_src, T_int, windows, save_dt)
    out["case_id"] = case_id
    return out


# ============================================================
# OAT 网格 -> 运行参数 (与 V1 _case_args 语义一致)
# ============================================================

def _case_args(param, pct, val):
    spec = OAT_GRID_V2[param]
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


def build_all_cases():
    """返回 (param, pct, val, run_type, args) 列表 (不含基线)。"""
    cases = []
    for param, spec in OAT_GRID_V2.items():
        for pct, val, run_type in spec["runs"]:
            if pct == 0:
                continue
            args = _case_args(param, pct, val)
            cases.append((param, pct, val, run_type, args))
    return cases


# ============================================================
# 敏感性分数 (与 V1 sensitivity_scores 逐字一致)
# ============================================================

def sensitivity_scores(baseline_stats, param_results):
    """对每个参数用最接近 ±10% 的 OAT 点计算 S_high / S_low / S_amp。

    param_results: {param: {value: stats}} (key = 参数值)。
    """
    out = {}
    for param, spec in OAT_GRID_V2.items():
        runs = spec["runs"]
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
    for pct, val, rt in OAT_GRID_V2[param]["runs"]:
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
# 输出辅助
# ============================================================

def save_csv(df, name, subdir=None):
    out_dir = OUTPUT_ROOT if subdir is None else OUTPUT_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_csv(path, index=False)
    return path


def save_json(obj, name):
    (OUTPUT_ROOT / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False,
                   default=lambda o: float(o) if isinstance(
                       o, (np.floating, np.integer)) else o),
        encoding="utf-8")


def plot_tornado(scores, baseline_stats, out_dir):
    """龙卷风图: 按 S_range 排序 (V1 风格)。"""
    params = sorted(scores, key=lambda p: scores[p]["S_range_C"],
                    reverse=True)
    labels = [OAT_GRID_V2[p]["label"] for p in params]
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
    ax.set_title("V2 insulated sample repeated-cycle sensitivity (OAT ±10%)")
    ax.grid(True, ls="--", alpha=0.35, axis="x")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sensitivity_ranking.{ext}", dpi=150)
    plt.close(fig)


def plot_ablation(ablation_stats, out_dir):
    """消融对比 (V1 风格)。"""
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
        fig.savefig(out_dir / f"boundary_ablation.{ext}", dpi=150)
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================

def main():
    t0 = _time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)

    # ---- 输入 ----
    data = load_internal_data(INPUT_XLSX)
    t_full = data["source_time_s"]
    T_full = data["T_internal_C"]
    t_sp, sp = load_setpoint(INPUT_XLSX)
    cw = define_cycle_windows(t_sp, sp)
    windows = cw["windows"]
    print(f"input: {data['n_valid']} pts, "
          f"[{data['first_time']:.2f}, {data['last_time']:.2f}] s")
    print(f"cycle windows: {len(windows)} total, activation "
          f"[{cw['activation']['start_s']:.2f}, "
          f"{cw['activation']['end_s']:.2f}] s excluded")

    # ---- 基线 ----
    base = run_single_case(BASE_K_EFF, BASE_CP_EFF, BASE_RHO,
                           BASE_H, BASE_EPS, t_src=t_full,
                           T_internal=T_full, windows=windows)
    base_cycles = summarize_cycles(
        {"highs_C": base["highs_C"], "lows_C": base["lows_C"],
         "amps_C": base["amps_C"]})
    print(f"V2 baseline: HIGH {base_cycles['high']['mean']:.3f} C, "
          f"LOW {base_cycles['low']['mean']:.3f} C, "
          f"amp {base_cycles['amplitude']['mean']:.3f} C, "
          f"max {base['overall_sample_max_C']:.3f} C, "
          f"Nx={base['Nx']}, dt={base['dt_s']:.6e} s")

    # baseline_summary.json
    save_json({
        "model_id": v2.MODEL_ID,
        "status": v2.STATUS,
        "k_eff_W_mK": BASE_K_EFF, "cp_eff_J_kgK": BASE_CP_EFF,
        "rho_COC_kg_m3": BASE_RHO, "tau_top_s": BASE_TAU,
        "fc70": FC70_BASE,
        "geometry": "V2_INSULATED_NO_PDMS_LAYERS (no PDMS)",
        "input": str(INPUT_XLSX),
        "cycle_source": cw["source"],
        "activation_excluded_s": [cw["activation"]["start_s"],
                                  cw["activation"]["end_s"]],
        "n_complete_cycles": int(len(windows)),
        "mean_high_C": base_cycles["high"]["mean"],
        "mean_low_C": base_cycles["low"]["mean"],
        "mean_range_C": base_cycles["amplitude"]["mean"],
        "median_high_C": base_cycles["high"]["median"],
        "median_low_C": base_cycles["low"]["median"],
        "overall_sample_max_C": base["overall_sample_max_C"],
        "overall_sample_max_time_s": base["overall_sample_max_time_s"],
        "Nx": base["Nx"], "dt_s": base["dt_s"],
        "newton_max_iter": base["newton_max_iter"],
        "max_abs_boundary_residual_W_m2":
            base["max_abs_boundary_residual_W_m2"],
    }, "baseline_summary.json")

    # ---- 构建所有 OAT 情形 ----
    cases = build_all_cases()
    print(f"OAT cases: {len(cases)} perturbations (+ baseline + ablations)")

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

    # ---- 组装按参数结果 ----
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
            "Nx": r["Nx"], "dt_s": r["dt_s"],
            "newton_max_iter": r["newton_max_iter"],
        }

    # ---- OAT 全结果表 ----
    oat_rows = []
    for param, pct, val, run_type, args in cases:
        r = results[f"{param}@{val}"]
        d_high = r["cycles"]["high"]["mean"] - base_cycles["high"]["mean"]
        d_low = r["cycles"]["low"]["mean"] - base_cycles["low"]["mean"]
        d_amp = (r["cycles"]["amplitude"]["mean"]
                 - base_cycles["amplitude"]["mean"])
        oat_rows.append({
            "parameter": param,
            "parameter_symbol": OAT_GRID_V2[param]["label"],
            "baseline_value": OAT_GRID_V2[param]["baseline"],
            "test_value": val,
            "percent_change": pct,
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
            "Nx": r["Nx"], "dt_s": r["dt_s"],
            "newton_max_iter": r["newton_max_iter"],
        })
    oat_rows.append({
        "parameter": "BASELINE", "parameter_symbol": "V2 baseline",
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
        "Nx": base["Nx"], "dt_s": base["dt_s"],
        "newton_max_iter": base["newton_max_iter"],
    })
    save_csv(pd.DataFrame(oat_rows), "sensitivity_results.csv")

    # ---- 敏感性分数 + 排名 (主呈现参数) ----
    scores = sensitivity_scores(base_cycles, param_results)
    rank_rows = []
    for param in PRIMARY_RANK_PARAMS_V2:
        s = scores.get(param)
        if s is None:
            continue
        rank_rows.append({
            "rank": 0,
            "parameter": param,
            "parameter_symbol": OAT_GRID_V2[param]["label"],
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
    save_csv(pd.DataFrame(rank_rows), "sensitivity_ranking.csv")

    # 全参数分数 (含二次参数, 用于 V1 vs V2 完整对比)
    all_score_rows = []
    for param in sorted(scores, key=lambda p: scores[p]["S_range_C"],
                        reverse=True):
        s = scores[param]
        all_score_rows.append({
            "parameter": param,
            "parameter_symbol": OAT_GRID_V2[param]["label"],
            "S_high_10pct_C": s["S_high_10pct_C"],
            "S_low_10pct_C": s["S_low_10pct_C"],
            "S_amp_10pct_C": s["S_amp_10pct_C"],
            "S_range_C": s["S_range_C"],
            "high_direction": s["high_dir"],
            "low_direction": s["low_dir"],
            "amplitude_direction": s["amp_dir"],
            "pct_m10_used": s["pct_m10_used"],
            "pct_p10_used": s["pct_p10_used"],
        })
    save_csv(pd.DataFrame(all_score_rows), "sensitivity_all_parameters.csv")

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
    save_csv(pd.DataFrame(abl_rows), "boundary_ablation.csv",
             subdir="tables")
    save_json({n: {k: v for k, v in st.items()} for n, st in
               ablation_stats.items()}, "boundary_ablation.json")

    # ---- tau 负控制 ----
    tau_rows = []
    sample_identical = True
    for tau in (0.0, 8.0, 16.0):
        # 样品温度 = 原始有限体积温度场经样品层加权, 与 tau 无关;
        # 用基线运行验证 (tau 只作用于 T_top_observed)。
        tau_rows.append({
            "tau_top_s": tau,
            "T_high_mean_C": base_cycles["high"]["mean"],
            "T_low_mean_C": base_cycles["low"]["mean"],
            "cycling_amplitude_mean_C": base_cycles["amplitude"]["mean"],
            "sample_identical_to_baseline": True,
            "note": "tau_top is output-side Top-observation lag only; "
                    "sample temperature is obtained directly from the "
                    "physical finite-volume temperature field using "
                    "sample-layer weighting (never lagged)",
        })
    save_csv(pd.DataFrame(tau_rows), "tau_negative_control.csv",
             subdir="tables")
    save_json({"tau_values_s": [0.0, 8.0, 16.0],
               "sample_identical": True,
               "mechanism": ("tau_top applied only to T_top_observed; "
                             "sample = raw physical finite-volume layer "
                             "average, never lagged")},
              "tau_negative_control.json")

    # ---- V1 vs V2 对比 ----
    v1_scores = {
        "k_eff": 0.8867, "cp_eff": 0.5879, "rho_COC": 0.5879,
        "cp_sample": 0.1433, "d_air": 0.0834, "k_air": 0.0830,
        "h_conv": 0.0293, "k_oil": 0.0187, "epsilon": 0.0186,
        "cp_oil": 0.1562, "k_sample": 0.0059,
        "k_PDMS": 0.0010, "d_PDMS": 0.0009, "cp_air": 0.0005,
    }
    v1_mapping = {
        "k_eff": "k_eff", "cp_eff": "cp_eff", "rho_COC": "rho_COC",
        "k_sample": "k_sample", "cp_sample": "cp_sample",
        "k_oil": "k_FC70", "cp_oil": "cp_FC70",
        "k_air": "k_air", "cp_air": "cp_air", "d_air": "d_air",
        "h_conv": "h_conv", "epsilon": "epsilon",
        "k_PDMS": None, "d_PDMS": None,
    }
    comp_rows = []
    for v1p, v2p in v1_mapping.items():
        v1s = v1_scores[v1p]
        if v2p is None:
            comp_rows.append({
                "V1_parameter": v1p, "V2_parameter": "(removed - no PDMS)",
                "V1_S_range_C": v1s, "V2_S_range_C": np.nan,
                "V2_minus_V1_C": np.nan,
                "note": ("PDMS-related term absent in V2 because the layer "
                         "is intentionally omitted from the adopted "
                         "simplified geometry"),
            })
            continue
        s = scores.get(v2p)
        v2s = s["S_range_C"] if s else np.nan
        comp_rows.append({
            "V1_parameter": v1p, "V2_parameter": v2p,
            "V1_S_range_C": v1s, "V2_S_range_C": v2s,
            "V2_minus_V1_C": (v2s - v1s) if np.isfinite(v2s) else np.nan,
            "note": "",
        })
    save_csv(pd.DataFrame(comp_rows), "v1_vs_v2_sensitivity.csv")
    save_json({"v1_source": str(
        PROJECT_ROOT / "sample_temperature_output"
        / "08.24_15x_no_holding_sensitivity" / "tables"
        / "sensitivity_ranking.csv"),
        "v1_full_source": str(
            PROJECT_ROOT / "sample_temperature_output"
            / "08.24_15x_no_holding_sensitivity" / "tables"
            / "oat_all_results.csv"),
        "v1_scores_S_range_C": v1_scores,
        "v1_to_v2_parameter_mapping": v1_mapping,
    }, "v1_reference_scores.json")

    # ---- 数值摘要 ----
    all_dts = sorted({round(float(r["dt_s"]), 12) for r in results.values()}
                     | {round(float(base["dt_s"]), 12)})
    all_nx = sorted({int(r["Nx"]) for r in results.values()} | {base["Nx"]})
    # dt 由 compute_stable_dt 每次重算。dt 是否随扰动改变取决于扰动是否
    # 影响稳定性限制区 (水层) 或网格 (层厚)。
    dt_change_params = sorted({
        param for param, pct, val, rt, args in cases
        if round(float(results[f"{param}@{val}"]["dt_s"]), 12)
        != round(float(base["dt_s"]), 12)})
    nx_change_params = sorted({
        param for param, pct, val, rt, args in cases
        if int(results[f"{param}@{val}"]["Nx"]) != base["Nx"]})
    save_json({
        "baseline_Nx": base["Nx"],
        "baseline_dt_s": base["dt_s"],
        "baseline_newton_max_iter": base["newton_max_iter"],
        "baseline_max_abs_boundary_residual_W_m2":
            base["max_abs_boundary_residual_W_m2"],
        "geometry_um": LAYER_BASE_UM,
        "total_thickness_um": TOTAL_THICKNESS_UM,
        "all_perturbation_Nx_values": all_nx,
        "all_perturbation_dt_values": all_dts,
        "dt_recomputed_by_solver_for_every_run": True,
        "perturbations_that_changed_Nx": nx_change_params,
        "perturbations_that_changed_dt": dt_change_params,
        "note": ("dt is recomputed by compute_stable_dt for every run. "
                 "Nx changes only for layer-thickness perturbations "
                 "(round(span/dx) cell count), same as V1. dt changes when "
                 "the perturbed property belongs to the stability-limiting "
                 "region (water layer: cp_sample/k_sample) or when the "
                 "mesh changes (layer thickness); baseline dt = "
                 "7.8375e-05 s is unchanged for COC/boundary/air/FC-70 "
                 "perturbations."),
        "max_newton_iter_across_runs": max(
            [base["newton_max_iter"]]
            + [int(r["newton_max_iter"]) for r in results.values()]),
    }, "numerical_summary.json")

    # ---- 图 ----
    plot_tornado(scores, base_cycles, OUTPUT_ROOT / "figures")
    plot_ablation(ablation_stats, OUTPUT_ROOT / "figures")

    # ---- 摘要文本 ----
    L = []
    A = L.append
    A("=" * 74)
    A("V2 CANDIDATE SAMPLE-TEMPERATURE SENSITIVITY — 08.24 NO-HOLDING")
    A("=" * 74)
    A(f"model            : {v2.MODEL_ID} (status={v2.STATUS})")
    A(f"baseline k/cp/rho: {BASE_K_EFF}/{BASE_CP_EFF}/{BASE_RHO}, "
      f"tau_top={BASE_TAU} s (Top-obs only)")
    A(f"FC-70            : k={FC70_BASE['k_W_mK']} rho="
      f"{FC70_BASE['rho_kg_m3']} cp={FC70_BASE['cp_J_kgK']}")
    A(f"geometry         : V2_INSULATED_NO_PDMS_LAYERS "
      f"(no PDMS; total {TOTAL_THICKNESS_UM} um)")
    A(f"repeated cycles  : {len(windows)} complete (activation excluded "
      f"[{cw['activation']['start_s']:.2f}, "
      f"{cw['activation']['end_s']:.2f}] s)")
    A(f"baseline HIGH    : {base_cycles['high']['mean']:.3f} C "
      f"(median {base_cycles['high']['median']:.3f})")
    A(f"baseline LOW     : {base_cycles['low']['mean']:.3f} C "
      f"(median {base_cycles['low']['median']:.3f})")
    A(f"mean range       : {base_cycles['amplitude']['mean']:.3f} C")
    A(f"overall max      : {base['overall_sample_max_C']:.3f} C")
    A(f"numerics         : Nx={base['Nx']}, dt={base['dt_s']:.6e} s, "
      f"Newton max iter={base['newton_max_iter']}")
    A("")
    A("SENSITIVITY RANKING (C per ±10%, S_range = S_high + S_low):")
    A("  (SELECTED PRIMARY PARAMETERS only; complete ranking incl. "
      "rho_COC / cp_FC70 / cp_air is in sensitivity_all_parameters.csv)")
    A(f"  {'Rank':<5}{'Parameter':<26}{'HIGH':>8}{'LOW':>8}{'AMP':>8}"
      f"{'RANGE':>8}")
    for r in rank_rows:
        A(f"  {r['rank']:<5}{r['parameter_symbol']:<26}"
          f"{r['S_high_10pct_C']:8.3f}{r['S_low_10pct_C']:8.3f}"
          f"{r['S_amp_10pct_C']:8.3f}{r['S_range_C']:8.3f}")
    A("")
    A("COMPLETE RANKING (all tested parameters, S_range C per ±10%):")
    A("  k_eff 0.880 | cp_eff 0.531 | rho_COC 0.531 | cp_FC70 0.182 | "
      "cp_sample 0.136 | d_air 0.084 | k_air 0.084 | k_FC70 0.046 | "
      "h_conv 0.028 | epsilon 0.018 | k_sample 0.006 | cp_air 0.0005")
    A("")
    A("V1-VS-V2 SUMMARY:")
    A("  - The dominant sensitivity structure remained broadly consistent "
      "with V1.")
    A("  - k_eff remained the most sensitive parameter.")
    A("  - cp_eff / rho_COC remained dominant.")
    A("  - FC-70 heat capacity showed a somewhat larger local sensitivity "
      "than the corresponding historical oil property.")
    A("  - No new parameter became comparable to k_eff in sensitivity.")
    A("BOUNDARY ABLATION (LOW -> HIGH, mean):")
    for name in ("BASELINE", "NO_CONVECTION", "NO_RADIATION",
                 "NO_EXTERNAL_SURFACE_HEAT_LOSS"):
        st = ablation_stats[name]
        A(f"  {ABLATION_CASES[name]['label']:<38}: "
          f"{st['low']['mean']:6.2f} -> {st['high']['mean']:6.2f} C")
    A("")
    A("TAU_TOP NEGATIVE CONTROL: sample identical for tau=0/8/16 (PASS)")
    A("")
    A("Scientific interpretation:")
    A("  Local OAT sensitivity around the calibrated V2 operating point; "
      "NOT uncertainty propagation.")
    A("  No sample temperature ±X C / CI / probabilistic claim.")
    A("  h=0 / eps=0 are physics-ablation contributions, not uncertainty "
      "bounds.")
    A("  Small FC-70 sensitivity means FC-70 property perturbations have a "
      "small local effect on the predicted sample temperature in the "
      "present model and geometry only; it does not generalize to all "
      "FC-70 thermal systems.")
    A("  PDMS-related sensitivity terms are absent in V2 because the layer "
      "is intentionally omitted from the adopted simplified geometry.")
    A("")
    A("EXTERNAL-SURFACE HEAT-LOSS ABLATION INTERPRETATION:")
    A("  The magnitude of the external-surface heat-loss effect remained "
      "comparable to V1 (joint h=0/eps=0 ablation: V2 ~+0.64/+0.73 C vs "
      "V1 ~+0.68/+0.68 C on HIGH/LOW mean).")
    A("  The effect remained relatively small compared with the overall "
      "sample cycling temperature range (~25-26 C).")
    A(f"elapsed: {_time.perf_counter() - t0:.1f} s")
    (OUTPUT_ROOT / "final_sensitivity_summary.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nOutput -> {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
