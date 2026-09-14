#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 CALIBRATION-PROPAGATED SAMPLE-PREDICTION SENSITIVITY (END-TO-END)
====================================================================

科学问题:
  "If one of the assumptions fixed before calibration were different, and the
   model were recalibrated accordingly, how much would the final
   sample-temperature prediction change?"

管线 (C 类 analysis, 在既有 A/B 之上):
  fixed assumption'
      -> (读取既有 29-run 校准敏感性输出: k_eff', cp_eff', tau_top')
      -> 用 (fixed assumption' + k_eff' + cp_eff') 做最终绝缘样品预测
      -> HIGH / LOW / range / max 变化
  tau_top 仅是输出侧 Top-观测滞后, 记录但绝不施加到样品温度。

关键规则 (TOTAL / END-TO-END):
  下游预测必须同时保留 ① 扰动后的 fixed input 和 ② 重校准的 k_eff'/cp_eff'。
  绝不允许校准时扰动、预测时把 fixed input 重置回 baseline —— 那样只测到
  通过拟合参数的间接效应, 不是完整的 end-to-end 后果。

分解 (对每个固定输入):
  DIRECT-ONLY:    扰动 fixed input + baseline k_eff/cp_eff (0.0700/700)
  INDIRECT-ONLY:  baseline fixed input + 重校准 k_eff'/cp_eff'
  TOTAL:          扰动 fixed input + 重校准 k_eff'/cp_eff' (主结果)
  interaction     = TOTAL - DIRECT - INDIRECT (近似分解残差, 不强制可加)

数据来源:
  - 校准敏感性: outputs/v2_calibration_sensitivity/runs/*.json (29 run, 只读)
    + build_runs() 的 PerturbConfig (扰动值权威定义, 已验证与 run JSON 一致)
  - 预测协议:   完全复用 workflows/v2_fc70_no_pdms/analyze_v2_sample_sensitivity.py
    的权威实现 (同一数据集/周期窗口/激活排除/HIGH/LOW 定义/样品加权/求解器)
  - frozen 敏感性对比: outputs/v2_fc70_no_pdms_sensitivity/*.csv (只读)

输出 (新目录, 不覆盖任何历史):
  outputs/v2_calibration_propagated_prediction_sensitivity/

绝不:
  - 修改 FINAL_FROZEN_THERMAL_MODEL_V2 / heat_model / v2_candidate 权威对象
  - 重新运行 29 个校准搜索 (已有结果完整)
  - 提交 / 推送 / 移动 tag
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

# ---- 既有分析 A: 校准敏感性 (只读) ----
from workflows.diagnostics.analyze_v2_calibration_sensitivity import (
    build_runs, baseline_config as cal_baseline_config,
    make_materials as cal_make_materials, make_layers as cal_make_layers,
    RHO_COC as CAL_RHO, H_CONV as CAL_H, EPS as CAL_EPS, F_VIEW as CAL_F,
    FC70_K as CAL_FC70_K, FC70_RHO as CAL_FC70_RHO, FC70_CP as CAL_FC70_CP,
    WATER_K as CAL_WATER_K, WATER_RHO as CAL_WATER_RHO,
    WATER_CP as CAL_WATER_CP, GEOM as CAL_GEOM,
)

# ---- 既有分析 B: frozen 样品预测敏感性 (协议权威, 只读复用) ----
from workflows.v2_fc70_no_pdms.analyze_v2_sample_sensitivity import (
    run_single_case, BASE_K_EFF, BASE_CP_EFF, BASE_RHO, BASE_H, BASE_EPS,
    BASE_TAU, SAVE_DT, INPUT_XLSX, N_WORKERS,
)
from workflows.diagnostics.analyze_insulated_sample_sensitivity import (
    define_cycle_windows, load_setpoint, summarize_cycles,
)
from workflows.prediction.predict_sample_temperature_frozen_model import (
    load_internal_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAL_RUNS_DIR = (PROJECT_ROOT / "outputs" / "v2_calibration_sensitivity"
                / "runs")
FROZEN_SENS_DIR = PROJECT_ROOT / "outputs" / "v2_fc70_no_pdms_sensitivity"
OUTPUT_ROOT = (PROJECT_ROOT / "outputs"
               / "v2_calibration_propagated_prediction_sensitivity")

# 校准敏感性 <-> 样品预测 之间的几何映射 (µm)
# 校准几何 = 裸顶 4 层; 样品预测几何 = 绝缘 5 层 (多 Air Gap 3000)。
# 同名层厚扰动按名字映射。
CAL_TO_PRED_LAYER = {"Bottom COC": "Bottom COC", "PCR Sample": "PCR Sample",
                     "FC-70": "FC-70", "Top COC": "Top COC"}


# ============================================================
# 1. 读取既有校准敏感性结果 + 扰动配置
# ============================================================

def load_calibration_runs():
    """读取 29 个 run JSON, 并用 build_runs() 的 PerturbConfig 补全扰动值。"""
    configs = {r.tag: r for r in build_runs()}
    configs["baseline"] = cal_baseline_config()
    runs = {}
    for p in sorted(CAL_RUNS_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        tag = d["tag"]
        cfg = configs.get(tag)
        if cfg is None:
            raise RuntimeError(f"run {tag} 无对应 PerturbConfig")
        if not (cfg.param == d["param"] and abs(cfg.pct - d["pct"]) < 1e-9
                and abs(cfg.value - d["value"]) < 1e-6):
            raise RuntimeError(f"run {tag} 与 PerturbConfig 不一致")
        runs[tag] = {"json": d, "cfg": cfg}
    missing = set(configs) - set(runs)
    if missing:
        raise RuntimeError(f"缺少 run JSON: {sorted(missing)}")
    return runs


def audit_source(runs):
    return {
        "n_runs": len(runs),
        "runs_dir": str(CAL_RUNS_DIR),
        "tags": sorted(runs),
        "baseline_reproduction": runs["baseline"]["json"].get(
            "metrics", {}).get("RMSE_C"),
        "consistency_check": "PerturbConfig vs run JSON: PASS (param/pct/value)",
    }


# ============================================================
# 2. 校准扰动 -> 样品预测覆盖项 (关键映射)
# ============================================================

def pred_overrides_from_cfg(cfg):
    """PerturbConfig -> 样品预测的 (k_eff, cp_eff, rho, h, eps, F, mat_over,
    lay_over)。

    只覆盖被该 case 扰动的输入; 其余保持预测基线。
    k_eff/cp_eff/rho: COC 有效参数 (k_eff/cp_eff 来自重校准, rho 来自扰动)。
    """
    k = BASE_K_EFF          # 由调用者用重校准值覆盖
    cp = BASE_CP_EFF
    rho = cfg.rho_COC       # rho_COC 扰动在此体现 (非 None 字段是扰动值)
    h = cfg.h_conv
    eps = cfg.epsilon
    F = cfg.F_view
    mat_over = []
    if cfg.fc70_k is not None or cfg.fc70_rho is not None \
            or cfg.fc70_cp is not None:
        mat_over.append(("FC70", "k_W_mK",
                         cfg.fc70_k if cfg.fc70_k is not None
                         else CAL_FC70_K))
        mat_over.append(("FC70", "rho_kg_m3",
                         cfg.fc70_rho if cfg.fc70_rho is not None
                         else CAL_FC70_RHO))
        mat_over.append(("FC70", "cp_J_kgK",
                         cfg.fc70_cp if cfg.fc70_cp is not None
                         else CAL_FC70_CP))
    if cfg.water_k is not None or cfg.water_rho is not None \
            or cfg.water_cp is not None:
        mat_over.append(("Water", "k_W_mK",
                         cfg.water_k if cfg.water_k is not None
                         else CAL_WATER_K))
        mat_over.append(("Water", "rho_kg_m3",
                         cfg.water_rho if cfg.water_rho is not None
                         else CAL_WATER_RHO))
        mat_over.append(("Water", "cp_J_kgK",
                         cfg.water_cp if cfg.water_cp is not None
                         else CAL_WATER_CP))
    lay_over = []
    for cal_name, pred_name in CAL_TO_PRED_LAYER.items():
        val = {"Bottom COC": cfg.thick_bottom_um,
               "PCR Sample": cfg.thick_sample_um,
               "FC-70": cfg.thick_fc70_um,
               "Top COC": cfg.thick_top_um}[cal_name]
        if val is not None:
            lay_over.append((pred_name, "thickness", float(val) * 1e-6))
    return k, cp, rho, h, eps, F, mat_over, lay_over


# ============================================================
# 3. 下游样品预测 (带 F_view 支持)
# ============================================================

def run_pred(k_eff, cp_eff, rho, h_conv, eps, F_view,
             mat_overrides=(), layer_overrides=(),
             t_src=None, T_internal=None, windows=None,
             save_dt=SAVE_DT):
    """与 analyze_v2_sample_sensitivity.run_single_case 相同的物理,
    但透传 view_factor (run_single_case 不暴露该参数)。

    tau_top 绝不施加: 样品 = 原始有限体积温度场样品层加权。
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
            cp_J_kgK=float(val) if field == "cp_J_kgK" else m.cp_J_kgK)
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
    T_init = float(T_internal[0])
    res = cr.run_convection_radiation_fdm(
        time_s=t_src, bottom_temperature_C=T_internal, materials=mats,
        layers=layers, T_air_C=T_init, T_surroundings_C=T_init,
        h_conv_W_m2K=h_conv, emissivity=eps, view_factor=F_view,
        save_dt=save_dt, T_initial_C=T_init)
    t_abs = float(t_src[0]) + res["t_array"]
    T_sample = res["T_sample_arr"]
    cyc = evaluate_cycles_safe(t_abs, T_sample, windows)
    st = summarize_cycles(cyc)
    return {
        "high_mean_C": st["high"]["mean"], "high_median_C": st["high"]["median"],
        "low_mean_C": st["low"]["mean"], "low_median_C": st["low"]["median"],
        "amp_mean_C": st["amplitude"]["mean"],
        "overall_max_C": float(np.max(T_sample)),
        "n_cycles_high": len(cyc["highs_C"]),
        "Nx": int(res["Nx"]), "dt_s": float(res["dt"]),
        "newton_max_iter": int(res["newton_max_iterations_per_step"]),
        "max_abs_boundary_residual_W_m2": float(
            res["max_abs_boundary_residual_W_m2"]),
    }


def evaluate_cycles_safe(t_abs, T_sample, windows):
    """复用权威 evaluate_cycles (从 analyze_insulated_sample_sensitivity 导入)。"""
    from workflows.diagnostics.analyze_insulated_sample_sensitivity import (
        evaluate_cycles)
    return evaluate_cycles(t_abs, T_sample, windows)


def _worker(args):
    (case_id, mode, k, cp, rho, h, eps, F, mat_over, lay_over,
     t_src, T_int, windows) = args
    out = run_pred(k, cp, rho, h, eps, F, mat_over, lay_over,
                   t_src, T_int, windows)
    out["case_id"] = case_id
    out["mode"] = mode
    return out


# ============================================================
# 4. 构建 end-to-end / direct / indirect 三类 case
# ============================================================

def build_all_propagation_cases(runs):
    """返回 dict: case_id -> dict(mode, k, cp, rho, h, eps, F, mat, lay)。

    TOTAL (end-to-end):    扰动 fixed input + 重校准 k'/cp'
    DIRECT:                扰动 fixed input + baseline k/cp
    INDIRECT:              baseline fixed input + 重校准 k'/cp'
    """
    cases = {}
    base_cfg = runs["baseline"]["cfg"]
    base_k = runs["baseline"]["json"]["best"]["k_eff_W_mK"]
    base_cp = runs["baseline"]["json"]["best"]["cp_eff_J_kgK"]
    for tag, item in runs.items():
        cfg = item["cfg"]
        best = item["json"]["best"]
        if tag == "baseline":
            continue
        k_eff_p, cp_eff_p = float(best["k_eff_W_mK"]), float(
            best["cp_eff_J_kgK"])
        # ---- TOTAL ----
        k0, cp0, rho, h, eps, F, mat, lay = pred_overrides_from_cfg(cfg)
        cases[f"TOTAL::{tag}"] = dict(
            mode="TOTAL", k=k_eff_p, cp=cp_eff_p, rho=rho, h=h, eps=eps,
            F=F, mat=mat, lay=lay)
        # ---- DIRECT (扰动 fixed input + baseline k/cp) ----
        cases[f"DIRECT::{tag}"] = dict(
            mode="DIRECT", k=BASE_K_EFF, cp=BASE_CP_EFF, rho=rho, h=h,
            eps=eps, F=F, mat=mat, lay=lay)
        # ---- INDIRECT (baseline fixed input + 重校准 k'/cp') ----
        cases[f"INDIRECT::{tag}"] = dict(
            mode="INDIRECT", k=k_eff_p, cp=cp_eff_p, rho=base_cfg.rho_COC,
            h=base_cfg.h_conv, eps=base_cfg.epsilon, F=base_cfg.F_view,
            mat=[], lay=[])
    # baseline (baseline config + baseline 校准 k/cp)
    cases["BASELINE"] = dict(
        mode="BASELINE", k=base_k, cp=base_cp, rho=base_cfg.rho_COC,
        h=base_cfg.h_conv, eps=base_cfg.epsilon, F=base_cfg.F_view,
        mat=[], lay=[])
    return cases


# ============================================================
# 5. 主流程
# ============================================================

def main():
    t0 = _time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)

    # ---- 读取既有校准结果 ----
    runs = load_calibration_runs()
    audit = audit_source(runs)
    print(f"source audit: {audit['n_runs']} calibration runs recovered")

    # ---- 输入数据 + 周期窗口 (与 frozen 分析同一权威协议) ----
    data = load_internal_data(INPUT_XLSX)
    t_full, T_full = data["source_time_s"], data["T_internal_C"]
    t_sp, sp = load_setpoint(INPUT_XLSX)
    cw = define_cycle_windows(t_sp, sp)
    windows = cw["windows"]
    print(f"input: {data['n_valid']} pts; cycle windows: {len(windows)}, "
          f"activation [{cw['activation']['start_s']:.2f}, "
          f"{cw['activation']['end_s']:.2f}] s excluded")

    # ---- 构建所有传播 case ----
    cases = build_all_propagation_cases(runs)
    n_total = len(cases)
    print(f"propagation cases: {n_total} "
          f"(1 baseline + {sum(1 for c in cases.values() if c['mode'] == 'TOTAL')} TOTAL "
          f"+ DIRECT + INDIRECT)")

    worker_args = []
    for case_id, c in cases.items():
        worker_args.append((
            case_id, c["mode"], c["k"], c["cp"], c["rho"], c["h"], c["eps"],
            c["F"], tuple(c["mat"]), tuple(c["lay"]),
            t_full, T_full, windows))

    results = {}
    with mp.Pool(N_WORKERS) as pool:
        for out in pool.imap_unordered(_worker, worker_args):
            results[out["case_id"]] = out

    base = results["BASELINE"]
    print(f"baseline: HIGH {base['high_mean_C']:.4f} LOW {base['low_mean_C']:.4f} "
          f"range {base['amp_mean_C']:.4f} max {base['overall_max_C']:.4f} "
          f"({results['BASELINE']['n_cycles_high']} cycles)")

    # ---- 数值检查 ----
    num_fail = []
    for cid, r in results.items():
        ok = (r["n_cycles_high"] == base["n_cycles_high"]
              and r["newton_max_iter"] <= 20
              and r["max_abs_boundary_residual_W_m2"] < 1e-6)
        if not ok:
            num_fail.append(cid)
    numerical_checks = {
        "n_forward_runs": n_total,
        "baseline_Nx": base["Nx"], "baseline_dt_s": base["dt_s"],
        "all_Nx_values": sorted({r["Nx"] for r in results.values()}),
        "dt_recomputed_per_run": True,
        "max_newton_iter": max(r["newton_max_iter"] for r in results.values()),
        "max_abs_boundary_residual_W_m2": max(
            r["max_abs_boundary_residual_W_m2"] for r in results.values()),
        "n_cycles_high_consistent": base["n_cycles_high"],
        "failed_cases": num_fail,
        "tau_top_applied_to_sample": False,
        "tau_top_mechanism": ("sample = raw finite-volume field sample-layer "
                              "average; tau_top only affects Top observation "
                              "and is recorded but not applied"),
        "no_pdms": True,
        "fc70_used": True,
    }
    save_json(numerical_checks, "numerical_checks.json")
    if num_fail:
        print(f"WARNING: numerical check failures: {num_fail}")

    # ---- 每参数汇总 ----
    rows = []
    for tag, item in runs.items():
        if tag == "baseline":
            continue
        cfg, best = item["cfg"], item["json"]["best"]
        tot = results[f"TOTAL::{tag}"]
        dire = results.get(f"DIRECT::{tag}")
        ind = results.get(f"INDIRECT::{tag}")
        rows.append({
            "tag": tag, "param": cfg.param, "pct": cfg.pct,
            "perturbed_value": cfg.value,
            "recal_k_eff": float(best["k_eff_W_mK"]),
            "recal_cp_eff": float(best["cp_eff_J_kgK"]),
            "recal_tau_top_s": float(best["tau_top_s"]),
            "calib_RMSE_C": float(item["json"]["metrics"]["RMSE_C"]),
            "calib_boundary_warnings": "|".join(
                item["json"]["boundary_warnings"]),
            # TOTAL (end-to-end)
            "TOTAL_high_mean_C": tot["high_mean_C"],
            "TOTAL_low_mean_C": tot["low_mean_C"],
            "TOTAL_amp_mean_C": tot["amp_mean_C"],
            "TOTAL_overall_max_C": tot["overall_max_C"],
            "d_TOTAL_high_C": tot["high_mean_C"] - base["high_mean_C"],
            "d_TOTAL_low_C": tot["low_mean_C"] - base["low_mean_C"],
            "d_TOTAL_amp_C": tot["amp_mean_C"] - base["amp_mean_C"],
            "d_TOTAL_max_C": tot["overall_max_C"] - base["overall_max_C"],
            # DIRECT
            "DIRECT_high_mean_C": dire["high_mean_C"] if dire else np.nan,
            "DIRECT_low_mean_C": dire["low_mean_C"] if dire else np.nan,
            "d_DIRECT_high_C": (dire["high_mean_C"]
                                - base["high_mean_C"]) if dire else np.nan,
            "d_DIRECT_low_C": (dire["low_mean_C"]
                               - base["low_mean_C"]) if dire else np.nan,
            "d_DIRECT_amp_C": (dire["amp_mean_C"]
                               - base["amp_mean_C"]) if dire else np.nan,
            # INDIRECT
            "INDIRECT_high_mean_C": ind["high_mean_C"] if ind else np.nan,
            "INDIRECT_low_mean_C": ind["low_mean_C"] if ind else np.nan,
            "d_INDIRECT_high_C": (ind["high_mean_C"]
                                  - base["high_mean_C"]) if ind else np.nan,
            "d_INDIRECT_low_C": (ind["low_mean_C"]
                                 - base["low_mean_C"]) if ind else np.nan,
            "d_INDIRECT_amp_C": (ind["amp_mean_C"]
                                 - base["amp_mean_C"]) if ind else np.nan,
            # interaction residual (TOTAL - DIRECT - INDIRECT)
            "interaction_high_C": ((tot["high_mean_C"] - base["high_mean_C"])
                                   - (dire["high_mean_C"]
                                      - base["high_mean_C"])
                                   - (ind["high_mean_C"]
                                      - base["high_mean_C"])) if (dire and
                                                                  ind)
                                  else np.nan,
            "interaction_low_C": ((tot["low_mean_C"] - base["low_mean_C"])
                                  - (dire["low_mean_C"] - base["low_mean_C"])
                                  - (ind["low_mean_C"] - base["low_mean_C"])
                                  ) if (dire and ind) else np.nan,
            "Nx": tot["Nx"], "dt_s": tot["dt_s"],
            "newton_max_iter": tot["newton_max_iter"],
        })
    df_all = pd.DataFrame(rows)
    save_csv(df_all, "propagated_prediction_all.csv")

    # ---- 敏感性分数 ----
    # 规则: 对每个 param 取 (最负 pct, 最正 pct) 两点, S = (|Δm|+|Δp|)/2,
    # 再按实际扰动幅度归一化为 "每 ±10%" (epsilon 为 ±11.1%, F_view 为 -10/-20)。
    # 单侧参数 (F_view: 仅负侧) 用负侧平均 |Δ| 再除以 0.5(单侧半幅) × ... -> 统一
    # 定义: S = mean(|Δ|)/max(|pct_m|,|pct_p|) * 10  (即每 ±10% 等效幅度)。
    sym_pairs = {}
    for tag in df_all["tag"]:
        cfg = runs[tag]["cfg"]
        if cfg.pct == 0:
            continue
        sym_pairs.setdefault(cfg.param, {})[cfg.pct] = tag

    score_rows = []
    ident_rows = []
    for param, pair in sym_pairs.items():
        item_m = next((t for p, t in sorted(pair.items()) if p < 0), None)
        item_p = next((t for p, t in sorted(pair.items()) if p > 0), None)
        if item_m is None or item_p is None:
            continue
        rm = df_all[df_all["tag"] == item_m].iloc[0]
        rp = df_all[df_all["tag"] == item_p].iloc[0]
        cfg_m = runs[item_m]["cfg"]
        cfg_p = runs[item_p]["cfg"]
        # 幅度归一: 与 frozen 分析的 S_range_C (每 ±10%) 可比
        amp = max(abs(cfg_m.pct), abs(cfg_p.pct))
        scale = 10.0 / amp
        is_bounded = cfg_m.param in ("epsilon", "F_view")
        is_onesided = (cfg_m.param == "F_view")  # 单侧: 只有负侧扰动
        if is_onesided:
            # 单侧: 用负侧 |Δ| 线性外推到 ±10% 等效幅度 (mean|Δ| × 10/|pct|)
            s_high = abs(rm["d_TOTAL_high_C"]) * scale
            s_low = abs(rm["d_TOTAL_low_C"]) * scale
            s_amp = abs(rm["d_TOTAL_amp_C"]) * scale
            s_max = abs(rm["d_TOTAL_max_C"]) * scale
        else:
            s_high = (abs(rm["d_TOTAL_high_C"])
                      + abs(rp["d_TOTAL_high_C"])) / 2 * scale
            s_low = (abs(rm["d_TOTAL_low_C"])
                     + abs(rp["d_TOTAL_low_C"])) / 2 * scale
            s_amp = (abs(rm["d_TOTAL_amp_C"])
                     + abs(rp["d_TOTAL_amp_C"])) / 2 * scale
            s_max = (abs(rm["d_TOTAL_max_C"])
                     + abs(rp["d_TOTAL_max_C"])) / 2 * scale
        score_rows.append({
            "param": param,
            "S_HIGH_end_to_end_C": s_high,
            "S_LOW_end_to_end_C": s_low,
            "S_AMP_end_to_end_C": s_amp,
            "S_MAX_end_to_end_C": s_max,
            "S_combined_end_to_end_C": s_high + s_low,
            "perturbation_pair": f"{item_m} / {item_p}",
            "pct_pair": f"{cfg_m.pct:+.1f}/{cfg_p.pct:+.1f}",
            "bounded_or_onesided": ("onesided" if is_onesided
                                    else ("bounded" if is_bounded else "no")),
            "score_convention": (
                "one-sided: |Δ| from available side, scaled to ±10% equivalent"
                if is_onesided else
                "mean |Δ| of both sides, scaled to ±10% equivalent"),
        })
        # identifiability flags (来自校准阶段)
        # 注意: df_all 中 calib_boundary_warnings 已是 "CP_MIN" 形式的字符串
        warn = (f"{rm['calib_boundary_warnings']}|"
                f"{rp['calib_boundary_warnings']}")
        cp_extreme = (max(abs(rm["recal_cp_eff"] - BASE_CP_EFF),
                          abs(rp["recal_cp_eff"] - BASE_CP_EFF))
                      > 0.5 * BASE_CP_EFF)
        ident_rows.append({
            "param": param,
            "boundary_warning_calib": warn,
            "cp_eff_shift_gt_50pct": bool(cp_extreme),
            "identifiability_flag": (
                "BOUNDARY_LIMITED_CALIBRATION" if ("CP_MAX" in warn
                                                   or "CP_MIN" in warn)
                else ("EXTREME_PARAMETER_SHIFT" if cp_extreme else "NONE")),
        })
    df_score = pd.DataFrame(score_rows).sort_values(
        "S_combined_end_to_end_C", ascending=False).reset_index(drop=True)
    df_score.insert(0, "rank", df_score.index + 1)
    save_csv(df_score, "end_to_end_sensitivity_ranking.csv")
    save_csv(pd.DataFrame(ident_rows), "identifiability_flags.csv")

    # ---- direct-only / indirect-only 独立敏感性 CSV (同一幅度归一) ----
    def _pair_sides(pair):
        item_m = next((t for p, t in sorted(pair.items()) if p < 0), None)
        item_p = next((t for p, t in sorted(pair.items()) if p > 0), None)
        return item_m, item_p

    def _scale_for(runs, param, pair):
        item_m = next((t for p, t in sorted(pair.items()) if p < 0), None)
        item_p = next((t for p, t in sorted(pair.items()) if p > 0), None)
        cm = runs[item_m]["cfg"]
        cp_ = runs[item_p]["cfg"]
        amp = max(abs(cm.pct), abs(cp_.pct))
        return 10.0 / amp

    dir_rows = []
    for param, pair in sym_pairs.items():
        item_m, item_p = _pair_sides(pair)
        if item_m is None or item_p is None:
            continue
        rm = df_all[df_all["tag"] == item_m].iloc[0]
        rp = df_all[df_all["tag"] == item_p].iloc[0]
        scale = _scale_for(runs, param, pair)
        if runs[item_m]["cfg"].param == "F_view":
            s_high = abs(rm["d_DIRECT_high_C"]) * scale
            s_low = abs(rm["d_DIRECT_low_C"]) * scale
            s_amp = abs(rm["d_DIRECT_amp_C"]) * scale
        else:
            s_high = (abs(rm["d_DIRECT_high_C"])
                      + abs(rp["d_DIRECT_high_C"])) / 2 * scale
            s_low = (abs(rm["d_DIRECT_low_C"])
                     + abs(rp["d_DIRECT_low_C"])) / 2 * scale
            s_amp = (abs(rm["d_DIRECT_amp_C"])
                     + abs(rp["d_DIRECT_amp_C"])) / 2 * scale
        dir_rows.append({
            "param": param,
            "S_HIGH_direct_only_C": s_high,
            "S_LOW_direct_only_C": s_low,
            "S_AMP_direct_only_C": s_amp,
        })
    save_csv(pd.DataFrame(dir_rows), "direct_only_sensitivity.csv")

    ind_rows = []
    for param, pair in sym_pairs.items():
        item_m, item_p = _pair_sides(pair)
        if item_m is None or item_p is None:
            continue
        rm = df_all[df_all["tag"] == item_m].iloc[0]
        rp = df_all[df_all["tag"] == item_p].iloc[0]
        scale = _scale_for(runs, param, pair)
        if runs[item_m]["cfg"].param == "F_view":
            s_high = abs(rm["d_INDIRECT_high_C"]) * scale
            s_low = abs(rm["d_INDIRECT_low_C"]) * scale
            s_amp = abs(rm["d_INDIRECT_amp_C"]) * scale
        else:
            s_high = (abs(rm["d_INDIRECT_high_C"])
                      + abs(rp["d_INDIRECT_high_C"])) / 2 * scale
            s_low = (abs(rm["d_INDIRECT_low_C"])
                     + abs(rp["d_INDIRECT_low_C"])) / 2 * scale
            s_amp = (abs(rm["d_INDIRECT_amp_C"])
                     + abs(rp["d_INDIRECT_amp_C"])) / 2 * scale
        ind_rows.append({
            "param": param,
            "S_HIGH_indirect_only_C": s_high,
            "S_LOW_indirect_only_C": s_low,
            "S_AMP_indirect_only_C": s_amp,
        })
    save_csv(pd.DataFrame(ind_rows), "indirect_recalibration_sensitivity.csv")

    # ---- 分解表 ----
    dec_rows = []
    for _, r in df_score.iterrows():
        param = r["param"]
        d_row = next((x for x in dir_rows if x["param"] == param), None)
        i_row = next((x for x in ind_rows if x["param"] == param), None)
        if d_row is None or i_row is None:
            continue
        dec_rows.append({
            "param": param,
            "S_HIGH_direct_C": d_row["S_HIGH_direct_only_C"],
            "S_HIGH_indirect_C": i_row["S_HIGH_indirect_only_C"],
            "S_HIGH_total_C": r["S_HIGH_end_to_end_C"],
            "S_HIGH_interaction_C": r["S_HIGH_end_to_end_C"]
            - d_row["S_HIGH_direct_only_C"]
            - i_row["S_HIGH_indirect_only_C"],
            "S_LOW_direct_C": d_row["S_LOW_direct_only_C"],
            "S_LOW_indirect_C": i_row["S_LOW_indirect_only_C"],
            "S_LOW_total_C": r["S_LOW_end_to_end_C"],
            "S_LOW_interaction_C": r["S_LOW_end_to_end_C"]
            - d_row["S_LOW_direct_only_C"]
            - i_row["S_LOW_indirect_only_C"],
        })
    save_csv(pd.DataFrame(dec_rows), "effect_decomposition.csv")

    # ---- frozen vs propagated 对比 ----
    # frozen 敏感性参数名 -> 校准敏感性参数名
    # (frozen 网格没有几何层厚度扰动, 因此这些行 frozen 值为 NaN, 属预期)
    frozen_all = pd.read_csv(
        FROZEN_SENS_DIR / "sensitivity_all_parameters.csv")
    frozen_map = {"rho_COC": "rho_COC", "h_conv": "h_conv",
                  "epsilon": "epsilon", "k_FC70": "fc70_k",
                  "cp_FC70": "fc70_cp", "k_sample": "water_k",
                  "cp_sample": "water_cp"}
    comp_rows = []
    for frozen_param, cal_param in frozen_map.items():
        fr = frozen_all[frozen_all["parameter"] == frozen_param]
        frozen_s = float(fr["S_range_C"].iloc[0]) if len(fr) else np.nan
        es = df_score[df_score["param"] == cal_param]
        etoe_s = (float(es["S_combined_end_to_end_C"].iloc[0])
                  if len(es) else np.nan)
        comp_rows.append({
            "frozen_parameter": frozen_param,
            "calibration_parameter": cal_param,
            "frozen_direct_S_range_C": frozen_s,
            "end_to_end_S_combined_C": etoe_s,
            "end_to_end_minus_frozen_C": (etoe_s - frozen_s
                                          if np.isfinite(etoe_s)
                                          and np.isfinite(frozen_s)
                                          else np.nan),
            "note": ("frozen: k_eff/cp_eff held at baseline; end-to-end: "
                     "perturbed fixed input + recalibrated k_eff'/cp_eff' "
                     "both propagated"),
        })
    save_csv(pd.DataFrame(comp_rows), "frozen_vs_propagated_sensitivity.csv")

    # ---- 图 ----
    plot_ranking(df_score, OUTPUT_ROOT / "figures")
    plot_frozen_vs_propagated(pd.DataFrame(comp_rows),
                              OUTPUT_ROOT / "figures")
    top_params = df_score.head(5)["param"].tolist()
    plot_decomposition(pd.DataFrame(dec_rows), top_params,
                       OUTPUT_ROOT / "figures")
    plot_rho_coc(df_all, runs, OUTPUT_ROOT / "figures")

    # ---- 输出 audit / baseline / 最终报告 ----
    save_json(audit, "source_calibration_runs_audit.json")
    save_json({
        "mode": "BASELINE (perturbed-free config + recalibrated baseline "
                "k/cp, identical to 0.0700/700)",
        "k_eff_W_mK": base and cases["BASELINE"]["k"],
        "cp_eff_J_kgK": cases["BASELINE"]["cp"],
        "rho_COC_kg_m3": cases["BASELINE"]["rho"],
        "tau_top_s_recorded_not_applied":
            runs["baseline"]["json"]["best"]["tau_top_s"],
        "mean_high_C": base["high_mean_C"],
        "median_high_C": base["high_median_C"],
        "mean_low_C": base["low_mean_C"],
        "median_low_C": base["low_median_C"],
        "mean_range_C": base["amp_mean_C"],
        "overall_sample_max_C": base["overall_max_C"],
        "n_complete_cycles": base["n_cycles_high"],
        "Nx": base["Nx"], "dt_s": base["dt_s"],
        "newton_max_iter": base["newton_max_iter"],
        "max_abs_boundary_residual_W_m2":
            base["max_abs_boundary_residual_W_m2"],
        "input": str(INPUT_XLSX),
        "activation_excluded_s": [cw["activation"]["start_s"],
                                  cw["activation"]["end_s"]],
    }, "baseline_prediction.json")

    txt = build_report(df_score, df_all, pd.DataFrame(dec_rows),
                       pd.DataFrame(comp_rows),
                       pd.DataFrame(ident_rows), base, runs, num_checks)
    (OUTPUT_ROOT / "final_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"[done in {_time.perf_counter() - t0:.0f}s] -> {OUTPUT_ROOT}")
    return 0


# ============================================================
# 绘图
# ============================================================

def plot_ranking(df_score, out_dir):
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    y = np.arange(len(df_score))
    ax.barh(y, df_score["S_HIGH_end_to_end_C"], height=0.4,
            color="#d62728", alpha=0.85,
            label="|Δ mean HIGH| (C per ±10%)")
    ax.barh(y + 0.4, df_score["S_LOW_end_to_end_C"], height=0.4,
            color="#1f77b4", alpha=0.85,
            label="|Δ mean LOW| (C per ±10%)")
    ax.set_yticks(y + 0.2)
    labels = [f"{r['param']} ({r['pct_pair']}%)" for _, r in
              df_score.iterrows()]
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("End-to-end sensitivity [C per ±10% fixed-input "
                  "perturbation]")
    ax.set_title("Calibration-propagated sample sensitivity "
                 "(perturbed input + recalibrated k/cp)")
    ax.grid(True, ls="--", alpha=0.35, axis="x")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"end_to_end_sensitivity_ranking.{ext}",
                    dpi=150)
    plt.close(fig)


def plot_frozen_vs_propagated(comp_df, out_dir):
    d = comp_df.dropna(subset=["frozen_direct_S_range_C",
                               "end_to_end_S_combined_C"]).copy()
    d = d[d["frozen_direct_S_range_C"] > 0]
    if d.empty:
        return
    d = d.sort_values("frozen_direct_S_range_C", ascending=False)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - 0.2, d["frozen_direct_S_range_C"], 0.4,
           color="#7f7f7f", alpha=0.9, label="Frozen direct (k/cp fixed)")
    ax.bar(x + 0.2, d["end_to_end_S_combined_C"], 0.4,
           color="#d62728", alpha=0.9,
           label="End-to-end (perturbed input + recalibrated k/cp)")
    ax.set_xticks(x)
    ax.set_xticklabels(d["frozen_parameter"], fontsize=9, rotation=20)
    ax.set_ylabel("Sample-temperature sensitivity [C]")
    ax.set_title("Frozen vs calibration-propagated sample sensitivity")
    ax.grid(True, ls="--", alpha=0.35, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"frozen_vs_propagated_sensitivity.{ext}",
                    dpi=150)
    plt.close(fig)


def plot_decomposition(dec_df, top_params, out_dir):
    d = dec_df[dec_df["param"].isin(top_params)].copy()
    if d.empty:
        return
    d = d.sort_values("S_HIGH_total_C", ascending=False)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - 0.27, d["S_HIGH_direct_C"], 0.26, color="#2ca02c",
           alpha=0.9, label="Direct (perturbed input, baseline k/cp)")
    ax.bar(x, d["S_HIGH_indirect_C"], 0.26, color="#ff7f0e", alpha=0.9,
           label="Indirect (recalibrated k/cp only)")
    ax.bar(x + 0.27, d["S_HIGH_total_C"], 0.26, color="#d62728",
           alpha=0.9, label="Total end-to-end")
    ax.set_xticks(x)
    ax.set_xticklabels(d["param"], fontsize=9, rotation=15)
    ax.set_ylabel("S_HIGH [C]")
    ax.set_title("Direct / indirect / total decomposition (mean HIGH)")
    ax.grid(True, ls="--", alpha=0.35, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"direct_indirect_total.{ext}", dpi=150)
    plt.close(fig)


def plot_rho_coc(df_all, runs, out_dir):
    tags = [t for t in df_all["tag"] if t.startswith("rho_COC")]
    if not tags:
        return
    order = sorted(tags, key=lambda t: runs[t]["cfg"].pct)
    x = np.arange(len(order))
    highs = [float(df_all[df_all["tag"] == t]["TOTAL_high_mean_C"].iloc[0])
             for t in order]
    lows = [float(df_all[df_all["tag"] == t]["TOTAL_low_mean_C"].iloc[0])
            for t in order]
    cps = [runs[t]["json"]["best"]["cp_eff_J_kgK"] for t in order]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 5.0))
    ax1.plot(x, highs, "o-", color="#d62728", label="mean HIGH")
    ax1.plot(x, lows, "o-", color="#1f77b4", label="mean LOW")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{runs[t]['cfg'].pct:+.0f}%" for t in order])
    ax1.set_xlabel("rho_COC perturbation")
    ax1.set_ylabel("Sample temperature [C]")
    ax1.set_title("rho_COC propagation to sample prediction\n"
                  "(identifiability warning: cp' on calibration boundary)")
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend(fontsize=9)
    ax2.plot(x, cps, "s-", color="#ff7f0e")
    ax2.axhline(700.0, color="gray", ls="--", lw=1, label="baseline cp_eff")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{runs[t]['cfg'].pct:+.0f}%" for t in order])
    ax2.set_xlabel("rho_COC perturbation")
    ax2.set_ylabel("recalibrated cp_eff [J/(kg K)]")
    ax2.set_title("Recalibrated cp_eff vs rho_COC")
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"rho_COC_propagation.{ext}", dpi=150)
    plt.close(fig)


# ============================================================
# 输出辅助
# ============================================================

def save_csv(df, name):
    path = OUTPUT_ROOT / name
    df.to_csv(path, index=False)
    return path


def save_json(obj, name):
    (OUTPUT_ROOT / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False,
                   default=lambda o: float(o) if isinstance(
                       o, (np.floating, np.integer)) else o),
        encoding="utf-8")


# ============================================================
# 最终报告
# ============================================================

def build_report(df_score, df_all, dec_df, comp_df, ident_df, base, runs,
                 num_checks):
    L = []
    A = L.append
    A("=" * 74)
    A("V2 CALIBRATION-PROPAGATED SAMPLE-PREDICTION SENSITIVITY REPORT")
    A("=" * 74)
    A(f"model: FINAL_FROZEN_THERMAL_MODEL_V2 (k=0.0700, cp=700, tau=8.0 s; "
      f"FC-70; insulated no-PDMS)")
    A(f"input: {INPUT_XLSX.name}; cycles: {base['n_cycles_high']} complete")
    A(f"baseline: HIGH={base['high_mean_C']:.4f} LOW={base['low_mean_C']:.4f} "
      f"range={base['amp_mean_C']:.4f} max={base['overall_max_C']:.4f} C")
    A("")
    A("1. END-TO-END SENSITIVITY RANKING (S = mean |Δ| across the pair; "
      "bounded params use available pair)")
    for _, r in df_score.iterrows():
        A(f"   {r['param']:<18s} S_HIGH={r['S_HIGH_end_to_end_C']:7.4f}  "
          f"S_LOW={r['S_LOW_end_to_end_C']:7.4f}  "
          f"S_AMP={r['S_AMP_end_to_end_C']:7.4f}  "
          f"combined={r['S_combined_end_to_end_C']:7.4f}  "
          f"pair={r['perturbation_pair']}")
    A("")
    A("2. DIRECT / INDIRECT / TOTAL DECOMPOSITION (S_HIGH / S_LOW)")
    for _, r in dec_df.iterrows():
        A(f"   {r['param']:<18s} HIGH: d={r['S_HIGH_direct_C']:.4f} "
          f"i={r['S_HIGH_indirect_C']:.4f} t={r['S_HIGH_total_C']:.4f} "
          f"(int {r['S_HIGH_interaction_C']:+.4f}) | "
          f"LOW: d={r['S_LOW_direct_C']:.4f} i={r['S_LOW_indirect_C']:.4f} "
          f"t={r['S_LOW_total_C']:.4f} (int {r['S_LOW_interaction_C']:+.4f})")
    A("")
    A("3. FROZEN vs END-TO-END")
    for _, r in comp_df.iterrows():
        A(f"   {r['frozen_parameter']:<16s} frozen={r['frozen_direct_S_range_C']:.4f} "
          f"end-to-end={r['end_to_end_S_combined_C']:.4f} "
          f"diff={r['end_to_end_minus_frozen_C']:+.4f}")
    A("")
    A("4. IDENTIFIABILITY FLAGS")
    for _, r in ident_df.iterrows():
        A(f"   {r['param']:<18s} {r['identifiability_flag']} "
          f"(warn={r['boundary_warning_calib'] or 'none'})")
    A("")
    A("5. NUMERICAL CHECKS")
    A(f"   forward runs={num_checks['n_forward_runs']}, "
      f"Nx values={num_checks['all_Nx_values']}, "
      f"max Newton iter={num_checks['max_newton_iter']}, "
      f"max |boundary residual|={num_checks['max_abs_boundary_residual_W_m2']:.2e}")
    A(f"   tau_top applied to sample: "
      f"{num_checks['tau_top_applied_to_sample']} (recorded only)")
    A(f"   failed cases: {num_checks['failed_cases'] or 'none'}")
    A("")
    A("6. KEY QUESTION")
    A("   Does parameter compensation that stabilizes the Top-COC calibration "
      "also stabilize the final sample-temperature prediction? -> see ranking "
      "vs frozen comparison above.")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    sys.exit(main())
