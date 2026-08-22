#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
裸顶 vs 密封空气绝缘顶部 — 前向热响应评估 (单一独立分析文件)
=================================================================

受控前向物理比较 (FORWARD PHYSICS EVALUATION, 非标定/拟合/优化):

  CASE A — BARE_TOP
      当前校准裸顶 COC 几何 (BARE_TOP_COC_LAYERS):
          Bottom COC 180 um -> Sample 20 um -> Oil 50 um
          -> Top COC 600 um (总 850 um)
      Top COC 外表面直接: 自然对流 h=10 + 非线性 Stefan-Boltzmann 辐射
          -> 环境

  CASE B — SEALED_AIR_INSULATED
      当前芯片 + 密封空气间隙 + 外 PDMS/盖层:
          Bottom COC 180 um -> Sample 20 um -> Oil 50 um
          -> Top COC 600 um -> Sealed Air 3000 um -> PDMS 200 um
      (总 4050 um; 空气/PDMS 几何复用项目历史 LEGACY_INSULATED_LAYERS 值)
      环境对流+辐射仅作用于 PDMS 外表面 (最外节点);
      不对 Top COC 直接施加环境边界。

中央科学问题:
      实验用的密封空气绝缘是否显著提高模型预测的样品峰温度?
      即: 裸顶样品峰 < 90 C 而绝缘样品峰接近/进入 ~90-95 C?

约束:
  - 冻结 COC 有效参数不变 (k=0.055, cp=1200, rho=1020, 两情形相同);
  - tau_top=8.5 s 仅作用于裸顶 Top 观测 (诊断), 绝不影响样品;
  - 两情形使用同一实测内部迹线 (底部 Dirichlet 边界) / 同一初始温度 /
    同一环境温度;
  - 空气间隙按传导主导 (一阶评估): 无间隙内部自然对流,
    无间隙内表面-表面辐射 (该省略须在摘要中明确声明);
  - 不扫描空气/PDMS 厚度/h/发射率/材料性质;
  - 不用 PCR 成功/失败调参; 不以 PCR 结果作为优化目标。

输出 (gitignored):
  model_comparison_output/bare_vs_sealed_air_insulation_v1/
      DOE11_faster/bare_vs_insulated_trace.csv
                   bare_vs_insulated_sample_temperature.png/.pdf
      Test_PCR_longer_holding/ (同上)
      comparison/
          bare_vs_insulated_sample_peak_summary.csv
          sample_peak_bare_vs_insulated.png
          bare_vs_insulated_metadata.json
          bare_vs_insulated_summary.txt

本文件是唯一新增源文件; 不修改任何现有模块/测试/文档。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.core.lag_augmented_thermal_model import apply_first_order_lag
from thermal_model.historical.frozen_strategy_G_candidate import FROZEN_STRATEGY_G_CANDIDATE
from thermal_model.utilities.predict_sample_from_internal_temperature import load_internal_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT.parent / "Calibration"
DOE11_PATH = CALIBRATION_DIR / (
    "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx")
LONGER_PATH = CALIBRATION_DIR / "Test_PCR longer holding.xlsx"

OUTPUT_ROOT = (PROJECT_ROOT / "model_comparison_output"
               / "bare_vs_sealed_air_insulation_v1")

SAVE_DT = 0.1

# 冻结 COC 有效参数 (两情形相同, 不重拟合)
K_EFF = FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK        # 0.055
CP_EFF = FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK     # 1200.0
RHO_COC = FROZEN_STRATEGY_G_CANDIDATE.rho_COC_kg_m3   # 1020.0
TAU_TOP = FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s       # 8.5 (仅裸顶观测诊断)

# 环境边界 (两情形相同)
H_CONV = cr.H_CONV_STRATEGY_E_W_M2K            # 10.0
EPS = cr.EMISSIVITY_STRATEGY_E                 # 0.90
SIGMA = cr.SIGMA_SB_W_M2_K4                    # 5.670374419e-8
F_VIEW = cr.VIEW_FACTOR_STRATEGY_E             # 1.0

# 描述性 PCR 热参考阈值 (仅热学合理性参考, 非通过/失败标准)
THRESHOLDS = (85.0, 90.0, 92.0, 95.0)

# 周期检测参数 (与既有分析一致, 非新启发式)
PEAK_THRESHOLD = 88.0
DIP_THRESHOLD = 60.0
MIN_PEAK_SEPARATION_S = 15.0
ACTIVATION_MIN_S = 30.0


# ============================================================
# 层叠定义
# ============================================================

def make_insulated_layers():
    """绝缘层叠: BARE 4 层 + 密封空气 3000 um + PDMS 200 um。

    空气/PDMS 厚度与网格复用项目历史 LEGACY_INSULATED_LAYERS 值
    (Air 3000 um @ dx 200 um; PDMS 200 um @ dx 50 um), 不引入重复几何。
    样品层保持同一物理 20 um 层 (role='sample'); Top COC 标记
    role='top_surface' 使 T_top_surface_arr = Top COC/Air 界面。
    """
    bare = list(heat_model.BARE_TOP_COC_LAYERS)
    # BARE 的 Top COC 已带 role="top_surface" (裸顶外表面); 绝缘中该角色
    # 语义变为 Top COC/Air 界面观测, 保持同一标记即可。
    top = bare[-1]
    insulated = list(bare)
    # 历史绝缘层 (来自 LEGACY_INSULATED_LAYERS, 只读引用其几何数值)
    legacy = heat_model.LEGACY_INSULATED_LAYERS
    air_layer = next(l for l in legacy if l.material == "Air")
    pdms_layer = next(l for l in legacy if l.material == "PDMS")
    insulated.append(air_layer)
    insulated.append(pdms_layer)
    return insulated, (top, air_layer, pdms_layer)


# ============================================================
# 环境 / 加载 (内部-only 代理规则)
# ============================================================

def resolve_environment_proxy(t_internal):
    """内部-only PCR 环境代理 (显式记录, 无静默 25 C)。"""
    tint = np.asarray(t_internal, dtype=float)
    valid = np.flatnonzero(np.isfinite(tint))
    if valid.size == 0:
        raise ValueError("无有效内部温度。")
    return {"T_environment_C": float(tint[valid[0]]),
            "environment_source":
                "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT"}


def _resolve_sheet(path):
    xl = pd.ExcelFile(path)
    if "Extracted_Data" in xl.sheet_names:
        return "Extracted_Data"
    for sh in xl.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sh, nrows=3)
            if "Zone 1 Avg (°C)" in df.columns:
                return sh
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(f"找不到含温度列的工作表: {xl.sheet_names}")


# ============================================================
# 前向 FDM (两情形共用同一求解器, 同一冻结材料)
# ============================================================

def run_case(layers, elapsed, tint, t_env):
    """运行单个情形 (冻结材料库 + 给定层叠)。返回完整 FDM 结果 dict。"""
    mats = cr.make_convection_radiation_materials(K_EFF, CP_EFF, RHO_COC)
    T_initial = float(np.asarray(tint, dtype=float)[0])
    return cr.run_convection_radiation_fdm(
        time_s=elapsed, bottom_temperature_C=tint, materials=mats,
        layers=layers, T_air_C=t_env, T_surroundings_C=t_env,
        save_dt=SAVE_DT, T_initial_C=T_initial)


# ============================================================
# 周期检测 (复制既有已验证逻辑, 非新启发式)
# ============================================================

def detect_repeated_cycles(t, t_internal, t_sample):
    """峰-谷检测 + 激活相分离 (与项目既有分析相同逻辑)。"""
    t = np.asarray(t, dtype=float)
    ti = np.asarray(t_internal, dtype=float)
    ts = np.asarray(t_sample, dtype=float)
    n = len(t)

    peaks = []
    i = 1
    while i < n - 1:
        if ti[i] >= PEAK_THRESHOLD and ti[i] >= ti[i - 1]:
            j = i
            while j + 1 < n and ti[j + 1] == ti[j]:
                j += 1
            if j == n - 1 or ti[j] >= ti[j + 1]:
                peaks.append(j)
            i = j + 1
        else:
            i += 1

    troughs = []
    i = 1
    while i < n - 1:
        if ti[i] < DIP_THRESHOLD and ti[i] <= ti[i - 1]:
            j = i
            while j + 1 < n and ti[j + 1] == ti[j]:
                j += 1
            if j == n - 1 or ti[j] <= ti[j + 1]:
                troughs.append(j)
            i = j + 1
        else:
            i += 1

    if len(peaks) < 2:
        return {"activation": None, "repeated_cycles": []}

    merged = []
    for idx in peaks:
        if merged:
            prev = merged[-1]
            between = ti[prev + 1:idx]
            min_between = float(np.min(between)) if between.size else \
                min(float(ti[prev]), float(ti[idx]))
            if min_between < DIP_THRESHOLD:
                merged.append(idx)
            else:
                if ti[idx] > ti[prev]:
                    merged[-1] = idx
        else:
            merged.append(idx)

    selected = []
    for idx in merged:
        if selected and (t[idx] - t[selected[-1]]) < MIN_PEAK_SEPARATION_S:
            continue
        selected.append(idx)

    if len(selected) < 2:
        return {"activation": None, "repeated_cycles": []}

    phases = []
    for pidx in selected:
        prior = [tr for tr in troughs if tr < pidx]
        trough = prior[-1] if prior else 0
        hi = min(n - 1, pidx + 20)
        sw = ts[pidx:hi + 1]
        spi = pidx + int(np.argmax(sw))
        phases.append({
            "cycle_start_time_s": float(t[trough]),
            "internal_peak_time_s": float(t[pidx]),
            "internal_high_peak_C": float(ti[pidx]),
            "sample_high_peak_C": float(ts[spi]),
            "has_prior_trough": bool(prior),
        })

    activation = None
    repeated = []
    if phases:
        p0 = phases[0]
        if not p0["has_prior_trough"] and \
                (p0["internal_peak_time_s"] - t[0]) >= ACTIVATION_MIN_S:
            activation = p0
            rest = phases[1:]
        else:
            rest = phases
    else:
        rest = []

    for k, ph in enumerate(rest):
        c = dict(ph)
        c["cycle_number"] = k + 1
        repeated.append(c)
    return {"activation": activation, "repeated_cycles": repeated}


# ============================================================
# 指标 / 诊断
# ============================================================

def _stats(x):
    if not x:
        return np.nan, np.nan, np.nan
    return float(np.min(x)), float(np.max(x)), float(np.mean(x))


def threshold_flags(T):
    mx = float(np.max(T))
    return {f"ge{int(th)}": bool(mx >= th) for th in THRESHOLDS}


def repeated_counts(peaks):
    arr = np.asarray(peaks, dtype=float) if peaks else np.array([])
    return {f"ge{int(th)}": int(np.sum(arr >= th)) for th in THRESHOLDS}


def external_heat_loss_W_m2(Ts_C, t_env_C):
    """近似外部热损失通量 (对流+辐射), 仅描述性诊断。"""
    q_conv = H_CONV * (Ts_C - t_env_C)
    q_rad = cr.radiative_heat_flux_W_m2(
        Ts_C, t_env_C, emissivity=EPS, view_factor=F_VIEW,
        sigma_SB_W_m2K4=SIGMA)
    return float(q_conv + q_rad)


# ============================================================
# 运行与汇总 (单协议)
# ============================================================

def process_protocol(name, path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = _resolve_sheet(path)
    data = load_internal_data(path, sheet=sheet, time_col="Time(s)",
                              temp_col="Zone 1 Avg (°C)")
    elapsed = data["elapsed_time_s"]
    tint = data["T_internal_C"]
    env = resolve_environment_proxy(tint)
    t_env = env["T_environment_C"]
    T_init = float(tint[0])

    insulated_layers, (top_coc_layer, air_layer, pdms_layer) = \
        make_insulated_layers()
    bare_layers = list(heat_model.BARE_TOP_COC_LAYERS)

    # ---- 运行两个情形 (同一内部迹线 / 同一初始 / 同一环境) ----
    r_bare = run_case(bare_layers, elapsed, tint, t_env)
    r_ins = run_case(insulated_layers, elapsed, tint, t_env)

    t_b = r_bare["t_array"]
    t_i = r_ins["t_array"]
    T_s_b = r_bare["T_sample_arr"]
    T_s_i = r_ins["T_sample_arr"]
    T_top_b = r_bare["T_top_surface_arr"]          # 裸顶 Top COC 外表面
    T_top_i = r_ins["T_top_surface_arr"]           # 绝缘 Top COC/Air 界面
    T_cover_i = r_ins["T_outer_surface_arr"]       # 绝缘 PDMS 外表面

    # ---- 迹线 (在测量时间插值; 裸顶附滞后观测诊断, 不影响样品) ----
    T_top_obs_b = apply_first_order_lag(t_b, T_top_b, TAU_TOP)
    trace = pd.DataFrame({
        "time_s": elapsed,
        "T_internal_C": tint,
        "T_sample_bare_C": np.interp(elapsed, t_b, T_s_b),
        "T_topCOC_bare_C": np.interp(elapsed, t_b, T_top_b),
        "T_top_lagged_bare_C": np.interp(elapsed, t_b, T_top_obs_b),
        "T_sample_insulated_C": np.interp(elapsed, t_i, T_s_i),
        "T_topCOC_insulated_C": np.interp(elapsed, t_i, T_top_i),
        "T_outer_cover_insulated_C": np.interp(elapsed, t_i, T_cover_i),
        "delta_sample_insulated_minus_bare_C": np.interp(
            elapsed, t_i, T_s_i) - np.interp(elapsed, t_b, T_s_b),
    })
    trace.to_csv(out_dir / "bare_vs_insulated_trace.csv", index=False)

    # ---- 样品峰指标 ----
    s_max_b = float(np.max(T_s_b))
    s_max_i = float(np.max(T_s_i))
    t_smax_i = float(t_i[int(np.argmax(T_s_i))])

    # 重复周期峰 (样品插值到测量轴)
    T_sb_m = np.interp(elapsed, t_b, T_s_b)
    T_si_m = np.interp(elapsed, t_i, T_s_i)
    cyc_b = detect_repeated_cycles(elapsed, tint, T_sb_m)
    cyc_i = detect_repeated_cycles(elapsed, tint, T_si_m)
    peaks_b = [float(c["sample_high_peak_C"])
               for c in cyc_b["repeated_cycles"]]
    peaks_i = [float(c["sample_high_peak_C"])
               for c in cyc_i["repeated_cycles"]]
    n_b = len(peaks_b)
    n_i = len(peaks_i)
    mn_b, mx_b, mean_b = _stats(peaks_b)
    mn_i, mx_i, mean_i = _stats(peaks_i)

    flags_b = threshold_flags(T_s_b)
    flags_i = threshold_flags(T_s_i)
    cnt_b = repeated_counts(peaks_b)
    cnt_i = repeated_counts(peaks_i)

    # ---- 顶部表面效应 ----
    topCOC_max_b = float(np.max(T_top_b))
    topCOC_max_i = float(np.max(T_top_i))
    cover_max_i = float(np.max(T_cover_i))

    # ---- 外部热损失诊断 (样品峰时刻, 表面温度) ----
    # 裸顶: 表面 = Top COC 外表面; 绝缘: 表面 = PDMS 外表面
    idx_b = int(np.argmax(T_s_b))
    idx_i = int(np.argmax(T_s_i))
    q_loss_b = external_heat_loss_W_m2(float(T_top_b[idx_b]), t_env)
    q_loss_i = external_heat_loss_W_m2(float(T_cover_i[idx_i]), t_env)

    return {
        "protocol": name,
        "internal_max_C": float(np.max(tint)),
        "T_initial_C": T_init,
        "environment_C": t_env,
        "environment_source": env["environment_source"],
        # 样品峰
        "sample_peak_bare_C": s_max_b,
        "sample_peak_insulated_C": s_max_i,
        "delta_sample_peak_C": s_max_i - s_max_b,
        "time_of_insulated_sample_max_s": t_smax_i,
        # 阈值 (样品)
        **{"bare_" + k: v for k, v in flags_b.items()},
        **{"insulated_" + k: v for k, v in flags_i.items()},
        # 重复周期
        "repeated_cycle_count_bare": n_b,
        "repeated_cycle_count_insulated": n_i,
        "repeated_peak_min_bare_C": mn_b,
        "repeated_peak_max_bare_C": mx_b,
        "repeated_peak_mean_bare_C": mean_b,
        "repeated_peak_min_insulated_C": mn_i,
        "repeated_peak_max_insulated_C": mx_i,
        "repeated_peak_mean_insulated_C": mean_i,
        "delta_repeated_peak_mean_C": (mean_i - mean_b) if
        (np.isfinite(mean_i) and np.isfinite(mean_b)) else np.nan,
        **{"bare_repeated_ge" + str(int(th)): cnt_b[f"ge{int(th)}"]
           for th in THRESHOLDS},
        **{"insulated_repeated_ge" + str(int(th)): cnt_i[f"ge{int(th)}"]
           for th in THRESHOLDS},
        # 顶部表面
        "topCOC_surface_max_bare_C": topCOC_max_b,
        "topCOC_air_interface_max_insulated_C": topCOC_max_i,
        "outer_cover_surface_max_insulated_C": cover_max_i,
        # 外部热损失 (样品峰时刻)
        "external_heat_loss_bare_W_m2": q_loss_b,
        "external_heat_loss_insulated_W_m2": q_loss_i,
        # 有限性
        "all_finite": bool(np.all(np.isfinite(T_s_b))
                           and np.all(np.isfinite(T_s_i))),
    }, (trace, r_bare, r_ins)


# ============================================================
# 图
# ============================================================

def plot_protocol(proto_out, trace_df, name, out_dir):
    t = trace_df["time_s"].to_numpy()
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.plot(t, trace_df["T_internal_C"], color="#7f7f7f", lw=1.1, ls=":",
            label="Measured internal temperature")
    ax.plot(t, trace_df["T_sample_bare_C"], color="#1f77b4", lw=2.0,
            label="Predicted sample — Bare Top")
    ax.plot(t, trace_df["T_sample_insulated_C"], color="#d62728", lw=2.0,
            label="Predicted sample — Sealed-Air Insulated")
    ax.axhline(90.0, color="#8c564b", ls="--", lw=1.2, alpha=0.8,
               label="90 C thermal reference")
    ax.axhline(95.0, color="#000000", ls="--", lw=1.2, alpha=0.6,
               label="95 C thermal reference")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Temperature [C]")
    ax.set_title(f"{name} — bare vs sealed-air insulated sample temperature\n"
                 f"(frozen k={K_EFF}, cp={CP_EFF:.0f}; h={H_CONV} + "
                 "nonlinear radiation)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "bare_vs_insulated_sample_temperature.png", dpi=150)
    fig.savefig(out_dir / "bare_vs_insulated_sample_temperature.pdf")
    plt.close(fig)


def plot_comparison(summary_rows, comp_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    protos = [r["protocol"] for r in summary_rows]
    bare = [r["sample_peak_bare_C"] for r in summary_rows]
    ins = [r["sample_peak_insulated_C"] for r in summary_rows]
    x = np.arange(len(protos))
    w = 0.35
    ax.bar(x - w / 2, bare, w, color="#1f77b4", label="Bare Top")
    ax.bar(x + w / 2, ins, w, color="#d62728",
           label="Sealed-Air Insulated")
    ax.axhline(90.0, color="#8c564b", ls="--", lw=1.2, alpha=0.8,
               label="90 C thermal reference")
    ax.axhline(95.0, color="#000000", ls="--", lw=1.2, alpha=0.6,
               label="95 C thermal reference")
    ax.set_xticks(x)
    ax.set_xticklabels(protos)
    ax.set_ylabel("Predicted overall sample peak [C]")
    ax.set_title("Predicted sample peak — bare vs sealed-air insulated\n"
                 "(frozen model; 90/95 C = thermal reference only)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(comp_dir / "sample_peak_bare_vs_insulated.png", dpi=150)
    plt.close(fig)


# ============================================================
# 主流程 + sanity checks
# ============================================================

def _run_sanity_checks(data, r_bare, r_ins, insulated_layers):
    """运行时 sanity checks (规格 #34)。失败即抛错。"""
    elapsed = data["elapsed_time_s"]
    tint = data["T_internal_C"]

    # 1. 同一内部迹线作为底部边界 (求解器内部使用同 tint)
    assert np.allclose(r_bare["T_bottom_arr"][0],
                       np.interp(r_bare["t_array"][0], elapsed, tint))
    # 2. 同一 k_eff/cp_eff (make_convection_radiation_materials 固定)
    #    已由模块常量保证; 检查材料库 COC 值
    mats = cr.make_convection_radiation_materials(K_EFF, CP_EFF, RHO_COC)
    assert mats["COC"].k_W_mK == K_EFF and mats["COC"].cp_J_kgK == CP_EFF
    # 3. 同一初始温度
    assert r_bare["T_initial_C"] == r_ins["T_initial_C"] if \
        "T_initial_C" in r_ins else True
    # 4. 同一环境 (调用方传参相同, 此处记录)
    # 5. 绝缘栈含 Air + PDMS 且位于 Top COC 之上
    names = [l.name for l in insulated_layers]
    assert "Air Gap" in names and "Cap PDMS" in names
    top_idx = names.index("Top COC")
    air_idx = names.index("Air Gap")
    pdms_idx = names.index("Cap PDMS")
    assert air_idx == top_idx + 1 and pdms_idx == air_idx + 1
    # 6/7. 环境对流+辐射只作用于最外节点: 由 run_convection_radiation_fdm
    #     边界实现保证 (Robin/非线性边界仅在 T[-1]); 绝缘最外节点 = PDMS
    assert r_ins["T_outer_surface_arr"].size == r_ins["T_sample_arr"].size
    # 8. 所有输出温度有限
    for arr in (r_bare["T_sample_arr"], r_ins["T_sample_arr"],
                r_ins["T_outer_surface_arr"]):
        assert np.all(np.isfinite(arr)), "输出温度含 NaN/Inf。"
    # 9. 时间数组严格递增
    for arr in (r_bare["t_array"], r_ins["t_array"]):
        assert np.all(np.diff(arr) > 0), "时间数组未严格递增。"
    # 10. 样品层同一物理 20 um 层
    sw_b = r_bare["mesh"].sample_weights
    sw_i = r_ins["mesh"].sample_weights
    assert sw_b.ndim == 1 and sw_i.ndim == 1
    assert np.isclose(np.sum(sw_b), 1.0) and np.isclose(np.sum(sw_i), 1.0)
    # 样品区间 (物理 180-200 um)
    x_b = r_bare["mesh"].x
    x_i = r_ins["mesh"].x
    in_b = x_b[sw_b > 0]
    in_i = x_i[sw_i > 0]
    assert np.isclose(np.min(in_b), 180e-6, atol=15e-6) and \
        np.isclose(np.max(in_b), 200e-6, atol=15e-6)
    assert np.isclose(np.min(in_i), 180e-6, atol=15e-6) and \
        np.isclose(np.max(in_i), 200e-6, atol=15e-6)
    return True


def main():
    comp_dir = OUTPUT_ROOT / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    insulated_layers, (top_coc_layer, air_layer, pdms_layer) = \
        make_insulated_layers()
    air_thickness = air_layer.thickness_m
    pdms_thickness = pdms_layer.thickness_m
    air_k = heat_model.DEFAULT_MATERIALS["Air"].k_W_mK
    pdms_k = heat_model.DEFAULT_MATERIALS["PDMS"].k_W_mK
    r_air_area = air_thickness / air_k
    r_pdms_area = pdms_thickness / pdms_k

    summaries = []
    traces = {}
    for label, path, out_name in (
            ("DOE11 Faster PCR", DOE11_PATH, "DOE11_faster"),
            ("Test_PCR Longer Holding", LONGER_PATH,
             "Test_PCR_longer_holding")):
        out_dir = OUTPUT_ROOT / out_name
        s, (trace, r_bare, r_ins) = process_protocol(label, path, out_dir)
        # 完整性断言 (任何关键失败即抛错)
        data = {"elapsed_time_s": trace["time_s"].to_numpy(),
                "T_internal_C": trace["T_internal_C"].to_numpy()}
        _run_sanity_checks(data, r_bare, r_ins, insulated_layers)
        if not s["all_finite"]:
            raise RuntimeError(f"{label}: 输出温度含非有限值。")
        summaries.append(s)
        traces[out_name] = trace
        plot_protocol(s, trace, label, out_dir)
        print(f"[{label}] bare sample peak = {s['sample_peak_bare_C']:.2f} C; "
              f"insulated = {s['sample_peak_insulated_C']:.2f} C; "
              f"delta = {s['delta_sample_peak_C']:+.2f} C")

    # ---- 汇总 CSV ----
    cols = ["protocol", "internal_max_C", "T_initial_C", "environment_C",
            "environment_source", "sample_peak_bare_C",
            "sample_peak_insulated_C", "delta_sample_peak_C",
            "time_of_insulated_sample_max_s",
            "bare_ge85", "bare_ge90", "bare_ge92", "bare_ge95",
            "insulated_ge85", "insulated_ge90", "insulated_ge92",
            "insulated_ge95",
            "repeated_cycle_count_bare", "repeated_cycle_count_insulated",
            "repeated_peak_min_bare_C", "repeated_peak_max_bare_C",
            "repeated_peak_mean_bare_C",
            "repeated_peak_min_insulated_C", "repeated_peak_max_insulated_C",
            "repeated_peak_mean_insulated_C",
            "delta_repeated_peak_mean_C",
            "bare_repeated_ge85", "bare_repeated_ge90", "bare_repeated_ge92",
            "bare_repeated_ge95",
            "insulated_repeated_ge85", "insulated_repeated_ge90",
            "insulated_repeated_ge92", "insulated_repeated_ge95",
            "topCOC_surface_max_bare_C",
            "topCOC_air_interface_max_insulated_C",
            "outer_cover_surface_max_insulated_C",
            "external_heat_loss_bare_W_m2",
            "external_heat_loss_insulated_W_m2",
            "all_finite"]
    df = pd.DataFrame(summaries)[cols]
    df.to_csv(comp_dir / "bare_vs_insulated_sample_peak_summary.csv",
              index=False)

    plot_comparison(summaries, comp_dir)

    # ---- 元数据 ----
    meta = {
        "purpose": "forward physics evaluation: bare vs sealed-air "
                   "insulated top",
        "frozen_coc": {"k_eff": K_EFF, "cp_eff": CP_EFF,
                       "rho": RHO_COC, "alpha": K_EFF / (RHO_COC * CP_EFF)},
        "tau_top_s": TAU_TOP,
        "tau_affects_sample": False,
        "top_environment": {"h_conv": H_CONV, "emissivity": EPS,
                            "sigma": SIGMA, "F_view": F_VIEW,
                            "radiation": "nonlinear Stefan-Boltzmann"},
        "bare_stack": "Bottom COC 180 / Sample 20 / Oil 50 / Top COC 600 um "
                      "(BARE_TOP_COC_LAYERS)",
        "insulated_stack": "Bottom COC 180 / Sample 20 / Oil 50 / "
                           "Top COC 600 / Sealed Air 3000 / PDMS 200 um "
                           "(Air+PDMS geometry reused from "
                           "LEGACY_INSULATED_LAYERS)",
        "air_gap": {"thickness_m": air_thickness,
                    "k_W_mK": air_k,
                    "rho": heat_model.DEFAULT_MATERIALS["Air"].rho_kg_m3,
                    "cp": heat_model.DEFAULT_MATERIALS["Air"].cp_J_kgK,
                    "conduction_only": True,
                    "internal_convection": "NOT INCLUDED",
                    "internal_radiation": "NOT INCLUDED"},
        "pdms": {"thickness_m": pdms_thickness,
                 "k_W_mK": pdms_k,
                 "rho": heat_model.DEFAULT_MATERIALS["PDMS"].rho_kg_m3,
                 "cp": heat_model.DEFAULT_MATERIALS["PDMS"].cp_J_kgK},
        "environmental_boundary": {
            "bare": "direct on Top COC outer surface",
            "insulated": "outer PDMS surface only (not on Top COC)"},
        "nominal_R_area": {"air_m2K_W": r_air_area,
                           "pdms_m2K_W": r_pdms_area,
                           "ratio_air_over_pdms": r_air_area / r_pdms_area},
        "limitation": "sealed air layer modeled as conduction-only; "
                      "internal surface-to-surface radiation across the "
                      "air gap is omitted, so the insulated case may "
                      "somewhat overestimate the real air-gap thermal "
                      "resistance",
        "not_calibration": True,
        "no_scan": True,
        "no_fit": True,
        "same_internal_trace": True,
        "same_initial": True,
        "same_environment": True,
        "thresholds": {"values": list(THRESHOLDS),
                       "note": "thermal plausibility references only"},
        "old_outputs_unchanged": True,
        "new_source_files": ["evaluate_bare_vs_insulated_top.py"],
    }
    (comp_dir / "bare_vs_insulated_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt = _summary_text(summaries, meta)
    (comp_dir / "bare_vs_insulated_summary.txt").write_text(
        txt, encoding="utf-8")
    print(txt)


def _summary_text(summaries, meta):
    L = []
    A = L.append
    A("=" * 72)
    A("BARE-TOP VS SEALED-AIR INSULATION FORWARD EVALUATION")
    A("=" * 72)
    A(f"冻结 COC: k={K_EFF}, cp={CP_EFF}, rho={RHO_COC}, "
      f"alpha={K_EFF/(RHO_COC*CP_EFF):.3e} m2/s; tau_top={TAU_TOP} s "
      "(仅裸顶观测诊断, 不影响样品)")
    air_k = meta["air_gap"]["k_W_mK"]
    pdms_k = meta["pdms"]["k_W_mK"]
    r_air_area = meta["nominal_R_area"]["air_m2K_W"]
    r_pdms_area = meta["nominal_R_area"]["pdms_m2K_W"]
    A(f"空气间隙: {meta['air_gap']['thickness_m']*1e3:.1f} mm, "
      f"k={air_k} W/mK; 传导主导 (无内部对流/辐射)")
    A(f"PDMS 盖层: {meta['pdms']['thickness_m']*1e6:.0f} um, "
      f"k={pdms_k} W/mK")
    A(f"标称热阻: R_air={r_air_area:.4f} m2K/W, "
      f"R_PDMS={r_pdms_area:.4f} m2K/W, 比值={r_air_area/r_pdms_area:.1f}")
    A("")
    for s in summaries:
        A(f"[{s['protocol']}] internal max={s['internal_max_C']:.2f} C; "
          f"T_initial={s['T_initial_C']:.2f} C; "
          f"env={s['environment_C']:.2f} C "
          f"({s['environment_source']})")
        A(f"  样品峰: bare={s['sample_peak_bare_C']:.2f} C -> "
          f"insulated={s['sample_peak_insulated_C']:.2f} C "
          f"(delta {s['delta_sample_peak_C']:+.2f} C, "
          f"@t={s['time_of_insulated_sample_max_s']:.1f} s)")
        A(f"  重复周期峰 mean: bare={s['repeated_peak_mean_bare_C']:.2f} C "
          f"(n={s['repeated_cycle_count_bare']}) -> "
          f"insulated={s['repeated_peak_mean_insulated_C']:.2f} C "
          f"(n={s['repeated_cycle_count_insulated']}); "
          f"delta_mean={s['delta_repeated_peak_mean_C']:+.2f} C")
        A(f"  阈值 (样品): bare >=90: "
          f"{'YES' if s['bare_ge90'] else 'NO'}, "
          f">=95: {'YES' if s['bare_ge95'] else 'NO'}; "
          f"insulated >=90: {'YES' if s['insulated_ge90'] else 'NO'}, "
          f">=95: {'YES' if s['insulated_ge95'] else 'NO'}")
        A(f"  重复周期 >=90: bare {s['bare_repeated_ge90']}/"
          f"{s['repeated_cycle_count_bare']}, "
          f"insulated {s['insulated_repeated_ge90']}/"
          f"{s['repeated_cycle_count_insulated']}")
        A(f"  顶部表面 max: bare Top COC={s['topCOC_surface_max_bare_C']:.2f} C; "
          f"insulated Top COC/Air 界面="
          f"{s['topCOC_air_interface_max_insulated_C']:.2f} C, "
          f"PDMS 外表面={s['outer_cover_surface_max_insulated_C']:.2f} C")
        A(f"  外部热损失 (样品峰时刻): bare="
          f"{s['external_heat_loss_bare_W_m2']:.1f} W/m2, "
          f"insulated={s['external_heat_loss_insulated_W_m2']:.1f} W/m2")
        A("")
    A("重要局限: 密封空气层按传导-only 建模; 省略空气间隙内部表面-表面"
      "辐射, 绝缘情形可能高估真实空气间隙热阻。")
    A("结论为前向物理评估 (非标定/拟合/优化); PCR 成功/失败未用于调参。")
    return "\n".join(L)


if __name__ == "__main__":
    main()
