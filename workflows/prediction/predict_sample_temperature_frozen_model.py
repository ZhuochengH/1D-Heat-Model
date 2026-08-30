#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可复用样品层温度预测工具 — 冻结热模型 (FINAL FROZEN THERMAL MODEL V2)
======================================================================

模型解释 (务必如实转述):
    This tool predicts sample-layer temperature using the frozen reduced-order
    thermal model calibrated against one Top COC dataset and externally
    validated against independent corrected bare-top experiments.

    Sample temperature is model-predicted and not directly measured.

    The external validation RMSE is several tenths to several degrees C,
    indicating reduced-order predictive accuracy rather than sub-degree
    precision. Validation RMSE is NOT sample-temperature uncertainty.

科学状态 (区分两种配置 / 两个模型版本):
    BARE mode:
        冻结降阶模型, 使用一个 Top COC 实验标定, 并针对独立裸顶数据集
        做过外部验证。
        裸顶预测可描述为 "来自冻结且外部验证过的降阶模型的预测"。

    INSULATED mode:
        同一冻结有效 COC 模型 + 显式 3 mm 密封空气层绝缘几何的前向扩展。
        默认 V2 无 PDMS (简化绝缘几何); 显式 --model-version v1 时使用
        历史 200 um PDMS 盖帽几何 (V1 复现)。
        **尚未**使用实测绝缘 Top COC 温度做独立验证。

    BOTH modes:
        样品温度均为模型预测, 非直接实测。

    MODEL VERSION:
        默认 v2 = FINAL_FROZEN_THERMAL_MODEL_V2 (FC-70, 无 PDMS);
        v1 = FINAL_FROZEN_THERMAL_MODEL_V1 (Oil + PDMS, 历史复现)。

绝缘假设 (仅一阶近似, 不引入新物理):
    "The insulated configuration treats the 3 mm sealed air layer as
     conduction-dominated. Internal radiative exchange across the air gap is
     omitted, so the insulation resistance may be somewhat overestimated."
    - 不加入空气隙内部自然对流;
    - 不加入空气隙表面间显式辐射。

冻结模型参数 (绝不重新拟合 / 不扫描 / 不优化):
    ID            : FINAL_FROZEN_THERMAL_MODEL_V2 (默认) /
                     FINAL_FROZEN_THERMAL_MODEL_V1 (显式 v1)
    k_eff         : 0.0700 (V2) / 0.0675 (V1) W/(m K)
    cp_eff        : 700 J/(kg K)
    rho_COC       : 1020 kg/m3
    tau_top       : 8.0 s  (输出侧滞后, 仅属于顶部观测模型;
                            本工具不把它施加到样品温度)
    h_conv        : 10.0 W/(m2 K)    (作用于外表面)
    emissivity    : 0.90
    sigma_SB      : 5.670374419e-8 W/(m2 K4)
    F_view        : 1.0

几何:
    V2 BARE       : V2_BARE_TOP_COC_LAYERS   (180 COC / 20 sample / 50 FC-70 /
                                              600 COC, 总 850 um)
    V2 INSULATED  : V2_INSULATED_NO_PDMS_LAYERS (180 COC / 20 sample /
                                              50 FC-70 / 600 COC / 3000 Air,
                                              总 3850 um; 无 PDMS)
    V1 BARE       : BARE_TOP_COC_LAYERS      (Oil; 850 um)
    V1 INSULATED  : LEGACY_INSULATED_LAYERS  (Oil + 200 um PDMS; 4050 um)

热历史语义 (关键):
    - 模拟 (SIMULATION HISTORY) 始终从完整源迹线起点 t0 开始, 使用完整实测
      内部温度历史, 传播到 --end-s (或末点);
    - 分析/显示窗口 (ANALYSIS WINDOW) 由 --start-s 控制: 仅在绘图/CSV/
      摘要/交互时裁剪输出, 绝不重置热状态;
    - T_initial = 完整源迹线第一个有效内部温度 (不是窗口首点);
    - T_environment = 同一值 (INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT),
      不随 --start-s 改变;
    - thermal_history_preserved = YES;
    - --end-s 可安全截断前向模拟 (无需计算窗口之后的源数据)。

数值方法:
    一维节点中心有限体积 (node-centered finite-volume) 模型
    (physical finite-volume solution)。历史代码/函数名中的 "fdm" 仅为
    历史命名, 不表示数值方法本身是 FDM。

样品温度:
    样品 = 原始有限体积温度场经样品层控制体积加权 (180-200 um)。
    绝不施加 tau_top。裸顶 raw Top / 绝缘 Top COC-Air 界面 / 绝缘外表面
    仅作次级诊断列。

CLI:
    --input      (必填) 内部温度工作簿 (.xlsx)
    --model      bare | insulated | both   (默认 bare)
    --model-version v2 | v1                (默认 v2)
    --start-s    分析窗口起点 (默认完整源起点)
    --end-s      模拟/分析终点 (默认完整源终点)
    --output-dir 自定义输出目录 (默认 sample_temperature_output/<stem>/<model>/)
    --no-gui     不启动交互窗口

交互:
    复用项目现有 draggable_hlines.py 的 DraggableHLine / find_intersections
    (AST 提取, 不改动原文件)。阈值统计应用于预测样品温度, 时间戳感知。
    单模式: 线应用于该模式的样品; both 模式: 一个公共可拖阈值同时显示
    bare 与 insulated 的 "time sample >= threshold"。交互线为探索工具,
    绝不写入模型配置。

权威验证参考 (只读, 不重新计算) — FINAL_FROZEN_THERMAL_MODEL_V2:
    66C 标定            : RMSE 0.6333 C
    60C 外部验证 (已知偏移): RMSE 1.3139 C
    72C 外部验证 (已知偏移): RMSE 3.0132 C (冷却相局限同 V1)
    3s 外部验证         : RMSE 1.0386 C
    外部验证均值         : 1.7886 C

科学状态:
    BARE     = 已标定 + 已外部验证。
    INSULATED = 简化绝缘几何 (3 mm 密封空气, 无 PDMS) 的前向扩展,
                未经绝缘 Top COC 实测独立验证。
    样品温度 = 模型预测的隐藏热状态, 非直接实测;
    外部 Top RMSE 不等于直接样品温度不确定性。

用法示例
--------
    uv run python workflows/prediction/predict_sample_temperature_frozen_model.py \
        --input "<xlsx>" --model bare --no-gui

    uv run python workflows/prediction/predict_sample_temperature_frozen_model.py \
        --input "<xlsx>" --model both --start-s 100 --end-s 200 --no-gui
"""
import argparse
import ast
import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# 注意: 不在模块顶层强制 Agg —— --no-gui 时在 main() 中切换;
# 交互窗口需要 GUI 后端。静态绘图在任意后端下均可 savefig。
import matplotlib.pyplot as plt

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.config.final_frozen_model_v2 import (
    FINAL_FROZEN_THERMAL_MODEL_V2,
)
from thermal_model.config.final_frozen_model import (
    FINAL_FROZEN_THERMAL_MODEL_V1,
)
# 注意: 不 import predict_sample_from_internal_temperature —— 其模块顶层强制
# matplotlib.use("Agg"), 会锁定非交互后端, 使本工具的交互窗口永远无法打开。
# 加载语义与该项目已验证加载器 (load_internal_data) 逐项一致, 在此内联实现
# (纯函数, 无 matplotlib 副作用)。

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# 冻结模型身份 (唯一事实来源 = thermal_model.config.final_frozen_model_v2)
# 生产默认 = FINAL_FROZEN_THERMAL_MODEL_V2 (FC-70 + no-PDMS);
# 历史 V1 (Oil + PDMS) 可通过 --model-version v1 显式调用。
# ============================================================

MODEL_VERSION_DEFAULT = "v2"
MODEL_VERSION_CHOICES = ("v1", "v2")


def _version_params(model_version):
    """返回 (k_eff, cp_eff, rho, tau_top, model_id, source_id) 参数集。"""
    if model_version == "v2":
        m = FINAL_FROZEN_THERMAL_MODEL_V2
        return (float(m.k_eff_W_mK), float(m.cp_eff_J_kgK),
                float(m.rho_COC_kg_m3), float(m.tau_top_s),
                m.model_id,
                "THERMAL_MODEL_V2_CANDIDATE -> "
                "FINAL_FROZEN_THERMAL_MODEL_V2 (k=0.0700, cp=700, tau=8.0; "
                "FC-70 + no-PDMS insulated geometry)")
    if model_version == "v1":
        m = FINAL_FROZEN_THERMAL_MODEL_V1
        return (float(m.k_eff_W_mK), float(m.cp_eff_J_kgK),
                float(m.rho_COC_kg_m3), float(m.tau_top_s),
                m.model_id,
                "66C_RECALIBRATED_CANDIDATE_V1 -> "
                "FINAL_FROZEN_THERMAL_MODEL_V1 (k=0.0675, cp=700, tau=8.0; "
                "Oil + PDMS legacy geometry)")
    raise ValueError(
        f"未知模型版本 {model_version!r}; 可用: {MODEL_VERSION_CHOICES}")


# 模块级默认 (生产路径 = V2); 历史 V1 用 _version_params("v1") 显式调用。
(K_EFF, CP_EFF, RHO_COC, TAU_TOP, MODEL_ID, FROZEN_MODEL_SOURCE_ID) = (
    _version_params(MODEL_VERSION_DEFAULT))

H_CONV = cr.H_CONV_STRATEGY_E_W_M2K
EPS = cr.EMISSIVITY_STRATEGY_E
SIGMA = cr.SIGMA_SB_W_M2_K4
F_VIEW = cr.VIEW_FACTOR_STRATEGY_E

ENVIRONMENT_SOURCE = "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT"

# 模型选择
MODEL_CHOICES = ("bare", "insulated", "both")
DEFAULT_MODEL = "bare"
INSULATION_AIR_THICKNESS_M = 3000e-6   # 密封空气 3 mm (V2 与 V1 一致)

# 权威验证参考 (只读, 仅用于文档/摘要; 默认 V2)
CALIB_RMSE_66C = FINAL_FROZEN_THERMAL_MODEL_V2.calibration_66C_RMSE_C
VALIDATION_RMSE_C = dict(
    FINAL_FROZEN_THERMAL_MODEL_V2.external_validation_RMSE_C)

SAVE_DT = 0.02  # FDM 输出下采样间隔: 越小, 部分窗口 vs 全历史切片的
                # 采样相位误差越小 (相位误差 <= save_dt, 插值误差 <= 斜率*save_dt)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "sample_temperature_output"
DESCRIPTIVE_THRESHOLDS_C = (85.0, 87.0, 90.0, 92.0, 95.0)

# ============================================================
# draggable_hlines.py 直接复用 (已重构为可安全 import)
# ============================================================

from thermal_model.utilities.draggable_hlines import (  # noqa: E402
    DraggableHLine,
    find_intersections,
)


# ============================================================
# 几何构建 (bare / insulated)
# ============================================================

def build_geometry(model, model_version=MODEL_VERSION_DEFAULT):
    """返回指定配置的层叠结构 (默认 V2; 显式 v1 保留历史几何)。

    bare      : V2  -> V2_BARE_TOP_COC_LAYERS (FC-70, 无 Air/PDMS);
                V1  -> BARE_TOP_COC_LAYERS (Oil, 无 Air/PDMS)。
    insulated : V2  -> V2_INSULATED_NO_PDMS_LAYERS 本地副本 + Top COC
                       role="top_surface" (无 PDMS; 5 层, 3850 um);
                V1  -> LEGACY_INSULATED_LAYERS 本地副本 + Top COC
                       role="top_surface" (Oil + PDMS; 6 层, 4050 um)。
    """
    if model_version == "v2":
        bare_layers = FINAL_FROZEN_THERMAL_MODEL_V2.bare_layers
        ins_template = FINAL_FROZEN_THERMAL_MODEL_V2.insulated_layers
    elif model_version == "v1":
        bare_layers = heat_model.BARE_TOP_COC_LAYERS
        ins_template = heat_model.LEGACY_INSULATED_LAYERS
    else:
        raise ValueError(
            f"未知模型版本 {model_version!r}; 可用: {MODEL_VERSION_CHOICES}")
    if model == "bare":
        return bare_layers
    if model == "insulated":
        layers = heat_model.copy_layers(ins_template)
        for layer in layers:
            if layer.name == "Top COC":
                layer.role = "top_surface"
        return layers
    raise ValueError(f"未知模型配置 {model!r}; 可用: {MODEL_CHOICES}")


def _assert_frozen_params(model_version=MODEL_VERSION_DEFAULT):
    """运行时断言冻结参数 (防御; 绝不重新拟合)。"""
    k, cp, rho, tau, mid, _src = _version_params(model_version)
    for name, val, ref in (("k_eff", K_EFF, k),
                           ("cp_eff", CP_EFF, cp),
                           ("rho_COC", RHO_COC, rho),
                           ("tau_top", TAU_TOP, tau),
                           ("h_conv", H_CONV, 10.0),
                           ("epsilon", EPS, 0.90)):
        if abs(float(val) - float(ref)) > 1e-12:
            raise RuntimeError(
                f"冻结参数断言失败: {name}={val} != {ref}")


# ============================================================
# 时间窗切分 (纯函数; V2 热历史语义)
# ============================================================

def _validate_bounds(start_s, end_s, first, last):
    """校验并归一化 --start-s / --end-s (引用工作簿 Time(s) 轴)。"""
    if start_s is None:
        start_s = first
    if end_s is None:
        end_s = last
    start_s = float(start_s)
    end_s = float(end_s)
    if not (np.isfinite(start_s) and np.isfinite(end_s)):
        raise ValueError("--start-s / --end-s 必须为有限数值。")
    if start_s < first - 1e-9 or start_s > last + 1e-9:
        raise ValueError(
            f"--start-s {start_s} 不在可用时间范围 [{first:.3f}, "
            f"{last:.3f}] s 内。")
    if end_s < first - 1e-9 or end_s > last + 1e-9:
        raise ValueError(
            f"--end-s {end_s} 不在可用时间范围 [{first:.3f}, "
            f"{last:.3f}] s 内。")
    if start_s >= end_s:
        raise ValueError(
            f"--start-s ({start_s}) 必须小于 --end-s ({end_s})。")
    return start_s, end_s


def select_analysis_window(t_original, T_internal, start_s=None, end_s=None):
    """选择分析/显示窗口: start_s <= t <= end_s。

    只负责裁剪 (绘图/CSV/摘要/交互), 绝不重置 FDM 热状态。
    返回 dict: mask / t_original / T_internal / start_s / end_s /
               n_points / duration_s。
    """
    t = np.asarray(t_original, dtype=float)
    T = np.asarray(T_internal, dtype=float)
    first = float(t[0])
    last = float(t[-1])
    start_s, end_s = _validate_bounds(start_s, end_s, first, last)
    mask = (t >= start_s - 1e-9) & (t <= end_s + 1e-9)
    t_win = t[mask]
    T_win = T[mask]
    if len(t_win) < 2:
        raise ValueError(
            f"分析窗口 [{start_s}, {end_s}] 内有效内部数据点不足 (<2)。")
    return {
        "mask": mask,
        "t_original": t_win,
        "T_internal": T_win,
        "start_s": float(t_win[0]),
        "end_s": float(t_win[-1]),
        "n_points": int(len(t_win)),
        "duration_s": float(t_win[-1] - t_win[0]),
    }


def select_simulation_range(t_original, end_s):
    """选择模拟 (热历史) 范围: 完整源 t0 -> end_s。

    end_s 为 None 时模拟到源末点。返回 (t_sim_src, T_sim_src, t_sim_rel):
        t_sim_rel = t_sim_src - t_sim_src[0]  (FDM 时间轴, 从真模拟起点 0 起)
    """
    t = np.asarray(t_original, dtype=float)
    first = float(t[0])
    last = float(t[-1])
    if end_s is None:
        end_s = last
    end_s = float(end_s)
    if end_s < first - 1e-9 or end_s > last + 1e-9:
        raise ValueError(
            f"--end-s {end_s} 不在可用时间范围 [{first:.3f}, "
            f"{last:.3f}] s 内。")
    mask = t <= end_s + 1e-9
    t_src = t[mask]
    if len(t_src) < 2:
        raise ValueError("模拟范围内有效内部数据点不足 (<2)。")
    return t_src, t_src - first


# ============================================================
# 阈值时序 (时间戳感知, 分段线性精确)
# ============================================================

def compute_threshold_timing(t, T, threshold):
    """对分段线性迹线 T(t) 计算 'T >= threshold' 的时间统计。

    时间戳感知: 每段用实际时间坐标线性插值, 绝不使用 '点数 x 1 s'。
    返回 dict (应用于预测样品温度):
        threshold_C / time_above_s / first_up_cross_s / last_above_s /
        n_intervals / applied_to = "PREDICTED SAMPLE"
    """
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    threshold = float(threshold)
    above = T >= threshold
    total = 0.0
    n_intervals = 0
    first_up = np.nan
    last_above = np.nan
    if above[0]:
        n_intervals = 1
        first_up = t[0]
    for i in range(len(t) - 1):
        y0, y1 = T[i], T[i + 1]
        dt = t[i + 1] - t[i]
        a0, a1 = above[i], above[i + 1]
        if a0 and a1:
            total += dt
        elif a0 or a1:
            tc = t[i] + (threshold - y0) / (y1 - y0) * dt
            if a0:  # 下降穿越: 高于阈值段 = [t[i], tc]
                total += tc - t[i]
                last_above = tc
            else:   # 上升穿越: 高于阈值段 = [tc, t[i+1]]
                total += t[i + 1] - tc
                n_intervals += 1
                if np.isnan(first_up):
                    first_up = tc
        if a1:
            last_above = t[i + 1]
    return {
        "threshold_C": threshold,
        "time_above_s": float(total),
        "first_up_cross_s": float(first_up),
        "last_above_s": float(last_above),
        "n_intervals": int(n_intervals),
        "applied_to": "PREDICTED SAMPLE",
    }


# ============================================================
# 内部温度加载 (语义与项目已验证加载器 load_internal_data 一致,
# 内联以避免其模块顶层 matplotlib.use("Agg") 副作用)
# ============================================================

TIME_COL = "Time(s)"
TEMP_COL = "Zone 1 Avg (°C)"
SHEET = "Extracted_Data"


def _find_column(df, column):
    if column in df.columns:
        return column
    col_norm = re.sub(r"\s+", " ", str(column).strip())
    for c in df.columns:
        if re.sub(r"\s+", " ", str(c).strip()) == col_norm:
            return c
    return None


def load_internal_data(path, sheet=SHEET, time_col=TIME_COL,
                       temp_col=TEMP_COL):
    """加载内部温度日志 (与已验证加载器语义逐项一致)。

    返回 dict:
        source_time_s / T_internal_C / n_valid / resolved_time_col /
        resolved_temp_col / median_dt / first_time / last_time / duration_s /
        strictly_increasing / T_min_C / T_max_C。
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
        raise KeyError(
            f"找不到温度列 {temp_col!r}; 可用列: {list(df.columns)}")

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
    return {
        "source_time_s": t,
        "T_internal_C": T,
        "n_valid": int(len(t)),
        "resolved_time_col": tc,
        "resolved_temp_col": gc,
        "median_dt": float(np.median(dt)),
        "first_time": float(t[0]),
        "last_time": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "strictly_increasing": True,
        "T_min_C": float(np.min(T)),
        "T_max_C": float(np.max(T)),
    }


# ============================================================
# 冻结模型正向预测 (V2: 完整热历史 + 分析窗口裁剪)
# ============================================================

def _run_single_configuration(t_sim_rel, T_sim_src, model, T_init, T_env,
                              save_dt, model_version=MODEL_VERSION_DEFAULT):
    """对单个配置 (bare / insulated) 运行冻结有限体积求解。

    返回全模拟时间序列 (样品 = 原始有限体积温度场经样品层加权,
    绝不经过 lag)。
    """
    _assert_frozen_params(model_version)
    layers = build_geometry(model, model_version=model_version)
    if model_version == "v2":
        mats = v2_materials()
    else:
        mats = cr.make_convection_radiation_materials(K_EFF, CP_EFF, RHO_COC)
    result = cr.run_convection_radiation_fdm(
        time_s=t_sim_rel,
        bottom_temperature_C=T_sim_src,
        materials=mats,
        layers=layers,
        T_air_C=T_env,
        T_surroundings_C=T_env,
        save_dt=save_dt,
        T_initial_C=T_init)
    # 顶部观测: role="top_surface" 节点 (裸顶 = Top COC 外表面;
    # 绝缘 = Top COC/Air 界面); 若空则退回最外层 (V1 绝缘 PDMS 外表面)。
    if result["T_top_surface_arr"].size:
        T_top_obs = result["T_top_surface_arr"]
    else:
        T_top_obs = result["T_outer_surface_arr"]
    return {
        "model": model,
        "model_version": model_version,
        "t_model": result["t_array"],
        "T_sample": result["T_sample_arr"],
        "T_top_obs": T_top_obs,
        "T_outer": result["T_outer_surface_arr"],
        "newton_max_iterations_per_step":
            int(result["newton_max_iterations_per_step"]),
        "max_abs_boundary_residual_W_m2":
            float(result["max_abs_boundary_residual_W_m2"]),
    }


def v2_materials():
    """V2 (FC-70) 材料库: COC 用冻结有效参数, FC-70 用制造商常数。"""
    from thermal_model.config import thermal_model_v2_candidate as v2c
    return v2c.make_v2_materials(K_EFF, CP_EFF, RHO_COC)


def _interp_to_analysis(sim, t_ana, first_full):
    """把全模拟输出插值到分析窗口实测时间 (查询轴 = 时间坐标)。"""
    q = np.asarray(t_ana, dtype=float) - first_full
    return {
        "T_sample": np.interp(q, sim["t_model"], sim["T_sample"]),
        "T_top_obs": np.interp(q, sim["t_model"], sim["T_top_obs"]),
        "T_outer": np.interp(q, sim["t_model"], sim["T_outer"]),
    }


def _sample_stats(t_ana, T_internal, T_sample, label):
    """分析窗口内的样品温度描述统计 + 描述性阈值 (非 PCR 判据)。"""
    i_max = int(np.argmax(T_sample))
    return {
        "label": label,
        "internal_max_C": float(np.max(T_internal)),
        "sample_max_C": float(np.max(T_sample)),
        "sample_max_time_s": float(t_ana[i_max]),
        "internal_at_sample_max_C": float(T_internal[i_max]),
        "sample_min_C": float(np.min(T_sample)),
        "sample_mean_C": float(np.mean(T_sample)),
        "sample_median_C": float(np.median(T_sample)),
        "thresholds": {
            th: bool(np.max(T_sample) >= th)
            for th in DESCRIPTIVE_THRESHOLDS_C
        },
        "all_finite": bool(np.all(np.isfinite(T_sample))),
    }


def run_prediction(input_path, model=DEFAULT_MODEL, start_s=None, end_s=None,
                   save_dt=SAVE_DT, model_version=MODEL_VERSION_DEFAULT):
    """完整正向预测 (V2 热历史语义): 全历史模拟 + 分析窗口裁剪, 不拟合。

    model: 'bare' | 'insulated' | 'both'。
    model_version: 'v2' (默认, FC-70 + no-PDMS) | 'v1' (历史 Oil + PDMS)。
    返回 dict 含时间轴 (t_original/t_sim/t_analysis)、各模式样品序列、
    统计、比较 (both) 与模型元数据。
    """
    if model not in MODEL_CHOICES:
        raise ValueError(f"未知模型 {model!r}; 可用: {MODEL_CHOICES}")
    if model_version not in MODEL_VERSION_CHOICES:
        raise ValueError(
            f"未知模型版本 {model_version!r}; 可用: {MODEL_VERSION_CHOICES}")
    data = load_internal_data(input_path)
    t_full = data["source_time_s"]
    T_full = data["T_internal_C"]
    first_full = float(t_full[0])
    last_full = float(t_full[-1])

    # 校验 + 归一化 (start_s 只在分析窗口用; end_s 同时限模拟)
    start_s, end_s = _validate_bounds(start_s, end_s, first_full, last_full)

    # ---- SIMULATION HISTORY: 完整源 t0 -> end_s (热状态连续) ----
    t_sim_src, t_sim_rel = select_simulation_range(t_full, end_s)
    T_sim_src = T_full[t_full <= end_s + 1e-9]

    # ---- ANALYSIS WINDOW: start_s -> end_s (仅裁剪输出) ----
    win = select_analysis_window(t_full, T_full, start_s, end_s)
    t_ana = win["t_original"]
    T_ana = win["T_internal"]

    # 初始/环境 = 完整源迹线首点 (绝不随 --start-s 改变)
    T_init = float(T_full[0])
    T_env = T_init

    if model == "both":
        bare_sim = _run_single_configuration(
            t_sim_rel, T_sim_src, "bare", T_init, T_env, save_dt,
            model_version=model_version)
        ins_sim = _run_single_configuration(
            t_sim_rel, T_sim_src, "insulated", T_init, T_env, save_dt,
            model_version=model_version)
        b = _interp_to_analysis(bare_sim, t_ana, first_full)
        i = _interp_to_analysis(ins_sim, t_ana, first_full)
        sample_bare = b["T_sample"]
        sample_ins = i["T_sample"]
        delta = sample_ins - sample_bare
        stats_bare = _sample_stats(t_ana, T_ana, sample_bare, "bare")
        stats_ins = _sample_stats(t_ana, T_ana, sample_ins, "insulated")
        i_dmax = int(np.argmax(delta))
        comparison = {
            "sample_max_increase_C":
                stats_ins["sample_max_C"] - stats_bare["sample_max_C"],
            "sample_mean_increase_C":
                stats_ins["sample_mean_C"] - stats_bare["sample_mean_C"],
            "sample_median_increase_C":
                stats_ins["sample_median_C"] - stats_bare["sample_median_C"],
            "max_instantaneous_delta_C": float(np.max(delta)),
            "time_of_max_delta_s": float(t_ana[i_dmax]),
        }
        data_out = {
            "model": "both",
            "sample_active": None,
            "sample_bare": sample_bare,
            "sample_insulated": sample_ins,
            "delta_sample_ins_minus_bare": delta,
            "top_bare_raw": b["T_top_obs"],
            "topCOC_insulated": i["T_top_obs"],
            "outer_PDMS_insulated": i["T_outer"],
            "stats_bare": stats_bare,
            "stats_insulated": stats_ins,
            "comparison": comparison,
            "all_finite": bool(
                np.all(np.isfinite(sample_bare))
                and np.all(np.isfinite(sample_ins))),
            "newton_max_iterations_per_step":
                max(bare_sim["newton_max_iterations_per_step"],
                    ins_sim["newton_max_iterations_per_step"]),
        }
    else:
        sim = _run_single_configuration(
            t_sim_rel, T_sim_src, model, T_init, T_env, save_dt,
            model_version=model_version)
        interp = _interp_to_analysis(sim, t_ana, first_full)
        sample_active = interp["T_sample"]
        stats = _sample_stats(t_ana, T_ana, sample_active, model)
        data_out = {
            "model": model,
            "sample_active": sample_active,
            "sample_bare": sample_active if model == "bare" else None,
            "sample_insulated": sample_active if model == "insulated" else None,
            "delta_sample_ins_minus_bare": None,
            "top_bare_raw": interp["T_top_obs"] if model == "bare" else None,
            "topCOC_insulated": interp["T_top_obs"]
                if model == "insulated" else None,
            "outer_PDMS_insulated": interp["T_outer"]
                if model == "insulated" else None,
            "stats_bare": stats if model == "bare" else None,
            "stats_insulated": stats if model == "insulated" else None,
            "comparison": None,
            "all_finite": stats["all_finite"],
            "newton_max_iterations_per_step":
                sim["newton_max_iterations_per_step"],
        }

    return {
        "input_path": str(Path(input_path).resolve()),
        **data_out,
        # 时间轴 (分析窗口; simulation 相对真模拟起点, analysis 相对窗口起点)
        "t_original": t_ana,
        "t_sim": t_ana - first_full,
        "t_analysis": t_ana - t_ana[0],
        "T_internal": T_ana,
        "sim_start_s": first_full,
        "sim_end_s": float(end_s),
        "window_start_s": float(t_ana[0]),
        "window_end_s": float(t_ana[-1]),
        "requested_start_s": (float(start_s) if start_s is not None
                              else None),
        "requested_end_s": (float(end_s) if end_s is not None else None),
        "first_recorded_s": first_full,
        "last_recorded_s": last_full,
        "n_points": win["n_points"],
        "duration_s": win["duration_s"],
        "initial_internal_C": T_init,
        "environment_C": T_env,
        "environment_source": ENVIRONMENT_SOURCE,
        "thermal_history_preserved": True,
        "source_time_col": data["resolved_time_col"],
        "source_temp_col": data["resolved_temp_col"],
        "source_median_dt_s": data["median_dt"],
        "geometry_bare": ("V2_BARE_TOP_COC_LAYERS (FC-70)" if
                          model_version == "v2"
                          else "BARE_TOP_COC_LAYERS (Oil)"),
        "geometry_insulated": ("V2_INSULATED_NO_PDMS_LAYERS (FC-70, no PDMS)"
                               if model_version == "v2"
                               else "LEGACY_INSULATED_LAYERS (Oil + PDMS)"),
        "model_version": model_version,
        "model_meta": {
            "id": MODEL_ID,
            "model_version": model_version,
            "frozen_source_id": FROZEN_MODEL_SOURCE_ID,
            "k_eff_W_mK": K_EFF,
            "cp_eff_J_kgK": CP_EFF,
            "rho_COC_kg_m3": RHO_COC,
            "tau_top_s": TAU_TOP,
            "h_conv_W_m2K": H_CONV,
            "emissivity": EPS,
            "sigma_SB_W_m2K4": SIGMA,
            "view_factor": F_VIEW,
            "calib_66C_RMSE_C": CALIB_RMSE_66C,
            "external_validation_RMSE_C": dict(VALIDATION_RMSE_C),
        },
    }


# ============================================================
# 输出目录
# ============================================================

def _slug(text):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")


def resolve_output_dir(input_path, output_dir=None, model=DEFAULT_MODEL):
    if output_dir is not None:
        return Path(output_dir)
    stem = _slug(Path(input_path).stem)
    return DEFAULT_OUTPUT_ROOT / stem / model


# ============================================================
# 静态输出 (按模式)
# ============================================================

def _figure_name(model):
    if model == "both":
        return "sample_temperature_bare_vs_insulated"
    return f"sample_temperature_prediction_{model}"


def plot_static(r, out_dir):
    """实测内部 + 预测样品 (按模式), 同一坐标轴, 独立 PNG + PDF。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.plot(r["t_original"], r["T_internal"], color="#1f77b4", lw=1.4,
            label="Measured Internal Temperature")
    if r["model"] == "both":
        ax.plot(r["t_original"], r["sample_bare"], color="#2ca02c", lw=1.8,
                label="Predicted Sample — Bare")
        ax.plot(r["t_original"], r["sample_insulated"], color="#d62728",
                lw=1.8, label="Predicted Sample — Air Insulated")
        title = ("Frozen Thermal Model — Sample Temperature Prediction "
                 "(bare vs air-insulated)")
    elif r["model"] == "bare":
        ax.plot(r["t_original"], r["sample_active"], color="#d62728", lw=1.8,
                label="Predicted Sample Temperature (bare)")
        title = ("Frozen Thermal Model — Sample Temperature Prediction "
                 "(bare)")
    else:
        ax.plot(r["t_original"], r["sample_active"], color="#d62728", lw=1.8,
                label="Predicted Sample Temperature (air-insulated)")
        title = ("Frozen Thermal Model — Sample Temperature Prediction "
                 "(air-insulated)")
    ax.set_title(title)
    ax.set_xlabel("Original experiment time [s]")
    ax.set_ylabel("Temperature [C]")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=10)
    fig.tight_layout()
    base = _figure_name(r["model"])
    png = out_dir / f"{base}.png"
    pdf = out_dir / f"{base}.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _csv_columns(r):
    outer_col = ("predicted_outer_surface_insulated_C" if
                 r.get("model_version") == "v2"
                 else "predicted_outer_PDMS_insulated_C")
    if r["model"] == "both":
        return {
            "original_time_s": r["t_original"],
            "simulation_time_s": r["t_sim"],
            "analysis_time_s": r["t_analysis"],
            "measured_internal_C": r["T_internal"],
            "predicted_sample_bare_C": r["sample_bare"],
            "predicted_sample_insulated_C": r["sample_insulated"],
            "delta_sample_insulated_minus_bare_C":
                r["delta_sample_ins_minus_bare"],
            "predicted_top_bare_raw_C": r["top_bare_raw"],
            "predicted_topCOC_insulated_C": r["topCOC_insulated"],
            outer_col: r["outer_PDMS_insulated"],
        }
    if r["model"] == "bare":
        return {
            "original_time_s": r["t_original"],
            "simulation_time_s": r["t_sim"],
            "analysis_time_s": r["t_analysis"],
            "measured_internal_C": r["T_internal"],
            "predicted_sample_bare_C": r["sample_active"],
            "predicted_top_bare_raw_C": r["top_bare_raw"],
        }
    # insulated
    return {
        "original_time_s": r["t_original"],
        "simulation_time_s": r["t_sim"],
        "analysis_time_s": r["t_analysis"],
        "measured_internal_C": r["T_internal"],
        "predicted_sample_insulated_C": r["sample_active"],
        "predicted_topCOC_air_interface_C": r["topCOC_insulated"],
        outer_col: r["outer_PDMS_insulated"],
    }


def write_csv(r, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_csv_columns(r))
    path = out_dir / f"{_figure_name(r['model'])}.csv"
    df.to_csv(path, index=False)
    return path


def _model_status_line(model, model_version=MODEL_VERSION_DEFAULT):
    if model_version == "v2":
        calib_rmse = FINAL_FROZEN_THERMAL_MODEL_V2.calibration_66C_RMSE_C
        v_rmse = FINAL_FROZEN_THERMAL_MODEL_V2.external_validation_RMSE_C
        ins_geo = ("explicit 3 mm sealed-air + FC-70 + no-PDMS simplified "
                   "insulation geometry (total 3850 um)")
        model_label = "FINAL_FROZEN_THERMAL_MODEL_V2"
    else:
        calib_rmse = FINAL_FROZEN_THERMAL_MODEL_V1.calibration_66C_RMSE_C
        v_rmse = FINAL_FROZEN_THERMAL_MODEL_V1.external_validation_RMSE_C
        ins_geo = ("explicit 3 mm sealed-air + 200 um PDMS insulation "
                   "geometry (total 4050 um)")
        model_label = "FINAL_FROZEN_THERMAL_MODEL_V1"
    if model == "bare":
        return (f"bare: frozen reduced-order model "
                f"({model_label}) calibrated using the "
                f"corrected 66 C Top COC experiment "
                f"(RMSE {calib_rmse:.4f} C) and "
                f"externally validated against three authoritative "
                f"bare-top datasets (60 C {v_rmse['60C']:.2f} C, "
                f"72 C {v_rmse['72C']:.2f} C, "
                f"3s {v_rmse['3s_extension']:.2f} C).")
    return (f"insulated: forward extension of the same frozen effective COC "
            f"model using {ins_geo}. This insulated configuration has not "
            f"yet been independently validated against measured insulated "
            f"Top COC temperature.")


def _threshold_lines(stats):
    L = []
    for th in DESCRIPTIVE_THRESHOLDS_C:
        L.append(f"      >= {th:.0f} C  : "
                 f"{'YES' if stats['thresholds'][th] else 'NO'}")
    return "\n".join(L)


def write_summary(r, out_dir):
    """按模式生成 sample_temperature_summary.txt。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    m = r["model_meta"]
    L = []
    A = L.append
    A("=" * 72)
    A("FROZEN THERMAL MODEL — SAMPLE TEMPERATURE PREDICTION SUMMARY (V2)")
    A("=" * 72)
    A(f"model configuration     : {r['model']}")
    A(f"model id                : {m['id']} ({m['frozen_source_id']})")
    A(f"source workbook         : {r['input_path']}")
    A(f"time column             : {r['source_time_col']} "
      f"(median dt {r['source_median_dt_s']:.4f} s)")
    A(f"internal column         : {r['source_temp_col']}")
    A(f"full source time range  : [{r['first_recorded_s']:.3f}, "
      f"{r['last_recorded_s']:.3f}] s")
    A(f"simulation start        : {r['sim_start_s']:.3f} s (full trace)")
    A(f"simulation end          : {r['sim_end_s']:.3f} s")
    A(f"selected analysis start : {r['window_start_s']:.3f} s "
      f"(requested {r['requested_start_s']})")
    A(f"selected analysis end   : {r['window_end_s']:.3f} s "
      f"(requested {r['requested_end_s']})")
    A(f"thermal_history_preserved: YES")
    A(f"initial internal        : {r['initial_internal_C']:.3f} C "
      f"(full trace first point)")
    A(f"environment temperature : {r['environment_C']:.3f} C "
      f"({r['environment_source']})")
    A("-" * 72)
    A(f"frozen k_eff            : {m['k_eff_W_mK']:.4f} W/(m K)")
    A(f"frozen cp_eff           : {m['cp_eff_J_kgK']:.1f} J/(kg K)")
    A(f"frozen tau_top          : {m['tau_top_s']:.2f} s (sample never lagged)")
    A(f"frozen h_conv           : {m['h_conv_W_m2K']:.2f} W/(m2 K)")
    A(f"frozen emissivity       : {m['emissivity']:.2f}")
    A(f"geometry bare           : {r['geometry_bare']}")
    A(f"geometry insulated      : {r['geometry_insulated']}")
    A("-" * 72)
    if r["model"] == "both":
        sb = r["stats_bare"]
        si = r["stats_insulated"]
        A("BARE:")
        A(f"  sample maximum        : {sb['sample_max_C']:.3f} C @ "
          f"{sb['sample_max_time_s']:.3f} s (original)")
        A(f"  sample mean           : {sb['sample_mean_C']:.3f} C")
        A(f"  sample median         : {sb['sample_median_C']:.3f} C")
        A(f"  sample minimum        : {sb['sample_min_C']:.3f} C")
        A(f"  thresholds (descriptive, NOT PCR success/failure):")
        A(_threshold_lines(sb))
        A("INSULATED:")
        A(f"  sample maximum        : {si['sample_max_C']:.3f} C @ "
          f"{si['sample_max_time_s']:.3f} s (original)")
        A(f"  sample mean           : {si['sample_mean_C']:.3f} C")
        A(f"  sample median         : {si['sample_median_C']:.3f} C")
        A(f"  sample minimum        : {si['sample_min_C']:.3f} C")
        A(f"  thresholds (descriptive, NOT PCR success/failure):")
        A(_threshold_lines(si))
        c = r["comparison"]
        A("COMPARISON (insulated minus bare):")
        A(f"  overall sample max delta      : {c['sample_max_increase_C']:+.3f} C")
        A(f"  sample mean delta             : {c['sample_mean_increase_C']:+.3f} C")
        A(f"  sample median delta           : {c['sample_median_increase_C']:+.3f} C")
        A(f"  max instantaneous delta       : {c['max_instantaneous_delta_C']:+.3f} C")
        A(f"  time of max delta             : {c['time_of_max_delta_s']:.3f} s")
    else:
        st = r["stats_bare"] if r["model"] == "bare" else r["stats_insulated"]
        A(f"internal maximum        : {st['internal_max_C']:.3f} C")
        A(f"sample maximum          : {st['sample_max_C']:.3f} C")
        A(f"time of sample maximum  : {st['sample_max_time_s']:.3f} s "
          f"(original experiment time)")
        A(f"internal at sample max  : {st['internal_at_sample_max_C']:.3f} C")
        A(f"sample minimum          : {st['sample_min_C']:.3f} C")
        A(f"sample mean             : {st['sample_mean_C']:.3f} C")
        A(f"sample median           : {st['sample_median_C']:.3f} C")
        A("Descriptive thresholds (NOT PCR success/failure criteria):")
        for th in DESCRIPTIVE_THRESHOLDS_C:
            A(f"  sample max >= {th:.0f} C  : "
              f"{'YES' if st['thresholds'][th] else 'NO'}")
    A("-" * 72)
    A("Scientific status:")
    A("  " + _model_status_line("bare", r.get("model_version")))
    if r["model"] in ("insulated", "both"):
        A("  " + _model_status_line("insulated", r.get("model_version")))
    A("  Sample temperature is model-predicted (several-degree accuracy), "
      "not directly measured.")
    A("Insulation assumption: sealed air layer treated as conduction-"
      "dominated; internal radiative exchange across the air gap is omitted, "
      "so the insulation resistance may be somewhat overestimated.")
    A("-" * 72)
    A("Authority: calibration 66C RMSE {:.4f} C; external validation "
      "RMSE {:} C".format(
          m["calib_66C_RMSE_C"],
          {k: f"{v:.4f}" for k, v in
           m["external_validation_RMSE_C"].items()}))
    path = out_dir / "sample_temperature_summary.txt"
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ============================================================
# 交互窗口 (复用 draggable_hlines.py)
# ============================================================

class SampleThresholdHLine(DraggableHLine):
    """复用 draggable_hlines.py 的 DraggableHLine, 附加时间戳感知统计。

    统计应用于预测样品温度 (self.y_data), 显示:
        threshold / Time sample >= threshold / first up-cross / last above /
        intervals。
    交互线为探索工具, 不写入模型配置。
    """

    def __init__(self, ax, y_init, x_data, y_data, color, label_prefix,
                 stats_xy=(0.02, 0.02)):
        self.stats_text = None
        self.stats_xy = stats_xy
        super().__init__(ax, y_init, x_data, y_data, color, label_prefix)
        self.stats_text = ax.text(
            stats_xy[0], stats_xy[1], "", transform=ax.transAxes,
            va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9))
        self._update_stats(float(y_init))

    def _update_stats(self, y_level):
        if self.stats_text is None:
            return
        s = compute_threshold_timing(self.x_data, self.y_data,
                                     float(y_level))
        self.stats_text.set_text(
            f"{self.label_prefix} — SAMPLE threshold "
            f"{s['threshold_C']:.1f} C\n"
            f"Time sample >= threshold: {s['time_above_s']:.2f} s\n"
            f"First up-cross: {s['first_up_cross_s']:.1f} s | "
            f"Last above: {s['last_above_s']:.1f} s | "
            f"Intervals: {s['n_intervals']}")

    def _update_intersections(self, y_level):
        super()._update_intersections(y_level)
        self._update_stats(y_level)


class BothSampleThresholdHLine(DraggableHLine):
    """both 模式: 一个公共可拖阈值, 同时显示 bare 与 insulated 的时长。

    继承 draggable_hlines.py 的 DraggableHLine (拖动/温度标签/交点基于
    bare 样品曲线), 统计文本显示:
        Threshold: X C
        Bare sample >= threshold: Y s
        Insulated sample >= threshold: Z s
    时间戳感知; 交互线为探索工具。
    """

    def __init__(self, ax, y_init, x_data, y_bare, y_ins, color,
                 label_prefix, stats_xy=(0.02, 0.02)):
        self.y_bare = np.asarray(y_bare, dtype=float)
        self.y_ins = np.asarray(y_ins, dtype=float)
        self.stats_text = None
        super().__init__(ax, y_init, x_data, self.y_bare, color,
                         label_prefix)
        self.stats_text = ax.text(
            stats_xy[0], stats_xy[1], "", transform=ax.transAxes,
            va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9))
        self._update_stats(float(y_init))

    def _update_stats(self, y_level):
        if self.stats_text is None:
            return
        tb = compute_threshold_timing(self.x_data, self.y_bare,
                                      float(y_level))
        ti = compute_threshold_timing(self.x_data, self.y_ins,
                                      float(y_level))
        self.stats_text.set_text(
            f"Threshold: {float(y_level):.1f} C\n"
            f"Bare sample >= threshold: {tb['time_above_s']:.2f} s\n"
            f"Insulated sample >= threshold: {ti['time_above_s']:.2f} s")

    def _update_intersections(self, y_level):
        super()._update_intersections(y_level)
        self._update_stats(y_level)


def launch_interactive(r, default_thresholds=(85.0, 92.0)):
    """打开交互窗口: 实测内部 + 预测样品 + 可拖动样品阈值线。

    单模式: 阈值线应用于该模式的预测样品;
    both 模式: 一个公共可拖阈值, 同时显示 bare 与 insulated 的时长。
    交互仅用于探索; 不修改任何已保存的静态预测 / 模型输出。
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(r["t_original"], r["T_internal"], color="#1f77b4", lw=1.4,
            label="Measured Internal Temperature")
    y_data = None
    lines = []
    if r["model"] == "both":
        ax.plot(r["t_original"], r["sample_bare"], color="#2ca02c", lw=1.8,
                label="Predicted Sample — Bare")
        ax.plot(r["t_original"], r["sample_insulated"], color="#d62728",
                lw=1.8, label="Predicted Sample — Air Insulated")
        ax.set_title("Frozen Thermal Model — Sample Temperature Prediction "
                     "(interactive, bare vs insulated)")
        for k, (th, color, label) in enumerate(zip(
                default_thresholds, ("darkorange", "purple"),
                ("Threshold", "Threshold 2"))):
            line = BothSampleThresholdHLine(
                ax, y_init=th, x_data=r["t_original"],
                y_bare=r["sample_bare"], y_ins=r["sample_insulated"],
                color=color, label_prefix=label,
                stats_xy=(0.02, 0.62 - 0.18 * k))
            lines.append(line)
    else:
        ax.plot(r["t_original"], r["sample_active"], color="#d62728", lw=1.8,
                label=f"Predicted Sample — {r['model'].title()}")
        ax.set_title("Frozen Thermal Model — Sample Temperature Prediction "
                     f"(interactive, {r['model']})")
        y_data = r["sample_active"]
        for k, (th, color, label) in enumerate(zip(
                default_thresholds, ("darkorange", "purple"),
                ("Threshold A", "Threshold B"))):
            line = SampleThresholdHLine(
                ax, y_init=th, x_data=r["t_original"],
                y_data=y_data, color=color,
                label_prefix=label, stats_xy=(0.02, 0.62 - 0.16 * k))
            lines.append(line)
    ax.set_xlabel("Original experiment time [s]")
    ax.set_ylabel("Temperature [C]")
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlim(r["t_original"][0],
                float(r["t_original"][-1]) * 1.06)
    ylo = float(min(np.min(r["T_internal"]), np.min(r["sample_active"])
                    if r["sample_active"] is not None
                    else np.min(r["sample_bare"]),
                    np.min(r["sample_insulated"])
                    if r["sample_insulated"] is not None
                    else np.min(r["sample_bare"])))
    yhi = float(max(np.max(r["T_internal"]), np.max(r["sample_active"])
                    if r["sample_active"] is not None
                    else np.max(r["sample_bare"]),
                    np.max(r["sample_insulated"])
                    if r["sample_insulated"] is not None
                    else np.max(r["sample_bare"])))
    ax.set_ylim(ylo - 5.0, yhi + 5.0)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()

    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        print("[interactive] 当前后端为 Agg (headless), 交互窗口不可用; "
              "交互逻辑已程序化测试, 请在 GUI 后端环境运行以显示窗口。")
    else:
        try:
            plt.show()
        except Exception as exc:  # noqa: BLE001 — headless GUI 失败非致命
            print(f"[interactive] GUI 打开失败 (headless?): {exc}")
    return fig, tuple(lines)


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="冻结热模型样品层温度预测 (正向预测, 不拟合; 默认 "
                    "FINAL_FROZEN_THERMAL_MODEL_V2)")
    p.add_argument("--input", required=True,
                   help="内部温度工作簿路径 (.xlsx)")
    p.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL,
                   help=f"热配置: {' | '.join(MODEL_CHOICES)} "
                        f"(默认 {DEFAULT_MODEL})")
    p.add_argument("--model-version", choices=MODEL_VERSION_CHOICES,
                   default=MODEL_VERSION_DEFAULT,
                   help=f"冻结模型版本: v2 (默认, FC-70 + no-PDMS) | "
                        f"v1 (历史 Oil + PDMS 复现)")
    p.add_argument("--start-s", type=float, default=None,
                   help="分析窗口起点 (工作簿 Time(s) 轴, 仅裁剪输出; "
                        "模拟始终从完整源迹线起点开始)")
    p.add_argument("--end-s", type=float, default=None,
                   help="模拟/分析终点 (工作簿 Time(s) 轴, 省略=末个有效时间)")
    p.add_argument("--output-dir", default=None,
                   help="输出目录 (省略 = sample_temperature_output/<stem>/"
                        "<model>/)")
    p.add_argument("--no-gui", action="store_true",
                   help="不启动交互窗口 (批处理/测试用)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.no_gui:
        matplotlib.use("Agg")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"输入工作簿不存在: {input_path}")

    out_dir = resolve_output_dir(input_path, args.output_dir, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    r = run_prediction(input_path, model=args.model,
                       start_s=args.start_s, end_s=args.end_s,
                       model_version=args.model_version)

    png, pdf = plot_static(r, out_dir)
    csv_path = write_csv(r, out_dir)
    summary_path = write_summary(r, out_dir)

    print(f"input       : {r['input_path']}")
    print(f"model       : {r['model']}")
    print(f"model version: {r['model_version']} "
          f"({r['model_meta']['id']})")
    print(f"simulation  : [{r['sim_start_s']:.3f} -> {r['sim_end_s']:.3f}] s "
          f"(full history preserved = {r['thermal_history_preserved']})")
    print(f"window      : [{r['window_start_s']:.3f}, "
          f"{r['window_end_s']:.3f}] s  (n={r['n_points']}, "
          f"duration={r['duration_s']:.3f} s)")
    print(f"environment : {r['environment_C']:.3f} C "
          f"({r['environment_source']})")
    if r["model"] == "both":
        print(f"sample max (bare)      : "
              f"{r['stats_bare']['sample_max_C']:.3f} C @ "
              f"{r['stats_bare']['sample_max_time_s']:.3f} s (original)")
        print(f"sample max (insulated) : "
              f"{r['stats_insulated']['sample_max_C']:.3f} C @ "
              f"{r['stats_insulated']['sample_max_time_s']:.3f} s (original)")
        print(f"delta sample max       : "
              f"{r['comparison']['sample_max_increase_C']:+.3f} C")
    else:
        st = r["stats_bare"] if r["model"] == "bare" else r["stats_insulated"]
        print(f"sample max             : {st['sample_max_C']:.3f} C @ "
              f"{st['sample_max_time_s']:.3f} s (original)")
    print(f"PNG         : {png}")
    print(f"PDF         : {pdf}")
    print(f"CSV         : {csv_path}")
    print(f"summary     : {summary_path}")

    if not args.no_gui:
        launch_interactive(r)

    return r


if __name__ == "__main__":
    main()
