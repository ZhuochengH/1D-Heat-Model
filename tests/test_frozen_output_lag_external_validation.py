#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冻结裸顶模型 — 两个修正后同步数据集的外部验证测试 (从零重建)
===============================================================

背景:
    先前的外部验证基于未温度转换的 Top COC 源文件 (伪伪影, RMSE ~12-14 C),
    已整体作废并回滚。本测试针对重建后的验证脚本
    validate_frozen_model_two_new_bare_top_datasets.py (修正后 T Avg 数据,
    无伪影过滤)。

覆盖 26 项规则 (规格 #39):
 1-5  冻结参数精确值 (k/cp/tau/h/epsilon)
 6    裸顶几何 (BARE_TOP_COC_LAYERS, 无 Air/PDMS)
 7    输出侧滞后 (apply_first_order_lag 仅作用于顶部观测)
 8    内部迹线直接 Dirichlet 底部边界
 9    环境 = 第一个修正后实测 Top (INITIAL_MEASURED_TOP)
10    初始 = 第一个实测内部温度
11    相对 t0 同步 (SIMULTANEOUS_START_RELATIVE_T0)
12    时移 = 0.0 (无附加时间校正)
13    无交叉相关
14    无时移优化
15    无模型伪影过滤 (仅结构性无效排除)
16    无中值过滤
17    无尖峰过滤
18    主 RMSE 基于修正后原始 Top (residual = pred - measured)
19    Top 实测时间为插值查询轴
20    温度永不作为插值坐标
21    样品温度缺席验证目标
22    PCR 结果缺席
23    无参数拟合
24    两数据集同冻结参数
25    无效旧输出不作为输入
26    校准 RMSE 只读参考

测试尽量以源码/结构断言为主, 少量合成数据行为验证;
不依赖无效旧指标的数值。
"""
from pathlib import Path

import numpy as np
import pytest

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from thermal_model.utilities import validate_frozen_model_two_new_bare_top_datasets as vmod
from thermal_model.historical.frozen_strategy_G_candidate import FROZEN_STRATEGY_G_CANDIDATE

SRC = Path(vmod.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------
# 合成数据 (行为测试, 不依赖真实文件)
# ---------------------------------------------------------------

def _synthetic_top():
    t = np.linspace(0.0, 200.0, 101)
    T = 25.0 + 30.0 * (1.0 - np.exp(-t / 60.0))
    return {"t_rel": t, "T": T}


def _synthetic_internal():
    t = np.linspace(0.0, 200.0, 201)
    T = 24.0 + 35.0 * (1.0 - np.exp(-t / 50.0))
    return {"t_rel": t, "T": T}


@pytest.fixture(scope="module")
def synth_run():
    return vmod.run_frozen_validation(_synthetic_top(),
                                      _synthetic_internal())


# ---------------------------------------------------------------
# 1-5. 冻结参数精确值
# ---------------------------------------------------------------

def test_frozen_k_eff_exactly_0p055():
    assert vmod.K_EFF == 0.055
    assert FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK == 0.055


def test_frozen_cp_eff_exactly_1200():
    assert vmod.CP_EFF == 1200.0
    assert FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK == 1200.0


def test_frozen_tau_top_exactly_8p5():
    assert vmod.TAU_TOP == 8.5
    assert FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s == 8.5


def test_frozen_h_conv_exactly_10():
    assert vmod.H_CONV == 10.0
    assert cr.H_CONV_STRATEGY_E_W_M2K == 10.0


def test_frozen_epsilon_exactly_0p90():
    assert vmod.EPS == 0.90
    assert cr.EMISSIVITY_STRATEGY_E == 0.90


# ---------------------------------------------------------------
# 6. 裸顶几何
# ---------------------------------------------------------------

def test_bare_top_geometry_no_insulation():
    layers = heat_model.BARE_TOP_COC_LAYERS
    assert len(layers) == 4
    mats = {l.material for l in layers}
    assert "Air" not in mats
    assert "PDMS" not in mats
    total = sum(l.thickness_m for l in layers)
    assert total == pytest.approx(850e-6, abs=0.0)
    assert layers[-1].role == "top_surface"
    assert layers[-1].material == "COC"


# ---------------------------------------------------------------
# 7-12. 模型架构 / 同步规则
# ---------------------------------------------------------------

def test_output_side_lag_only_top_observation(synth_run):
    assert "apply_first_order_lag(t_arr, T_top_fdm, TAU_TOP)" in SRC
    v = synth_run
    assert len(v["T_top_predicted_lagged"]) == len(v["T_top_measured"])
    # 动态输入下, 滞后输出应与 FDM 原始不同 (tau > 0)
    assert not np.allclose(v["T_top_predicted_lagged"],
                           v["T_top_fdm_raw"])


def test_internal_trace_direct_dirichlet_bottom():
    assert "time_s=t_int" in SRC
    assert "bottom_temperature_C=T_int" in SRC
    # 底部边界来自实测内部迹线, 不是常数
    assert "bottom_temperature_C=T_int" in SRC
    assert "T_initial_C=T_init" in SRC


def test_environment_from_first_measured_top(synth_run):
    v = synth_run
    top = _synthetic_top()
    assert v["t_env_C"] == pytest.approx(float(top["T"][0]), abs=0.0)
    assert v["environment_source"] == "INITIAL_MEASURED_TOP"


def test_initial_condition_from_first_internal(synth_run):
    v = synth_run
    internal = _synthetic_internal()
    assert v["T_initial_C"] == pytest.approx(float(internal["T"][0]),
                                             abs=0.0)


def test_relative_t0_simultaneous_start(synth_run):
    assert vmod.run_frozen_validation.__doc__ is not None
    assert "SIMULTANEOUS_START_RELATIVE_T0" in SRC
    v = synth_run
    assert v["synchronization_rule"] == "SIMULTANEOUS_START_RELATIVE_T0"


def test_time_shift_zero(synth_run):
    v = synth_run
    assert v["time_shift_applied_s"] == 0.0
    assert '"time_shift_applied_s": 0.0' in SRC


# ---------------------------------------------------------------
# 13-17. 无伪影过滤 / 无优化
# ---------------------------------------------------------------

def test_no_cross_correlation():
    assert "np.correlate" not in SRC
    assert "correlate(" not in SRC


def test_no_time_shift_optimization():
    assert "np.correlate" not in SRC
    assert "argmax" not in SRC
    assert "time_shift_applied_s" in SRC  # 只作为固定 0.0 记录


def test_no_model_based_artifact_filtering():
    # load_top_series 的掩码只做结构性无效排除
    assert "np.isfinite(t_abs) & np.isfinite(T_avg) & (T_avg > 0.0)" in SRC
    # 明确无剔除模型分歧点
    assert "no_model_based_point_rejection" in SRC
    # 主残差计算前不做任何逐点剔除 (n == len(resid))
    assert "n = len(resid)" in SRC


def test_no_median_filter():
    assert "medfilt" not in SRC
    assert "scipy.signal" not in SRC
    # 唯一 rolling 平滑仅用于 regime 诊断分类 (斜率平滑), 不参与数据过滤
    assert "_regime_labels" in SRC
    # 检查 rolling 调用行只出现在 _regime_labels 诊断函数内
    diag_start = SRC.index("def _regime_labels")
    diag_end = SRC.index("def process_dataset")
    diag_src = SRC[diag_start:diag_end]
    assert "rolling(window" in diag_src
    assert "rolling(window" not in SRC.replace(diag_src, "")


def test_no_spike_filter():
    # spike 只作为"不做过滤"的说明出现, 绝无 spike 检测/剔除实现
    assert "no_artifact_filtering" in SRC
    assert "detect_spike" not in SRC
    assert "remove_spike" not in SRC
    assert "is_spike" not in SRC
    assert "spike_removed" not in SRC


# ---------------------------------------------------------------
# 18-20. 主验证指标 / 插值规则
# ---------------------------------------------------------------

def test_primary_rmse_from_corrected_raw_top(synth_run):
    v = synth_run
    m = v["metrics"]
    resid = v["residual"]
    assert np.allclose(resid,
                       v["T_top_predicted_lagged"] - v["T_top_measured"])
    assert m["RMSE_C"] == pytest.approx(
        float(np.sqrt(np.mean(resid ** 2))), rel=1e-9)
    assert m["MAE_C"] == pytest.approx(float(np.mean(np.abs(resid))),
                                       rel=1e-9)
    assert m["n_points"] == len(resid)
    assert "R_squared" in m


def test_top_time_is_interpolation_query_axis(synth_run):
    # 预测插值到实测 Top 时间: np.interp(实测时间, 模型时间, 模型输出)
    assert "np.interp(t_top_c, t_arr_c, T_obs_c)" in SRC
    v = synth_run
    assert len(v["T_top_predicted_lagged"]) == len(v["t_top"])


def test_temperature_never_interpolation_coordinate():
    # 查询轴必须是时间, 绝不允许温度作为插值坐标
    assert "np.interp(T" not in SRC
    assert "np.interp(t_top_c, t_arr_c, T_obs_c)" in SRC


# ---------------------------------------------------------------
# 21-22. 样品温度 / PCR 结果缺席
# ---------------------------------------------------------------

def test_sample_temperature_absent_from_targets():
    # predict_sample 仅作为 import 行 (导入内部温度加载器), 绝无样品预测
    assert "from thermal_model.utilities.predict_sample_from_internal_temperature" \
        " import load_internal_data" in SRC
    assert "predict_sample(" not in SRC
    assert "sample_prediction" not in SRC
    assert "sample_peak" not in SRC
    assert "sample_high_peak" not in SRC


def test_pcr_outcome_absent():
    assert "pcr_outcome" not in SRC
    assert "cycle" not in SRC.lower()
    # 验证不评判 PCR 扩增结果
    assert "sample/PCR outcome not used" in SRC


# ---------------------------------------------------------------
# 23-26. 无拟合 / 同参数 / 旧输出 / 校准参考
# ---------------------------------------------------------------

def test_no_parameter_fitting():
    assert "curve_fit" not in SRC
    assert "minimize" not in SRC
    assert "least_squares" not in SRC
    assert "scipy.optimize" not in SRC
    assert "grid_search" not in SRC.lower()


def test_same_frozen_parameters_both_datasets():
    # 两数据集走同一冻结常量 (无每数据集差异化)
    assert "process_dataset(\"66C_redo\", DS1_TOP, DS1_INT, out1)" in SRC
    assert "process_dataset(\"PCR_3s_extension\", DS2_TOP, DS2_INT, out2)" \
        in SRC
    assert vmod.K_EFF == FROZEN_STRATEGY_G_CANDIDATE.k_eff_W_mK
    assert vmod.CP_EFF == FROZEN_STRATEGY_G_CANDIDATE.cp_eff_J_kgK
    assert vmod.TAU_TOP == FROZEN_STRATEGY_G_CANDIDATE.tau_lag_s


def test_invalidated_old_output_not_input():
    # 脚本只写输出目录, 不读旧输出
    assert "read_csv" not in SRC
    assert "read_excel" in SRC  # 仅读取源 Top/内部 Excel
    assert "to_csv" in SRC
    # 输出目录完全重建 (非增量)
    assert "OUTPUT_ROOT / \"comparison\"" in SRC


def test_calibration_rmse_readonly_reference():
    assert vmod.CALIB_RMSE_72C == pytest.approx(0.8891597125869538,
                                                abs=0.0)
    # 校准 RMSE 只作参考行 (role=CALIBRATION), 不参与验证计算
    assert '"role": "CALIBRATION"' in SRC
    # 校准 RMSE 用于只读参考图/文本
    assert "calibration_reference" in SRC
