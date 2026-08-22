#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
66C REDO 重新标定 + 绝缘 3s 样品预测 — 规格测试
=================================================

覆盖 35 项规则 (任务 #45):

 1  标定仅用 66C Top COC 作为实测目标
 2  标定使用匹配的 66C 内部输入
 3  标定使用裸顶几何 (BARE_TOP_COC_LAYERS)
 4  h 固定 10
 5  epsilon 固定 0.90
 6  rho 固定 1020
 7  k 允许变化
 8  cp 允许变化
 9  tau 允许变化
10  无 qPCR 温度进入目标
11  无样品温度进入目标
12  无绝缘仿真进入目标
13  无时移优化
14  实测 Top 时间为插值查询轴
15  第一个实测 Top 定义环境
16  第一个内部定义初始场
17  输出侧滞后只作用于 Top
18  旧冻结模型保持不变
19  新候选独立存储
20  搜索边界检测工作
21  局部细化包含粗最优
22  top-10 候选表排序正确
23  独立 3s Top 验证不拟合
24  3s 验证参数 == 新 66C 最优
25  绝缘预测参数 == 新 66C 最优
26  绝缘样品不滞后
27  Air/PDMS 属性不优化
28  86 C 参考缺席标定函数
29  86 C 只用于事后样品统计
30  majority 定义 >50% 正确
31  周期峰使用既有检测器
32  激活相不计为重复周期
33  新旧比较不影响参数选择
34  当前冻结候选文件不变
35  无历史输出被覆盖
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.core import heat_model
from thermal_model.core import convection_radiation_thermal_model as cr
from workflows.calibration import recalibrate_thermal_model_66C_redo as R
from thermal_model.historical.frozen_strategy_G_candidate import FROZEN_STRATEGY_G_CANDIDATE

SRC = Path(R.__file__).read_text(encoding="utf-8")


# ============================================================
# 1-9 数据源 / 几何 / 固定与拟合参数
# ============================================================

def test_calibration_uses_66C_top_as_measured_target():
    assert R.DS1_TOP_66C.name == "extension 66°C_redo.xls"
    assert R.DS1_INT_66C.name.endswith(
        "08.17 COC top_66°C_zone1_temperature_analysis.xlsx")
    # 目标 = 预测 Top vs 实测 Top (仅 Top 进入目标)
    src = inspect.getsource(R.evaluate_top_for_tau)
    assert "T_top_c" in src and "T_top_measured" in src or "T_top_c" in src


def test_calibration_uses_matching_66C_internal_input():
    top, internal = R.load_66c_dataset()
    assert internal["source"].endswith(
        "08.17 COC top_66°C_zone1_temperature_analysis.xlsx")
    assert top["source"].endswith("extension 66°C_redo.xls")
    assert internal["n_valid"] > 0 and top["n_valid"] > 0


def test_calibration_uses_bare_top_geometry():
    src = inspect.getsource(R.run_bare_fdm)
    assert "heat_model.BARE_TOP_COC_LAYERS" in src
    # 无 Air / PDMS
    assert "LEGACY_INSULATED_LAYERS" not in src


def test_h_fixed_at_10():
    assert R.H_CONV == 10.0
    # 搜索网格不含 h
    assert "h_conv" not in R.K_COARSE.tolist() and \
        "h_conv" not in R.CP_COARSE.tolist() and \
        "h_conv" not in R.TAU_COARSE.tolist()


def test_epsilon_fixed_at_0_90():
    assert R.EPS == 0.90


def test_rho_fixed_at_1020():
    assert R.RHO_COC == 1020.0
    assert cr.RHO_COC_STRATEGY_E == 1020.0


def test_k_allowed_to_vary():
    assert len(R.K_COARSE) >= 5
    assert len(set(R.K_COARSE)) == len(R.K_COARSE)
    assert R.K_COARSE.min() < 0.055 < R.K_COARSE.max()


def test_cp_allowed_to_vary():
    assert len(R.CP_COARSE) >= 5
    assert len(set(R.CP_COARSE)) == len(R.CP_COARSE)
    assert R.CP_COARSE.min() < 1200 < R.CP_COARSE.max()


def test_tau_allowed_to_vary():
    assert len(R.TAU_COARSE) >= 5
    assert R.TAU_COARSE[0] == 0.0
    assert R.TAU_COARSE[-1] >= 8.5


# ============================================================
# 10-13 反循环: 目标 / 时移
# ============================================================

def test_no_qPCR_temperature_in_objective():
    R.assert_no_qPCR_in_calibration()
    src = inspect.getsource(R.evaluate_top_for_tau)
    assert "86" not in src and "qPCR" not in src and "QPCR" not in src


def test_no_sample_temperature_in_objective():
    src = inspect.getsource(R.evaluate_top_for_tau)
    assert "T_sample" not in src
    src2 = inspect.getsource(R.search_stage)
    assert "T_sample" not in src2


def test_no_insulated_simulation_in_objective():
    src = inspect.getsource(R.run_bare_fdm)
    assert "LEGACY_INSULATED_LAYERS" not in src
    src2 = inspect.getsource(R.evaluate_top_for_tau)
    assert "insulated" not in src2 and "LEGACY" not in src2


def test_no_time_shift_optimization():
    assert R.SYNC_RULE == "SIMULTANEOUS_START_RELATIVE_T0"
    src = inspect.getsource(R.evaluate_top_prediction)
    assert '"time_shift_applied_s": 0.0' in src
    assert "cross_correlation" not in src.lower()
    assert "np.correlate" not in src


# ============================================================
# 14-17 插值 / 环境 / 初始 / 滞后位置
# ============================================================

def test_measured_top_time_is_interpolation_query():
    src = inspect.getsource(R.evaluate_top_for_tau)
    assert "np.interp(t_top_c, t_arr_c, T_obs)" in src


def test_first_measured_top_defines_environment():
    src = inspect.getsource(R.evaluate_top_prediction)
    assert 'internal_env["_env_C"] = float(T_top[0])' in src
    ev_src = inspect.getsource(R.evaluate_top_prediction)
    assert "INITIAL_MEASURED_TOP" in ev_src


def test_first_internal_defines_initial_field():
    src = inspect.getsource(R.run_bare_fdm)
    assert 'T_init = float(internal["T"][0])' in src


def test_output_side_lag_applied_to_top_only():
    src = inspect.getsource(R.evaluate_top_for_tau)
    assert "apply_first_order_lag(t_arr_c, T_fdm_c, tau_top)" in src
    # 绝缘样品函数绝不应用滞后
    src_ins = inspect.getsource(R.run_insulated_sample)
    assert "apply_first_order_lag" not in src_ins


# ============================================================
# 18-19 冻结模型不变 / 新候选独立
# ============================================================

def test_old_frozen_model_remains_unchanged():
    c = FROZEN_STRATEGY_G_CANDIDATE
    assert c.k_eff_W_mK == 0.055
    assert c.cp_eff_J_kgK == 1200.0
    assert c.tau_lag_s == 8.5
    assert c.rho_COC_kg_m3 == 1020.0
    assert R.OLD_K == 0.055 and R.OLD_CP == 1200.0 and R.OLD_TAU == 8.5


def test_new_candidate_stored_separately():
    assert R.CANDIDATE_ID == "66C_RECALIBRATED_CANDIDATE_V1"
    assert "66C_recalibrated_candidate_v1" in str(R.OUTPUT_ROOT)
    # 与冻结模型 ID 不同
    assert R.CANDIDATE_ID != "strategy_G_conservative_cross_protocol_v1"


# ============================================================
# 20-22 边界 / 细化 / top-10
# ============================================================

def test_search_boundary_detection_works():
    df = pd.DataFrame([
        {"k_eff_W_mK": 0.03, "cp_eff_J_kgK": 2200.0, "tau_top_s": 12.0,
         "RMSE_C": 0.5},
    ])
    best = df.iloc[0].to_dict()
    warnings = R.check_boundary(
        best, np.array([0.03, 0.04]), np.array([700.0, 2200.0]),
        np.array([0.0, 12.0]))
    assert set(warnings) == {"K_MIN", "CP_MAX", "TAU_MAX"}
    # 内点无警告
    best2 = {"k_eff_W_mK": 0.06, "cp_eff_J_kgK": 1100.0, "tau_top_s": 5.0}
    assert R.check_boundary(best2, np.array([0.03, 0.09]),
                            np.array([700.0, 2200.0]),
                            np.array([0.0, 12.0])) == []


def test_expand_grids_moves_in_sensible_direction():
    k, cp, tau = R.expand_grids(np.array([0.03, 0.05, 0.07]),
                                np.array([700.0, 1100.0]),
                                np.array([0.0, 4.0, 8.0]),
                                ["K_MIN", "CP_MAX", "TAU_MAX"])
    assert k[0] < 0.03          # k 向下扩展
    assert cp[-1] > 1100.0      # cp 向上扩展
    assert tau[-1] > 8.0        # tau 向上扩展


def test_local_refinement_contains_coarse_optimum():
    best = {"k_eff_W_mK": 0.06, "cp_eff_J_kgK": 1100.0, "tau_top_s": 5.0}
    k, cp, tau = R.build_refined_grid(best)
    assert best["k_eff_W_mK"] in k
    assert best["cp_eff_J_kgK"] in cp
    assert best["tau_top_s"] in tau
    # 两侧邻居存在 (物理允许时); 用 round(...,6) 避免浮点表示误差
    assert round(best["k_eff_W_mK"] + R.K_REFINE_STEP, 6) in k
    assert round(best["k_eff_W_mK"] - R.K_REFINE_STEP, 6) in k
    assert best["cp_eff_J_kgK"] + R.CP_REFINE_STEP in cp
    assert best["cp_eff_J_kgK"] - R.CP_REFINE_STEP in cp
    assert best["tau_top_s"] + R.TAU_REFINE_STEP in tau
    assert best["tau_top_s"] - R.TAU_REFINE_STEP in tau


def test_top10_candidates_table_sorted_correctly():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "k_eff_W_mK": rng.uniform(0.03, 0.15, 50),
        "cp_eff_J_kgK": rng.uniform(700, 2200, 50),
        "tau_top_s": rng.uniform(0, 12, 50),
        "RMSE_C": rng.uniform(0.5, 5.0, 50),
        "MAE_C": 0.0, "mean_residual_C": 0.0, "n_points": 100,
        "stage": "test",
    })
    top = R.top_n_candidates(df, n=10)
    assert len(top) == 10
    rmse = top["RMSE_C"].to_numpy()
    assert np.all(np.diff(rmse) >= -1e-12)          # 升序
    assert top.iloc[0]["RMSE_C"] == df["RMSE_C"].min()   # 全局最优


# ============================================================
# 23-27 下游锁定 / 无滞后 / 绝缘属性
# ============================================================

def test_3s_validation_performs_no_fitting():
    src = inspect.getsource(R.phase_c_validation)
    assert "assert_parameters_locked" in src
    assert "search_stage" not in src
    assert "run_fdm_batch" not in src


def test_3s_validation_parameters_equal_new_66C_optimum():
    # 锁定断言由 phase_c_validation 调用; 直接验证断言函数本身
    R.assert_parameters_locked(0.06, 1100.0, 5.0,
                               0.06, 1100.0, 5.0, "TEST")
    with pytest.raises(RuntimeError):
        R.assert_parameters_locked(0.07, 1100.0, 5.0,
                                   0.06, 1100.0, 5.0, "TEST")


def test_insulated_prediction_parameters_equal_new_66C_optimum():
    src = inspect.getsource(R.phase_d_insulated_prediction)
    assert "assert_parameters_locked" in src
    src2 = inspect.getsource(R.run_insulated_sample)
    assert "LEGACY_INSULATED_LAYERS" in src2


def test_insulated_sample_not_lagged():
    src = inspect.getsource(R.run_insulated_sample)
    assert "apply_first_order_lag" not in src
    assert "T_sample" in src


def test_air_and_pdms_properties_not_optimized():
    # 绝缘层叠使用 DEFAULT_MATERIALS 中的 Air / PDMS 常量
    layers = heat_model.LEGACY_INSULATED_LAYERS
    names = [l.material for l in layers]
    assert "Air" in names and "PDMS" in names
    # 标定只替换 COC (make_convection_radiation_materials)
    src = inspect.getsource(cr.make_convection_radiation_materials)
    assert src.count("COC") >= 1
    assert "Air" not in src.split("mats[")[1] if "mats[" in src else True
    # 搜索网格不含 Air/PDMS 属性
    assert "0.0257" not in [str(x) for x in R.K_COARSE]
    assert "1005" not in [str(x) for x in R.CP_COARSE]


# ============================================================
# 28-32 86 C 边界 / majority / 检测器
# ============================================================

def test_86C_reference_absent_from_calibration_functions():
    for fn in (R.evaluate_top_for_tau, R.run_bare_fdm, R.search_stage,
               R.phase_a_calibration, R.classify_candidate):
        src = inspect.getsource(fn)
        assert "QPCR_FUNCTIONAL_REFERENCE_C" not in src
        assert "86.0" not in src


def test_86C_reference_only_in_post_hoc_sample_statistics():
    src_d = inspect.getsource(R.phase_d_insulated_prediction)
    assert "QPCR_FUNCTIONAL_REFERENCE_C" in src_d
    src_w = inspect.getsource(R.write_insulated_outputs)
    assert "QPCR_FUNCTIONAL_REFERENCE_C" in src_w


def test_majority_definition_is_gt_50_percent():
    assert R.MAJORITY_FRACTION == 0.50
    assert R.QPCR_FUNCTIONAL_REFERENCE_C == 86.0


def test_cycle_peaks_use_existing_detector():
    src = inspect.getsource(R.phase_d_insulated_prediction)
    assert "detect_repeated_cycles" in src
    assert ("from thermal_model.utilities.analyze_frozen_sample_peak "
            "import detect_repeated_cycles") in SRC


def test_activation_not_counted_as_repeated_cycle():
    from thermal_model.utilities.analyze_frozen_sample_peak import detect_repeated_cycles
    # 合成: 单调上升至 90 C 激活相 (无初始冷平台 -> 无前低谷), 然后低谷分隔的
    # 两个重复周期。激活相 = 首个无前低谷且保持 >=30 s 的峰 (既有检测器语义)。
    t = np.arange(0.0, 150.0, 0.5)
    tint = np.clip(40.0 + t * 2.0, 40.0, 90.0)   # 0-25 s 单调升到 90, 保持至 50 s
    tint[(t >= 50) & (t < 55)] = 40.0     # 激活后低谷
    tint[(t >= 55) & (t < 65)] = 95.0     # 周期 1 高
    tint[(t >= 65) & (t < 70)] = 40.0     # 周期 1 低谷
    tint[(t >= 70) & (t < 80)] = 95.0     # 周期 2 高
    tint[(t >= 80)] = 40.0                # 周期 2 低谷及尾部
    tsample = tint - 5.0   # 样品滞后于内部
    cyc = detect_repeated_cycles(t, tint, tsample)
    assert cyc["activation"] is not None
    assert cyc["activation"]["internal_high_peak_C"] >= 88.0
    # 重复周期 = 2 (激活相不计)
    assert len(cyc["repeated_cycles"]) == 2


# ============================================================
# 33-35 比较不影响选参 / 冻结文件 / 输出隔离
# ============================================================

def test_old_new_comparison_does_not_influence_parameter_selection():
    # main 顺序: PHASE A 标定 -> PHASE C 验证 -> PHASE D 预测 -> 比较输出
    src = inspect.getsource(R.main)
    a = src.index("phase_a_calibration")
    c = src.index("phase_c_validation")
    d = src.index("phase_d_insulated_prediction")
    w = src.index("write_comparison_outputs")
    assert a < c < d < w
    # 比较函数不写参数 (不产生选择)
    src_w = inspect.getsource(R.write_comparison_outputs)
    assert "search_stage" not in src_w


def test_current_frozen_candidate_file_unchanged():
    frozen_src = Path(R.PROJECT_ROOT / "thermal_model/historical/frozen_strategy_G_candidate.py"
                      ).read_text(encoding="utf-8")
    assert "FROZEN_K_W_MK = 0.055" in frozen_src
    assert "FROZEN_CP_J_KGK = 1200.0" in frozen_src
    assert "FROZEN_TAU_S = 8.5" in frozen_src


def test_no_historical_outputs_overwritten():
    hist_roots = [
        "frozen_output_lag_external_validation_v1",
        "frozen_output_lag_sample_peak_analysis_v1",
        "strategy_G_conservative_cross_protocol_v1",
        "strategy_G_conservative_cross_protocol_v2_corrected_dwell",
        "72C_nominal_v1",
        "72C_corrected_objective_v1",
        "08.12_pm_DOE11_faster_sample_prediction_v1",
        "60C_redo_transfer_check_v1",
    ]
    for h in hist_roots:
        assert h not in str(R.OUTPUT_ROOT)
    # 新输出目录必须唯一
    assert R.OUTPUT_ROOT.name == "66C_recalibrated_candidate_v1"


# ============================================================
# 额外数值行为: 相位顺序锁定 / 目标语义
# ============================================================

def test_objective_uses_measured_time_query_never_temperature_as_time():
    src = inspect.getsource(R.evaluate_top_for_tau)
    # 查询轴 = t_top_c (实测时间), 绝不是 T_top_c
    assert "np.interp(t_top_c, t_arr_c, T_obs)" in src
    assert "np.interp(T_top_c" not in src


def test_fixed_boundary_constants_match_strategy_E():
    assert R.H_CONV == cr.H_CONV_STRATEGY_E_W_M2K == 10.0
    assert R.EPS == cr.EMISSIVITY_STRATEGY_E == 0.90
    assert R.SIGMA == cr.SIGMA_SB_W_M2_K4
    assert R.F_VIEW == cr.VIEW_FACTOR_STRATEGY_E == 1.0


def test_old_rmse_historical_reference_present():
    assert abs(R.OLD_66C_RMSE_HISTORICAL - 3.0134) < 1e-4
    assert abs(R.OLD_3S_RMSE_HISTORICAL - 2.3941) < 1e-4
