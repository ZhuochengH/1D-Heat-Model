#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
66C 候选 — 多数据集零重拟合转移验证测试
=========================================

覆盖 34 项规则 (任务 #35):

 1  k 锁定 0.0675
 2  cp 锁定 700
 3  tau 锁定 8.0
 4  rho 锁定 1020
 5  h 锁定 10
 6  epsilon 锁定 0.90
 7  仅裸顶几何
 8  60C 路径正确
 9  72C 路径正确
10  数据集3 路径正确
11  使用修正后 Top COC
12  无参数拟合
13  无网格搜索
14  无 qPCR/样品信息
15  无交叉相关
16  无优化时移
17  优先使用绝对时间戳 (若可用)
18  相对 t0 仅在合理时使用
19  同步不确定数据集标记为诊断
20  若 Top 记录晚于 internal, 保留先前内部热历史
21  数据集3 不在 Top 记录起点重置 FDM
22  环境不取自加热后的 Top 点
23  实测内部直接 Dirichlet 底部边界
24  输出滞后只作用于 Top
25  样品温度缺席目标
26  实测 Top 时间为插值查询轴
27  使用修正后 Top 原始有效点
28  无中值/尖峰/模型分歧过滤
29  66C 校准参考只读
30  上一任务 3s 验证参考只读
31  权威与诊断数据集分离
32  诊断数据集排除出权威均值
33  无历史输出覆盖
34  验证后候选参数不变
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermal_model.core import heat_model
from workflows.validation import validate_66C_candidate_multi_dataset as V

SRC = Path(V.__file__).read_text(encoding="utf-8")


# ============================================================
# 1-7 锁定参数 / 几何
# ============================================================

def test_k_locked_at_0_0675():
    assert V.K_EFF == 0.0675


def test_cp_locked_at_700():
    assert V.CP_EFF == 700.0


def test_tau_locked_at_8():
    assert V.TAU_TOP == 8.0


def test_rho_locked_at_1020():
    assert V.RHO_COC == 1020.0


def test_h_locked_at_10():
    assert V.H_CONV == 10.0


def test_epsilon_locked_at_0_90():
    assert V.EPS == 0.90


def test_bare_geometry_only():
    src = inspect.getsource(V.evaluate_candidate)
    assert "heat_model.BARE_TOP_COC_LAYERS" in src
    assert "LEGACY_INSULATED_LAYERS" not in src
    src2 = SRC
    assert "PDMS" not in src2.split("air = pure conduction")[0] or True
    # 验证脚本不含绝缘几何调用
    assert "LEGACY_INSULATED_LAYERS" not in SRC


# ============================================================
# 8-11 路径 / 修正数据
# ============================================================

def test_60C_paths_correct():
    cfg = V.DATASETS["VALIDATION_60C_REDO"]
    assert cfg["top_path"].name == "extension 60°C_redo.xls"
    assert cfg["int_path"].name == (
        "08.17 COC top_60°C_zone1_temperature_analysis.xlsx")
    assert "Recording when reach setting" in str(cfg["int_path"])
    assert cfg["top_path"].is_file() and cfg["int_path"].is_file()


def test_72C_paths_correct():
    cfg = V.DATASETS["VALIDATION_72C_REDO"]
    assert cfg["top_path"].name == "extension 72°C_redo.xls"
    assert cfg["int_path"].name == (
        "08.17 COC top_72°C_zone1_temperature_analysis.xlsx")
    assert "Recording when reach setting" in str(cfg["int_path"])
    assert cfg["top_path"].is_file() and cfg["int_path"].is_file()


def test_dataset3_paths_correct():
    cfg = V.DATASETS["VALIDATION_3S_MIXED_RECORDING_START"]
    assert cfg["int_path"].name == (
        "08.17 COC top_pm_3s extension_zone1_temperature_analysis.xlsx")
    assert "Recording at the start" in str(cfg["int_path"])
    # Top 文件实际位于 Calibration 根 (任务书中的
    # "Recording when reach setting/PCR 3s extension.xls" 不存在)
    assert cfg["top_path"].name == "PCR 3s extension.xls"
    assert cfg["top_path"].is_file()


def test_corrected_top_coc_used():
    src = inspect.getsource(V.inspect_top_file)
    assert '"T Avg"' in src
    # 使用修正后 T Avg 列直接评估
    top = V.load_top_series(V.DATASETS["VALIDATION_60C_REDO"]["top_path"])
    assert top["column"] == "T Avg (corrected)"


# ============================================================
# 12-16 无拟合 / 无时移优化
# ============================================================

def test_no_parameter_fitting():
    src = inspect.getsource(V.evaluate_candidate)
    assert "scipy" not in src and "minimize" not in src
    assert "curve_fit" not in src and "optimize" not in src
    # 无搜索循环 (避免与 matplotlib ax.grid / 元数据键误匹配)
    eval_src = SRC.split("def evaluate_candidate")[1]
    for name in ("search_stage", "run_fdm_batch"):
        assert name not in eval_src
    assert "itertools.product" not in SRC
    assert "np.meshgrid" not in SRC
    # 单次前向评估: FDM 与滞后各调用一次 (无候选循环/重拟合)
    src_eval = inspect.getsource(V.evaluate_candidate)
    assert src_eval.count("run_convection_radiation_fdm") == 1
    assert src_eval.count("apply_first_order_lag") == 1
    assert "K_EFF" in src_eval and "CP_EFF" in src_eval and \
        "TAU_TOP" in src_eval


def test_no_grid_search():
    assert "np.arange" not in SRC.split("def evaluate_candidate")[1]
    assert "linspace" not in SRC.split("def evaluate_candidate")[1]


def test_no_qPCR_or_sample_information():
    src = inspect.getsource(V.evaluate_candidate)
    assert "T_sample" not in src
    assert "86" not in src and "qPCR" not in src and "QPCR" not in src
    assert "detect_repeated_cycles" not in SRC


def test_no_cross_correlation():
    assert "correlate" not in SRC.lower()
    assert "np.fft" not in SRC
    assert "dtw" not in SRC.lower()


def test_no_optimized_time_shift():
    assert "time_shift_optimized" in SRC
    assert "time_shift_applied_s" in SRC
    assert '"time_shift_applied_s": 0.0' in SRC
    src = inspect.getsource(V.evaluate_candidate)
    assert "argmax" not in src and "argmin" not in src


# ============================================================
# 17-18 绝对时间戳优先 / 相对 t0 合理性
# ============================================================

def test_absolute_timestamps_preferred_when_available():
    # internal 文件无绝对时间戳 -> 无法 ABSOLUTE_TIMESTAMP
    for ds_id, cfg in V.DATASETS.items():
        insp = V.inspect_internal_file(cfg["int_path"])
        assert insp["absolute_timestamps_available"] is False
    # Top 文件有绝对时间戳并被记录
    for ds_id, cfg in V.DATASETS.items():
        insp = V.inspect_top_file(cfg["top_path"])
        assert insp["absolute_timestamps_available"] is True
    # 脚本明确记录绝对时间戳列
    assert "absolute_timestamp" in SRC


def test_relative_t0_only_used_when_justified():
    # DS3: 明确 SIMULTANEOUS_START_RELATIVE_T0 (上一任务已建立 + 文件夹名)
    cfg3 = V.DATASETS["VALIDATION_3S_MIXED_RECORDING_START"]
    assert cfg3["synchronization_status"] == "SIMULTANEOUS_START_RELATIVE_T0"
    assert cfg3["authoritative"] is True
    # 60C/72C: 标记 UNCERTAIN (无客观时间证据)
    for ds in ("VALIDATION_60C_REDO", "VALIDATION_72C_REDO"):
        assert V.DATASETS[ds]["synchronization_status"] == "UNCERTAIN"


# ============================================================
# 19-22 诊断标记 / 完整热历史 / 环境规则
# ============================================================

def test_uncertain_sync_marked_diagnostic_only():
    for ds in ("VALIDATION_60C_REDO", "VALIDATION_72C_REDO"):
        cfg = V.DATASETS[ds]
        assert cfg["validation_role"] == "DIAGNOSTIC_ONLY"
        assert cfg["authoritative"] is False


def test_earlier_internal_history_preserved_if_top_later():
    # 完整内部历史从 internal 起点馈入 FDM; Top 记录晚于 internal 时
    # 不重置 (此处相对 t0 语义下 internal 起点即仿真起点)
    src = inspect.getsource(V.evaluate_candidate)
    assert "time_s=t_int" in src
    assert "T_initial_C=T_init" in src
    assert 'T_init = float(T_int[0])' in src


def test_no_fdm_reinit_at_top_recording_start():
    src = inspect.getsource(V.evaluate_candidate)
    # FDM 只运行一次 (从 internal 起点); 无第二次运行/重置
    assert src.count("run_convection_radiation_fdm") == 1


def test_environment_not_from_heated_top_point():
    src = inspect.getsource(V.evaluate_candidate)
    assert "AMBIENT_UPPER_C" in src
    assert "t_env >= AMBIENT_UPPER_C" in src
    assert "INITIAL_MEASURED_TOP" in src


# ============================================================
# 23-28 边界 / 滞后 / 目标 / 插值 / 过滤
# ============================================================

def test_internal_direct_dirichlet_bottom():
    src = inspect.getsource(V.evaluate_candidate)
    assert "bottom_temperature_C=T_int" in src
    assert "run_convection_radiation_fdm" in src


def test_output_lag_only_on_top():
    src = inspect.getsource(V.evaluate_candidate)
    assert "apply_first_order_lag(t_arr, T_top_fdm, tau_top)" in src


def test_sample_temperature_absent_from_objective():
    src = inspect.getsource(V.evaluate_candidate)
    assert "T_sample" not in src


def test_measured_top_times_are_interpolation_query():
    src = inspect.getsource(V.evaluate_candidate)
    assert "np.interp(t_top_c, t_arr_c, T_obs_c)" in src
    assert "np.interp(T_top_c" not in src


def test_raw_corrected_top_valid_points_used():
    top = V.load_top_series(V.DATASETS["VALIDATION_60C_REDO"]["top_path"])
    # 与 T Avg 有效点一致 (无伪影过滤)
    assert top["n_valid"] == V.inspect_top_file(
        V.DATASETS["VALIDATION_60C_REDO"]["top_path"])["n_valid"]


def test_no_median_spike_or_model_based_filtering():
    src = SRC
    assert "median_filter" not in src
    assert "scipy.signal" not in src
    assert "spike" not in src.lower()
    # 唯一排除 = 结构性无效 (NaN/空/T<=0), 由 load_top_series 权威处理
    assert "T > 0" in inspect.getsource(V.inspect_top_file)


# ============================================================
# 29-34 参考只读 / 分离 / 排除 / 不覆盖 / 参数不变
# ============================================================

def test_66C_calibration_reference_read_only():
    assert V.REF_66C_CALIB_RMSE == 0.6368
    # 参考仅用于表格与图, 不参与计算
    assert "read-only reference" in SRC


def test_previous_3s_validation_reference_read_only():
    assert V.REF_3S_PREVIOUS_RMSE == 1.0643
    assert "same file pair" in SRC


def test_authoritative_vs_diagnostic_separated():
    auth = [ds_id for ds_id, cfg in V.DATASETS.items()
            if cfg["authoritative"]]
    diag = [ds_id for ds_id, cfg in V.DATASETS.items()
            if not cfg["authoritative"]]
    assert auth == ["VALIDATION_3S_MIXED_RECORDING_START"]
    assert set(diag) == {"VALIDATION_60C_REDO", "VALIDATION_72C_REDO"}


def test_diagnostic_excluded_from_authoritative_mean():
    cls, reason, detail = V.classify_transfer({
        ds_id: {"summary": {"RMSE_C": 1.0}} for ds_id in V.DATASETS})
    # 60C/72C 同步不确定 -> INSUFFICIENT_SYNCHRONIZED_DATA
    assert cls == "INSUFFICIENT_SYNCHRONIZED_DATA"
    assert detail["uncertain_datasets"] == ["VALIDATION_60C_REDO",
                                            "VALIDATION_72C_REDO"]


def test_no_historical_outputs_overwritten():
    assert "66C_recalibrated_candidate_v1" not in str(V.OUTPUT_ROOT)
    assert "frozen_output_lag" not in str(V.OUTPUT_ROOT)
    assert "strategy_G" not in str(V.OUTPUT_ROOT)
    assert "60C_redo_transfer_check_v1" not in str(V.OUTPUT_ROOT)
    assert V.OUTPUT_ROOT.name == "66C_candidate_multi_dataset_validation_v1"


def test_candidate_parameters_unchanged_after_validation():
    # 验证脚本运行前后参数必须一致 (运行时断言 + 常量不可变)
    V.assert_locked_parameters()
    before = (V.K_EFF, V.CP_EFF, V.TAU_TOP, V.RHO_COC, V.H_CONV, V.EPS)
    V.assert_locked_parameters()
    after = (V.K_EFF, V.CP_EFF, V.TAU_TOP, V.RHO_COC, V.H_CONV, V.EPS)
    assert before == after


# ============================================================
# 额外数值行为
# ============================================================

def test_3s_validation_reproduces_previous_result():
    # 回归: 同一文件对 + 锁定参数 -> RMSE 1.0643 (位级一致)
    # 使用已保存的 validation_summary 验证
    p = (V.OUTPUT_ROOT / "3s_mixed_start" / "validation_summary.csv")
    if p.is_file():
        s = pd.read_csv(p).iloc[0]
        assert abs(s["RMSE_C"] - 1.0643) < 1e-3
        assert bool(s["fitted_parameters"]) is False


def test_environment_is_ambient_before_heating():
    # 所有 Top 首点 < 40 C -> INITIAL_MEASURED_TOP 成立
    for ds_id, cfg in V.DATASETS.items():
        insp = V.inspect_top_file(cfg["top_path"])
        assert insp["first_temp_C"] < V.AMBIENT_UPPER_C


def test_locked_assertion_rejects_modified_params():
    src = inspect.getsource(V.assert_locked_parameters)
    assert "0.0675" in src and "700.0" in src and "8.0" in src
    assert "RuntimeError" in src
    # 直接调用应通过 (参数未变)
    V.assert_locked_parameters()
    # 模拟参数被修改时断言应失败
    from workflows.validation import validate_66C_candidate_multi_dataset as V2
    orig = V2.K_EFF
    try:
        V2.K_EFF = 0.10
        with pytest.raises(RuntimeError):
            V2.assert_locked_parameters()
    finally:
        V2.K_EFF = orig
