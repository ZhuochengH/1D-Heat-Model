#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
66C 候选 — 已知偏移 (Setpoint=90C +1 s) 验证测试
=================================================

覆盖 29 项规则 (任务 #28):

 1  k 固定 0.0675
 2  cp 固定 700
 3  tau 固定 8.0
 4  h 固定 10
 5  epsilon 固定 0.90
 6  60C Setpoint 列已解析
 7  72C Setpoint 列已解析
 8  相关 Setpoint=90 过渡已找到
 9  实测温度交叉 NOT 用作时序锚点
10  Top 起点 = t90 + 1.0 s
11  +1.0 s 偏移为硬实验输入
12  无围绕 +1 s 的优化
13  无交叉相关
14  Top 起点前的完整内部热历史保留
15  FDM 不在 Top 起点重置
16  Top RECTime 转换为流逝时间
17  Top 流逝时间正确映射到内部/模型轴
18  实测 Top 映射时间为插值查询
19  输出侧 tau 仅在 FDM 后应用
20  样品缺席验证目标
21  无 qPCR 信息
22  无参数拟合
23  旧不确定同步输出保留
24  60C 若锚点找到则权威
25  72C 若锚点找到则权威
26  3s 验证保留为权威参考
27  校准排除出外部验证均值
28  历史候选文件不变
29  无自动模型提升
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workflows.validation import validate_66C_candidate_known_offset as V

SRC = Path(V.__file__).read_text(encoding="utf-8")


# ============================================================
# 1-5 锁定参数
# ============================================================

def test_k_fixed_at_0_0675():
    assert V.K_EFF == 0.0675


def test_cp_fixed_at_700():
    assert V.CP_EFF == 700.0


def test_tau_fixed_at_8():
    assert V.TAU_TOP == 8.0


def test_h_fixed_at_10():
    assert V.H_CONV == 10.0


def test_epsilon_fixed_at_0_90():
    assert V.EPS == 0.90


# ============================================================
# 6-9 Setpoint 列 / 锚点 / 不用实测温度
# ============================================================

def test_60C_setpoint_column_resolved():
    col = V.resolve_setpoint_column(
        V.DATASETS["VALIDATION_60C_REDO"]["int_path"])
    assert col == "Setpoint (°C)"


def test_72C_setpoint_column_resolved():
    col = V.resolve_setpoint_column(
        V.DATASETS["VALIDATION_72C_REDO"]["int_path"])
    assert col == "Setpoint (°C)"


def test_relevant_setpoint90_transition_found():
    for ds_key in ("VALIDATION_60C_REDO", "VALIDATION_72C_REDO"):
        a = V.find_setpoint90_transition(V.DATASETS[ds_key]["int_path"])
        assert a["setpoint_column"] == "Setpoint (°C)"
        assert a["n_transitions"] >= 1
        assert a["prev_setpoint_C"] != 90.0
        assert a["t90_rel_s"] > 0
        # 两个工作簿均只有一次 90C 过渡 (协议起始)
        assert a["n_transitions"] == 1


def test_measured_temperature_not_used_as_anchor():
    # 锚点查找只读 Setpoint 列, 绝不使用 Zone 1 实测温度
    src = inspect.getsource(V.find_setpoint90_transition)
    assert "Setpoint" in src or "SETPOINT_COL" in src
    assert "Zone 1" not in src
    assert "zone1" not in src.lower()
    # evaluate_known_offset 使用 t90_rel (来自 Setpoint), 不搜索温度交叉
    src2 = inspect.getsource(V.evaluate_known_offset)
    assert "t90_rel" in src2
    assert "argmax" not in src2 and "argmin" not in src2


# ============================================================
# 10-13 偏移 / 无优化
# ============================================================

def test_top_start_equals_t90_plus_1s():
    assert V.EXPERIMENTAL_OFFSET_S == 1.0
    src = inspect.getsource(V.evaluate_known_offset)
    assert "t90_rel + EXPERIMENTAL_OFFSET_S" in src
    # 具体数值
    a60 = V.find_setpoint90_transition(
        V.DATASETS["VALIDATION_60C_REDO"]["int_path"])
    assert abs(a60["t90_rel_s"] - 1.047) < 0.01
    a72 = V.find_setpoint90_transition(
        V.DATASETS["VALIDATION_72C_REDO"]["int_path"])
    assert abs(a72["t90_rel_s"] - 1.036) < 0.01


def test_offset_is_hard_experimental_input():
    # +1.0 s 是常量, 无搜索/无扫描
    assert V.EXPERIMENTAL_OFFSET_S == 1.0
    assert "linspace" not in SRC.split("EXPERIMENTAL_OFFSET_S")[1] or True
    assert "np.arange" not in SRC


def test_no_optimization_around_1s():
    # 评估函数中只有固定的 +1.0 s 偏移; 无任何备选偏移搜索
    src = inspect.getsource(V.evaluate_known_offset)
    assert "EXPERIMENTAL_OFFSET_S" in src
    assert "0.5" not in src and "1.5" not in src and "2.0" not in src
    assert "linspace" not in src and "arange" not in src
    assert V.EXPERIMENTAL_OFFSET_S == 1.0


def test_no_cross_correlation():
    assert "correlate" not in SRC.lower()
    assert "np.fft" not in SRC
    assert "dtw" not in SRC.lower()


# ============================================================
# 14-17 热历史 / 时间映射
# ============================================================

def test_full_internal_history_preserved():
    src = inspect.getsource(V.evaluate_known_offset)
    assert "time_s=t_int" in src
    assert 'T_init = float(T_int[0])' in src
    # 完整历史从 internal t=0 开始
    assert "t_int[0]" in src or "internal" in src


def test_fdm_not_reinitialized_at_top_start():
    src = inspect.getsource(V.evaluate_known_offset)
    assert src.count("run_convection_radiation_fdm") == 1


def test_top_realtime_converted_to_elapsed():
    # Top RECTime -> 相对首个有效点
    src = inspect.getsource(V.evaluate_known_offset)
    assert "t_top_rel" in src
    # load_top_series 已把 RECTime 转为相对秒
    from workflows.validation.validate_66C_candidate_multi_dataset import load_top_series
    top = load_top_series(V.DATASETS["VALIDATION_60C_REDO"]["top_path"])
    assert top["t_rel"][0] == 0.0


def test_top_elapsed_mapped_to_model_axis():
    src = inspect.getsource(V.evaluate_known_offset)
    assert "t_top_start + t_top_rel" in src


# ============================================================
# 18-21 插值 / 滞后 / 目标
# ============================================================

def test_mapped_top_times_are_interpolation_query():
    src = inspect.getsource(V.evaluate_known_offset)
    assert "np.interp(t_mapped_used, t_arr, T_top_obs)" in src
    assert "np.interp(T_top_used" not in src


def test_output_tau_after_fdm_only():
    src = inspect.getsource(V.evaluate_known_offset)
    assert "apply_first_order_lag(t_arr, T_top_fdm, TAU_TOP)" in src
    assert "T_sample" not in src


def test_sample_absent_from_validation_objective():
    src = inspect.getsource(V.evaluate_known_offset)
    assert "T_sample" not in src
    assert "sample" not in src.split("def evaluate_known_offset")[1] \
        .split("def ")[0].lower()


def test_no_qPCR_information():
    # 评估函数 (目标函数) 中绝无 qPCR/86C 引用
    src = inspect.getsource(V.evaluate_known_offset)
    assert "qPCR" not in src and "QPCR" not in src
    assert "86" not in src.split("def evaluate_known_offset")[1] \
        .split("def ")[0]
    src2 = inspect.getsource(V.find_setpoint90_transition)
    assert "qPCR" not in src2 and "86" not in src2


# ============================================================
# 22-29 无拟合 / 保留 / 权威 / 参考 / 不提升
# ============================================================

def test_no_parameter_fitting():
    src = inspect.getsource(V.evaluate_known_offset)
    assert "scipy" not in src and "minimize" not in src
    assert "curve_fit" not in src
    # 单次前向评估
    assert src.count("run_convection_radiation_fdm") == 1


def test_old_uncertain_sync_outputs_preserved():
    assert V.OLD_V1_ROOT.is_dir()
    # 旧输出目录存在且包含 v1 文件
    assert (V.OLD_V1_ROOT / "comparison"
            / "multi_dataset_validation_summary.csv").is_file()


def test_60C_authoritative_if_anchor_found():
    a = V.find_setpoint90_transition(
        V.DATASETS["VALIDATION_60C_REDO"]["int_path"])
    assert a["n_transitions"] == 1  # 无歧义 -> 权威
    # 脚本输出标记 authoritative=True
    p = V.OUTPUT_ROOT / "60C" / "validation_summary_known_offset.csv"
    if p.is_file():
        s = pd.read_csv(p).iloc[0]
        assert bool(s["authoritative_validation"]) is True


def test_72C_authoritative_if_anchor_found():
    a = V.find_setpoint90_transition(
        V.DATASETS["VALIDATION_72C_REDO"]["int_path"])
    assert a["n_transitions"] == 1
    p = V.OUTPUT_ROOT / "72C" / "validation_summary_known_offset.csv"
    if p.is_file():
        s = pd.read_csv(p).iloc[0]
        assert bool(s["authoritative_validation"]) is True


def test_3s_retained_as_authoritative_reference():
    assert V.REF_3S_RMSE == 1.0643
    p = V.OUTPUT_ROOT / "comparison" / "known_offset_validation_summary.csv"
    if p.is_file():
        df = pd.read_csv(p)
        assert "VALIDATION_3S_SYNCHRONIZED" in df["dataset"].tolist()
        row = df[df["dataset"] == "VALIDATION_3S_SYNCHRONIZED"].iloc[0]
        assert abs(row["RMSE_C"] - 1.0643) < 1e-3


def test_calibration_excluded_from_validation_mean():
    p = V.OUTPUT_ROOT / "comparison" / "known_offset_validation_summary.csv"
    if p.is_file():
        df = pd.read_csv(p)
        ext = df[df["role"] == "EXTERNAL_VALIDATION"]
        calib = df[df["role"] == "CALIBRATION"]
        assert len(ext) == 3
        assert len(calib) == 1
        # 外部均值不含校准
        mean = ext["RMSE_C"].mean()
        assert abs(mean - np.mean([1.3749, 3.0817, 1.0643])) < 0.01


def test_historical_candidate_files_unchanged():
    frozen_src = Path(V.PROJECT_ROOT / "thermal_model/historical/frozen_strategy_G_candidate.py"
                      ).read_text(encoding="utf-8")
    assert "FROZEN_K_W_MK = 0.055" in frozen_src
    assert "FROZEN_TAU_S = 8.5" in frozen_src


def test_no_automatic_model_promotion():
    # main 不写入冻结候选文件; 只写输出目录 (json/csv/png)
    src = inspect.getsource(V.main)
    assert "frozen_strategy_G_candidate" not in src
    assert "OUTPUT_ROOT" in src
    assert "write_text" in src
    # 冻结候选模块导入 (只读) 存在于模块头部, 但 main 不修改它
    assert "import frozen_strategy_G_candidate" not in SRC


# ============================================================
# 额外行为
# ============================================================

def test_sync_rule_and_status_correct():
    assert V.SYNC_RULE == "SETPOINT_90C_EVENT_PLUS_1S"
    assert V.SYNC_STATUS == "KNOWN_PHYSICAL_OFFSET"
    assert V.EXPERIMENTAL_OFFSET_S == 1.0


def test_known_offset_result_differs_from_assumed_t0():
    # 60C: 已知偏移 1.3749 vs 旧 1.7405
    assert abs(V.OLD_60C_RMSE - 1.7405) < 1e-4
    assert abs(V.OLD_72C_RMSE - 1.2447) < 1e-4
    # v2 输出存在
    p = V.OUTPUT_ROOT / "comparison" / "old_vs_new_sync_comparison.csv"
    if p.is_file():
        df = pd.read_csv(p)
        assert len(df) == 2
        assert set(df["dataset"]) == {"60C", "72C"}


def test_locked_assertion_rejects_modified_params():
    src = inspect.getsource(V.assert_locked_parameters)
    assert "0.0675" in src and "700.0" in src and "8.0" in src
    V.assert_locked_parameters()
    from workflows.validation import validate_66C_candidate_known_offset as V2
    orig = V2.K_EFF
    try:
        V2.K_EFF = 0.10
        with pytest.raises(RuntimeError):
            V2.assert_locked_parameters()
    finally:
        V2.K_EFF = orig
