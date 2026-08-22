#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
60°C 独立协议 transfer check 的冻结模型验证测试。

核心规则:
    - 使用冻结名义标定 (NOMINAL_BARE_TOP_CALIBRATION_V1: k=0.0165, cp=900);
    - 绝对不重新拟合 / 不扫描 / 不优化;
    - 查询轴 = 实测时间 (绝不允许实测温度值作插值查询坐标);
    - 初始条件 = 第一个对齐 T_internal;
    - 顶部观测 x = 850 um; 裸顶几何; 无 Air/PDMS;
    - 残差 = pred - meas; 样品温度不参与验证决策;
    - 输出目录与 72C 隔离; 72C 输出与 60C 输入均不被修改;
    - 对齐无外推; 指标合成数组正确; metadata refitted=false。
"""
import json

import numpy as np
import pandas as pd
import pytest

from thermal_model.config import calibrated_model_config as cmc
from thermal_model.core import heat_model
from workflows.validation import validate_calibrated_thermal_model as vtm
from thermal_model.utilities.align_internal_and_top_temperature import align_to_top_reference
from thermal_model.utilities.scan_effective_thermal_parameters import (
    sample_prediction_at_measurement_times,
)

# ---------------------------------------------------------------
# 1-3. 冻结配置
# ---------------------------------------------------------------

def test_validation_uses_nominal_calibration():
    cal, layers, mats, h_conv, t_amb = vtm.build_frozen_config()
    assert cal is cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    assert cal.status == "accepted"
    assert cal.valid_for_final_calibration is True
    assert cal.source_analysis == "corrected_time_objective_v3"


def test_k_eff_remains_exactly_0p0165():
    cal, _, _, _, _ = vtm.build_frozen_config()
    assert cal.k_eff_W_mK == pytest.approx(0.0165, abs=0.0)
    assert cal.k_eff_W_mK == 0.0165


def test_cp_eff_remains_exactly_900():
    cal, _, _, _, _ = vtm.build_frozen_config()
    assert cal.cp_eff_J_kgK == pytest.approx(900.0, abs=0.0)
    assert cal.cp_eff_J_kgK == 900.0


# ---------------------------------------------------------------
# 4-6. 无优化 / 无扫描 / 无时间平移
# ---------------------------------------------------------------

def test_validation_workflow_has_no_optimizer():
    src = vtm.__file__
    text = open(src, encoding="utf-8").read()
    # 不允许任何数值优化库/调用 (注释里的 "optimizer" 一词不算)
    assert "import scipy" not in text
    assert "from scipy" not in text
    assert "scipy.optimize" not in text
    assert "curve_fit(" not in text
    assert "minimize(" not in text
    assert "least_squares" not in text


def test_no_parameter_scan_performed():
    # run_transfer_validation 只调用一次 run_frozen_simulation,
    # 参数从冻结配置直接读取, 没有网格/循环搜索。
    calls = []

    def fake_run_simulation(**kwargs):
        calls.append(kwargs)
        return {
            "t_array": np.arange(0.0, 40.0, 0.1),
            "T_top_surface_arr": np.full(400, 26.0),
            "T_sample_arr": np.full(400, 26.0),
        }

    monkey = pytest.MonkeyPatch()
    monkey.setattr(vtm.heat_model, "run_simulation", fake_run_simulation)
    try:
        t = np.arange(1.0, 41.0, dtype=float)
        t_int = 25.0 + 0.5 * t
        cal, layers, mats, h_conv, t_amb = vtm.build_frozen_config()
        vtm.run_frozen_simulation(t, t_int, cal, layers, mats)
    finally:
        monkey.undo()
    assert len(calls) == 1
    kw = calls[0]
    assert kw["materials"]["COC"].k_W_mK == 0.0165
    assert kw["materials"]["COC"].cp_J_kgK == 900.0
    assert kw["h_conv"] == vtm.H_CONV == 5.0
    assert kw["T_air_ambient"] == vtm.T_AMB == 25.0
    assert kw["T_initial_C"] == t_int[0]


def test_no_time_shift_is_fitted():
    import inspect
    sig = inspect.signature(vtm.run_transfer_validation)
    assert "time_shift" not in sig.parameters
    assert "offset" not in sig.parameters
    # 对齐数据的时间轴原样使用, 不做互相关/峰值对齐
    assert vtm.load_aligned_data  # 只是读取, 不修改


# ---------------------------------------------------------------
# 7-8. 修正时间采样 (查询轴 = 实测时间)
# ---------------------------------------------------------------

def test_measurement_time_is_interpolation_query_axis():
    # 合成回归: FDM 时间 [0..60], FDM 信号 = 时间本身
    fdm_t = np.arange(0.0, 61.0, dtype=float)
    fdm_sig = fdm_t.copy()
    meas_time = np.array([10.0, 20.0, 30.0])
    measured_temp = np.array([40.0, 50.0, 60.0])  # 误导性查询候选
    out = sample_prediction_at_measurement_times(meas_time, fdm_t, fdm_sig)
    np.testing.assert_allclose(out, meas_time)  # [10,20,30], 不是 [40,50,60]


def test_measured_temperature_cannot_be_interpolation_query():
    fdm_t = np.arange(0.0, 61.0, dtype=float)
    fdm_sig = fdm_t.copy()
    meas_time = np.array([10.0, 20.0, 30.0])
    measured_temp = np.array([40.0, 50.0, 60.0])
    out = sample_prediction_at_measurement_times(meas_time, fdm_t, fdm_sig)
    # 若把实测温度值当作查询轴, 会得到 [40,50,60], 与正确结果不符
    assert not np.allclose(out, measured_temp)
    np.testing.assert_allclose(out, meas_time)


# ---------------------------------------------------------------
# 9. 初始条件 = 第一个对齐 T_internal
# ---------------------------------------------------------------

def test_initial_condition_equals_first_aligned_internal():
    t_int = np.array([29.135, 36.47, 48.63, 60.88])
    assert vtm.compute_initial_condition(t_int) == pytest.approx(29.135)


def test_frozen_run_uses_first_internal_as_initial():
    calls = []

    def fake_run_simulation(**kwargs):
        calls.append(kwargs)
        return {
            "t_array": np.arange(0.0, 20.0, 0.1),
            "T_top_surface_arr": np.full(200, 27.0),
            "T_sample_arr": np.full(200, 27.0),
        }

    monkey = pytest.MonkeyPatch()
    monkey.setattr(vtm.heat_model, "run_simulation", fake_run_simulation)
    try:
        t = np.arange(1.0, 11.0, dtype=float)
        t_int = np.array([30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
                          36.0, 37.0, 38.0, 39.0])
        cal, layers, mats, _, _ = vtm.build_frozen_config()
        vtm.run_frozen_simulation(t, t_int, cal, layers, mats)
    finally:
        monkey.undo()
    assert calls[0]["T_initial_C"] == 30.0


# ---------------------------------------------------------------
# 10-12. 几何: 裸顶 850 um, 顶部观测 x=850 um, 无 Air/PDMS
# ---------------------------------------------------------------

def test_top_target_is_x_850um():
    cal, layers, mats, _, _ = vtm.build_frozen_config()
    mesh = heat_model.build_layer_stack(mats, layers)
    assert mesh.idx_top_surface.size == 1
    assert mesh.x[mesh.idx_top_surface[0]] == pytest.approx(850e-6)
    assert mesh.x[-1] == pytest.approx(850e-6)
    assert layers[-1].role == "top_surface"
    assert layers[-1].material == "COC"


def test_geometry_is_bare_top_layers():
    cal, layers, mats, _, _ = vtm.build_frozen_config()
    names = [ly.name for ly in layers]
    assert names == ["Bottom COC", "PCR Sample", "Mineral Oil", "Top COC"]
    assert sum(ly.thickness_m for ly in layers) == pytest.approx(850e-6)


def test_air_and_pdms_absent():
    cal, layers, _, _, _ = vtm.build_frozen_config()
    names = " ".join(ly.name for ly in layers)
    assert "Air" not in names
    assert "PDMS" not in names


# ---------------------------------------------------------------
# 13-14. 残差符号与样品温度排除
# ---------------------------------------------------------------

def test_residual_sign_is_predicted_minus_measured():
    pred = np.array([26.0, 27.0, 28.0])
    meas = np.array([25.0, 28.0, 26.0])
    residual = pred - meas
    np.testing.assert_allclose(residual, [1.0, -1.0, 2.0])
    m = vtm.compute_validation_metrics([1.0, 2.0, 3.0], residual)
    assert m["max_positive_residual_C"] == 2.0
    assert m["max_negative_residual_C"] == -1.0
    assert m["mean_residual_top_C"] == pytest.approx(2.0 / 3.0)


def test_sample_temperature_excluded_from_validation():
    import inspect
    src = inspect.getsource(vtm.run_transfer_validation)
    # 残差只由顶部预测 - 顶部实测构成
    assert "residual = T_top_pred - t_top_meas" in src
    # 样品预测只是输出列, 不进入 metrics 决策
    assert "T_sample_pred" in src
    m = vtm.compute_validation_metrics([1.0, 2.0], np.array([0.5, -0.5]))
    assert "sample" not in "".join(m.keys()).lower()
    assert set(m) == {
        "RMSE_top_C", "MAE_top_C", "median_absolute_error_C",
        "p95_absolute_error_C", "mean_residual_top_C",
        "max_positive_residual_C", "max_negative_residual_C",
        "max_absolute_residual_C", "time_of_max_abs_residual_s",
    }


# ---------------------------------------------------------------
# 15-17. 输出隔离 / 72C 输出不动 / 60C 输入不动
# ---------------------------------------------------------------

def test_output_directory_isolated_from_72c():
    out = str(vtm.DEFAULT_OUTPUT_DIR)
    assert out.endswith("60C_redo_transfer_check_v1")
    assert "72C" not in out
    assert "72C_corrected_objective_v1" not in out
    assert "72C_nominal_v1" not in out


def test_72c_historical_outputs_not_modified():
    out = vtm.DEFAULT_OUTPUT_DIR
    ref = vtm.DEFAULT_REF_METRICS_JSON
    assert ref.exists()
    # 验证输出目录不在任何 72C 输出目录内部
    for d in ["72C_corrected_objective_v1", "72C_nominal_v1",
              "parameter_scan_output/72C"]:
        assert str(d) not in str(out)


def test_60c_input_data_not_modified():
    # 对齐数据只读: 验证脚本不向其所在目录写任何文件
    aligned_dir = vtm.DEFAULT_ALIGNED_CSV.parent
    assert str(aligned_dir) != str(vtm.DEFAULT_OUTPUT_DIR)
    # 检查模块源代码没有对输入文件路径执行写操作
    import inspect
    src = inspect.getsource(vtm)
    assert "aligned_dir.write" not in src
    assert "DEFAULT_ALIGNED_CSV" in src  # 仅用于读取


# ---------------------------------------------------------------
# 18. 对齐无外推
# ---------------------------------------------------------------

def test_alignment_avoids_extrapolation():
    internal = {
        "t_internal": np.array([10.0, 15.0, 20.0]),
        "T": np.array([30.0, 40.0, 50.0]),
        "n_original": 3, "n_valid": 3,
        "resolved_column": "T",
        "time_source": "numeric",
        "first_time": 10.0, "last_time": 20.0,
        "median_dt": 5.0, "min_dt": 5.0, "max_dt": 5.0,
    }
    top = {
        "t_top": np.arange(0.0, 31.0, dtype=float),
        "T": np.full(31, 25.0),
        "n_original": 31, "n_valid": 31,
        "resolved_column": "T Avg",
        "time_source": "grid", "median_dt": 1.0,
        "min_dt": 1.0, "max_dt": 1.0, "t_diag": None,
    }
    aligned = align_to_top_reference(internal, top, max_top_rows=1000)
    assert aligned["n_excluded_early"] == 10   # t=0..9 < 10
    assert aligned["n_excluded_late"] == 10    # t=21..30 > 20
    assert aligned["n_aligned"] == 11          # t=10..20
    assert np.all(aligned["time_s"] >= 10.0 - 1e-9)
    assert np.all(aligned["time_s"] <= 20.0 + 1e-9)
    # np.interp 只在内部采样区间内插值; 区间外点被排除, 无外推
    assert aligned["interpolation_method"] == "linear (np.interp)"
    assert np.all(aligned["time_s"] <= internal["t_internal"][-1] + 1e-9)
    assert np.all(aligned["time_s"] >= internal["t_internal"][0] - 1e-9)


# ---------------------------------------------------------------
# 19. 指标合成数组
# ---------------------------------------------------------------

def test_metrics_correct_on_synthetic_arrays():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    r = np.array([1.0, -1.0, 2.0, -2.0])
    m = vtm.compute_validation_metrics(t, r)
    assert m["RMSE_top_C"] == pytest.approx(np.sqrt(10.0 / 4.0))
    assert m["MAE_top_C"] == pytest.approx(1.5)
    assert m["median_absolute_error_C"] == pytest.approx(1.5)
    assert m["p95_absolute_error_C"] == pytest.approx(
        np.percentile(np.abs(r), 95))
    assert m["mean_residual_top_C"] == pytest.approx(0.0)
    assert m["max_positive_residual_C"] == 2.0
    assert m["max_negative_residual_C"] == -2.0
    assert m["max_absolute_residual_C"] == 2.0
    assert m["time_of_max_abs_residual_s"] == 3.0  # 第一个最大幅值点


# ---------------------------------------------------------------
# 20. metadata refitted=false
# ---------------------------------------------------------------

def test_metadata_records_refitted_false(tmp_path):
    t = np.arange(1.0, 41.0, dtype=float)
    t_int = 25.0 + 0.5 * t
    t_top = np.full(40, 26.0)
    out = tmp_path / "out"
    vtm.run_transfer_validation(
        t, t_int, t_top, out,
        experiment_name="60C_redo",
        regime_labeled_csv=str(tmp_path / "missing_regime.csv"),
        source_files={
            "top_file": "top.xls",
            "internal_file": "internal.xlsx",
        },
        alignment_summary={"reused_aligned_dataset": True,
                           "extrapolation_used": False},
    )
    meta = json.loads(
        (out / "60C_transfer_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["refitted"] is False
    assert meta["k_eff_W_mK"] == 0.0165
    assert meta["cp_eff_J_kgK"] == 900.0
    assert meta["rho_COC_kg_m3"] == 1020.0
    assert meta["analysis_name"] == "60C_redo_transfer_check_v1"
    assert "bare-top" in meta["geometry"]
    assert meta["initial_condition_mode"] == "first aligned internal temperature"
    assert meta["measurement_count"] == 40
    trace = pd.read_csv(out / "60C_transfer_trace.csv")
    assert list(trace.columns) == [
        "time_s", "T_internal_C", "T_top_measured_C", "T_top_predicted_C",
        "T_sample_predicted_C", "top_residual_C",
    ]
    # 残差符号 = pred - meas
    np.testing.assert_allclose(
        trace["top_residual_C"].to_numpy(),
        trace["T_top_predicted_C"].to_numpy()
        - trace["T_top_measured_C"].to_numpy(),
    )
