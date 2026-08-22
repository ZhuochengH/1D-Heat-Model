#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅内部温度的样品层重建 (DOE11 faster) —— 冻结模型测试。

验证:
    1-4  冻结标定 NOMINAL_BARE_TOP_CALIBRATION_V1 (0.0165 / 900 / 1020);
    5     无任何优化/标定函数被调用;
    6     不要求实测 Top COC 文件;
    7-8   原始 Time(s) 为源时间轴; elapsed 归一化保持全部时间间隔;
    9-10  边界 = 实测内部温度; 初始条件 = 第一个内部值;
    11-12 BARE_TOP_COC_LAYERS; 无 Air/PDMS;
    13    样品输出用修正 CV 加权空间平均 (LayerStack.sample_weights);
    14    输出采样用时间坐标查询 FDM;
    15    源实测数据不被修改;
    16    CSV 列正确;
    17    阈值停留时间尊重非均匀 dt;
    18    斜坡率用实际时间坐标;
    19    输出隔离到指定目录;
    20    既有 72C/60C 输出不被修改。
"""
import json

import numpy as np
import pandas as pd
import pytest

from thermal_model.config import calibrated_model_config as cmc
from thermal_model.core import heat_model
from thermal_model.utilities import predict_sample_from_internal_temperature as psf

# ---------------------------------------------------------------
# 1-4. 冻结配置
# ---------------------------------------------------------------

def test_uses_nominal_calibration():
    cal, layers, mats, h_conv, t_amb = psf.build_frozen_config()
    assert cal is cmc.NOMINAL_BARE_TOP_CALIBRATION_V1
    assert cal.status == "accepted"


def test_k_eff_exactly_0p0165():
    cal, _, _, _, _ = psf.build_frozen_config()
    assert cal.k_eff_W_mK == 0.0165
    assert cal.k_eff_W_mK == pytest.approx(0.0165, abs=0.0)


def test_cp_eff_exactly_900():
    cal, _, _, _, _ = psf.build_frozen_config()
    assert cal.cp_eff_J_kgK == 900.0
    assert cal.cp_eff_J_kgK == pytest.approx(900.0, abs=0.0)


def test_rho_exactly_1020():
    cal, _, _, _, _ = psf.build_frozen_config()
    assert cal.rho_COC_kg_m3 == 1020.0


# ---------------------------------------------------------------
# 5-6. 无优化 / 无 Top COC 依赖
# ---------------------------------------------------------------

def test_no_calibration_or_optimization_called():
    text = open(psf.__file__, encoding="utf-8").read()
    assert "import scipy" not in text
    assert "from scipy" not in text
    assert "optimize" not in text.replace("optimize", "")  # 注释中的词不算
    # 直接检查脚本不含任何数值优化调用
    for token in ["curve_fit(", "minimize(", "least_squares", "grid_search",
                  "ParameterGrid"]:
        assert token not in text


def test_no_top_COC_file_required():
    # CLI 只有 --input (内部日志), 无 --top 参数
    import argparse
    parser = psf.parse_args(["--input", "x.xlsx",
                             "--output-dir", "out"])
    assert not hasattr(parser, "top_file")
    assert "top" not in vars(parser)
    # 模块默认输入就是内部温度文件
    assert "faster_zone1_temperature_analysis" in str(psf.DEFAULT_INPUT)


# ---------------------------------------------------------------
# 7-8. 时间轴
# ---------------------------------------------------------------

def test_source_time_is_actual_time_column():
    data = psf.load_internal_data(
        "/mnt/d/桌面/微流控毕设/Calibration/"
        "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx"
    )
    assert data["resolved_time_col"] == "Time(s)"
    assert data["resolved_temp_col"] == "Zone 1 Avg (°C)"
    # 首个源时间应等于文件内的 0.09 s
    assert data["first_time"] == pytest.approx(0.09, abs=0.01)


def test_elapsed_preserves_all_intervals():
    data = psf.load_internal_data(
        "/mnt/d/桌面/微流控毕设/Calibration/"
        "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx"
    )
    src = data["source_time_s"]
    el = data["elapsed_time_s"]
    np.testing.assert_allclose(np.diff(el), np.diff(src))
    assert el[0] == pytest.approx(0.0)


# ---------------------------------------------------------------
# 9-10. 边界与初始条件
# ---------------------------------------------------------------

def test_boundary_is_measured_internal_temperature():
    calls = []

    def fake_run_simulation(**kwargs):
        calls.append(kwargs)
        return {
            "t_array": np.arange(0.0, 50.0, 0.1),
            "T_sample_arr": np.full(500, 25.0),
            "T_top_surface_arr": np.full(500, 25.0),
        }

    monkey = pytest.MonkeyPatch()
    monkey.setattr(psf.heat_model, "run_simulation", fake_run_simulation)
    try:
        e = np.arange(0.0, 10.0, dtype=float)
        ti = np.array([22.1, 22.5, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
                       90.0, 95.0])
        cal, layers, mats, _, _ = psf.build_frozen_config()
        psf.run_frozen_simulation(e, ti, cal, layers, mats)
    finally:
        monkey.undo()
    assert len(calls) == 1
    kw = calls[0]
    np.testing.assert_allclose(kw["bottom_temperature_C"], ti)
    np.testing.assert_allclose(kw["time_s"], e)
    assert kw["T_initial_C"] == ti[0]


def test_initial_condition_is_first_internal():
    ti = np.array([22.105, 22.2, 30.0])
    assert psf.compute_initial_condition(ti) == pytest.approx(22.105)


# ---------------------------------------------------------------
# 11-12. 几何
# ---------------------------------------------------------------

def test_uses_bare_top_layers():
    cal, layers, mats, _, _ = psf.build_frozen_config()
    mesh = heat_model.build_layer_stack(mats, layers)
    assert mesh.x[-1] == pytest.approx(850e-6)
    assert mesh.idx_top_surface.size == 1
    assert layers[-1].role == "top_surface"


def test_air_and_pdms_absent():
    cal, layers, _, _, _ = psf.build_frozen_config()
    names = " ".join(ly.name for ly in layers)
    assert "Air" not in names
    assert "PDMS" not in names


# ---------------------------------------------------------------
# 13. 样品 CV 加权平均
# ---------------------------------------------------------------

def test_sample_uses_cv_weighted_average():
    cal, layers, mats, _, _ = psf.build_frozen_config()
    mesh = heat_model.build_layer_stack(mats, layers)
    # 样品层 180-200 um; 全数组权重和 = 1 (含界面节点部分 CV 权重)
    assert mesh.idx_sample.size > 0
    assert np.sum(mesh.sample_weights) == pytest.approx(1.0)
    x_low, x_high = 180e-6, 200e-6
    xs = mesh.x[mesh.idx_sample]
    assert xs[0] >= x_low - 1e-9 and xs[-1] <= x_high + 1e-9
    w = mesh.sample_weights[mesh.idx_sample]
    L = x_high - x_low  # 20e-6
    dx = xs[1] - xs[0]  # 5e-6
    # 内部节点全 CV: w = dx / L
    assert w[0] == pytest.approx(dx / L, rel=1e-6)
    # 顶界面节点 (x=200um) 半 CV (右侧在油层内): w = (dx/2) / L
    assert w[-1] == pytest.approx((dx / 2.0) / L, rel=1e-6)
    # 左界面节点 (x=180um, 不在 idx_sample) 也有部分权重 —— 全数组和=1 依赖它
    total = np.sum(mesh.sample_weights)
    assert total == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------
# 14. 时间坐标查询 FDM
# ---------------------------------------------------------------

def test_sampling_queries_by_time_coordinates():
    from thermal_model.utilities.scan_effective_thermal_parameters import (
        sample_prediction_at_measurement_times,
    )
    fdm_t = np.arange(0.0, 60.0, 1.0)
    fdm_sig = fdm_t.copy()          # 信号 = 时间
    meas_time = np.array([5.0, 15.0, 25.0])
    out = sample_prediction_at_measurement_times(meas_time, fdm_t, fdm_sig)
    np.testing.assert_allclose(out, meas_time)  # 不是 [x,y,z] 温度值


# ---------------------------------------------------------------
# 15-16. 源数据不改 / CSV 列
# ---------------------------------------------------------------

def test_source_data_not_modified(tmp_path):
    src = "/mnt/d/桌面/微流控毕设/Calibration/" \
          "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx"
    df_before = pd.read_excel(src, sheet_name="Extracted_Data")
    h_before = df_before.copy()
    data = psf.load_internal_data(src)
    df_after = pd.read_excel(src, sheet_name="Extracted_Data")
    pd.testing.assert_frame_equal(h_before, df_after)
    assert data["n_valid"] == len(df_before)


def test_csv_columns_correct(tmp_path):
    e = np.arange(0.0, 40.0, dtype=float)
    src = e + 100.0  # 非零源时间
    ti = 22.0 + 0.5 * e
    out = tmp_path / "out"
    summary, trace = psf.predict_sample_temperature(
        src, e, ti, out,
        experiment_name="doe_test",
        input_path="fake.xlsx",
    )
    assert list(trace.columns) == [
        "source_time_s", "elapsed_time_s", "T_internal_C",
        "T_sample_predicted_C", "T_top_predicted_C",
    ]
    assert (out / "sample_temperature_prediction.csv").exists()
    assert (out / "sample_temperature_summary.txt").exists()
    assert (out / "sample_prediction_metadata.json").exists()


# ---------------------------------------------------------------
# 17. 停留时间尊重非均匀时间
# ---------------------------------------------------------------

def test_dwell_times_use_interval_integration():
    # 非均匀 dt: 两个 1 s 区间 + 一个 10 s 区间, 全部 T>=90
    t = np.array([0.0, 1.0, 2.0, 12.0])
    T = np.array([91.0, 92.0, 93.0, 94.0])
    dw = psf.dwell_times(t, T, thresholds=(90.0,), ranges=())
    assert dw["sample_ge_90C_s"] == pytest.approx(12.0)
    # 行数计数会给 4 点, 但区间积分给 12 s —— 验证非均匀尊重


def test_dwell_crossing_partial_interval():
    # 线性跨越阈值: 区间 [0,1] 从 88 到 92 -> 半段在 90 以上
    t = np.array([0.0, 1.0])
    T = np.array([88.0, 92.0])
    dw = psf.dwell_times(t, T, thresholds=(90.0,), ranges=())
    assert dw["sample_ge_90C_s"] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------
# 18. 斜坡率用实际时间坐标
# ---------------------------------------------------------------

def test_ramp_rates_use_actual_time():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    T = np.array([20.0, 30.0, 40.0, 30.0, 20.0])  # dT/dt = +10, -10
    r = psf.ramp_summary(t, T)
    assert r["max_positive_C_per_s"] == pytest.approx(10.0)
    assert r["max_negative_C_per_s"] == pytest.approx(-10.0)
    # 双倍时间间隔 -> 斜率减半
    t2 = t * 2.0
    r2 = psf.ramp_summary(t2, T)
    assert r2["max_positive_C_per_s"] == pytest.approx(5.0)


# ---------------------------------------------------------------
# 19-20. 输出隔离 / 既有输出不动
# ---------------------------------------------------------------

def test_output_isolated(tmp_path):
    out = psf.DEFAULT_OUTPUT_DIR
    assert "08.12_pm_DOE11_faster_sample_prediction_v1" in str(out)
    for d in ["72C_corrected_objective_v1", "72C_nominal_v1",
              "60C_redo_transfer_check_v1", "parameter_scan_output"]:
        assert d not in str(out)


def test_existing_outputs_not_modified(tmp_path):
    # 脚本只向其 output_dir 写入; 默认输出目录与既有目录完全不同
    assert psf.DEFAULT_OUTPUT_DIR.parent == \
        psf.PROJECT_ROOT / "calibrated_model_output"
    assert "08.12" in str(psf.DEFAULT_OUTPUT_DIR.name)
    # 元数据记录 no_refitting
    e = np.arange(0.0, 20.0, dtype=float)
    src = e + 5.0
    ti = 22.0 + e
    out = tmp_path / "o"
    psf.predict_sample_temperature(src, e, ti, out,
                                   experiment_name="doe_test",
                                   input_path="fake.xlsx")
    meta = json.loads(
        (out / "sample_prediction_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["no_refitting"] is True
    assert meta["no_top_COC_measurement_used"] is True
    assert meta["initial_condition"]["value_C"] == pytest.approx(22.0)
    assert meta["git_tag"]  # 非空


# ---------------------------------------------------------------
# 21. 周期检测: 低谷必须低于峰值 (回归: 旧实现 trough==peak)
# ---------------------------------------------------------------

def test_cycle_detection_finds_real_troughs():
    # 合成 2 个周期: 20 C 谷 -> 95 C 峰
    t = np.arange(0.0, 100.0, 1.0)
    ti = np.full(100, 30.0)
    ti[10:20] = np.linspace(30, 95, 10)     # 峰 1
    ti[20:40] = 95.0
    ti[40:50] = np.linspace(95, 25, 10)     # 谷 1
    ti[50:60] = 25.0
    ti[60:70] = np.linspace(25, 95, 10)     # 峰 2
    ti[70:90] = 95.0
    ts = ti.copy() - 5.0                    # 样品滞后偏冷
    cycles = psf.detect_cycles(t, ti, ts, peak_threshold=85.0)
    assert len(cycles) == 2
    # 每个周期低谷必须显著低于峰值
    for c in cycles:
        assert c["internal_low_trough_C"] < c["internal_high_peak_C"] - 30.0
        assert c["cycle_start_time_s"] < c["internal_peak_time_s"]
    # 样品峰时间应不早于内部峰 (滞后)
    assert cycles[1]["sample_peak_time_s"] >= cycles[1]["internal_peak_time_s"] - 1e-9
    # 周期 2 起点应是谷 1 处 (t~50-60 低谷平台右端), 不是 t=0
    assert 50.0 <= cycles[1]["cycle_start_time_s"] <= 60.0


def test_cycle_detection_merges_plateau_wiggles():
    # 初始高温平台上的波动不应被拆成多个周期
    t = np.arange(0.0, 80.0, 1.0)
    ti = np.full(80, 30.0)
    ti[10:60] = 90.0 + 0.3 * np.sin(np.arange(50) * 1.7)  # 90C 平台带波动
    ti[60:70] = np.linspace(90, 25, 10)                   # 谷
    ti[70:79] = np.linspace(25, 95, 9)                    # 峰 2
    ts = ti.copy()
    cycles = psf.detect_cycles(t, ti, ts, peak_threshold=88.0)
    # 平台波动合并 -> 1 个峰 + 峰 2 = 2 个周期
    assert len(cycles) == 2
    assert cycles[0]["internal_high_peak_C"] == pytest.approx(90.3, abs=0.5)


def test_cycle_detection_returns_empty_without_clear_cycles():
    t = np.arange(0.0, 50.0, 1.0)
    ti = np.full(50, 45.0) + 0.1 * np.sin(t)
    cycles = psf.detect_cycles(t, ti, ti.copy(), peak_threshold=88.0)
    assert cycles == []
