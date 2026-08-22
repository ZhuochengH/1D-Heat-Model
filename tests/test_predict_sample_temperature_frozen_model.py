#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冻结热模型样品温度预测工具 V2 — 测试 (40 项)
============================================

覆盖 (规格 #40):
  1-4   CLI 模型选择 (bare/insulated/both; 无效拒绝)
  5-10  裸顶/绝缘几何 (层、厚度、外边界位置)
 11-14 both 模式: 相同 k/cp / 相同内部输入 / 相同初始 / 相同环境
 15-16 样品绝不施加滞后 (裸顶/绝缘)
 17    --start-s 不重置热状态
 18-19 部分窗口 == 全历史切片 (裸顶/绝缘, 数值容差)
 20    --end-s 正确限制模拟
 21-23 时间轴: original 保留 / simulation 相对真模拟起点 /
       analysis 相对显示起点
 24-27 静态 PNG/PDF (裸顶/绝缘/both)
 28    CSV schema (三种模式)
 29    summary (三种模式)
 30    both 模式 delta 正确
 31    draggable_hlines.py 复用
 32-33 阈值时序引用裸顶/绝缘样品
 34    both 公共阈值同时显示两个时长
 35    时间戳感知时长保留
 36    无参数拟合
 37    无时移优化
 38    绝缘模式文档化为前向扩展 (非独立验证)
 39    裸顶验证状态文档正确
 40    权威模型文件未被修改
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermal_model.core import heat_model
from workflows.prediction import predict_sample_temperature_frozen_model as psf
from thermal_model.historical.frozen_strategy_G_candidate import FROZEN_STRATEGY_G_CANDIDATE

SRC = Path(psf.__file__).read_text(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_WORKBOOK = (
    PROJECT_ROOT.parent / "Calibration"
    / "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx"
)

FROZEN_FILES = [
    "thermal_model/core/heat_model.py",
    "thermal_model/core/convection_radiation_thermal_model.py",
    "thermal_model/historical/frozen_strategy_G_candidate.py",
    "thermal_model/utilities/draggable_hlines.py",
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


HASHES_BEFORE = {f: _sha(PROJECT_ROOT / f) for f in FROZEN_FILES}
WORKBOOK_HASH_BEFORE = _sha(TEST_WORKBOOK)


@pytest.fixture(scope="module")
def full_bare():
    return psf.run_prediction(TEST_WORKBOOK, model="bare")


@pytest.fixture(scope="module")
def full_insulated():
    return psf.run_prediction(TEST_WORKBOOK, model="insulated")


@pytest.fixture(scope="module")
def full_both():
    return psf.run_prediction(TEST_WORKBOOK, model="both")


@pytest.fixture(scope="module")
def partial_bare():
    return psf.run_prediction(TEST_WORKBOOK, model="bare",
                              start_s=100, end_s=200)


@pytest.fixture(scope="module")
def partial_insulated():
    return psf.run_prediction(TEST_WORKBOOK, model="insulated",
                              start_s=100, end_s=200)


@pytest.fixture(scope="module")
def main_bare(tmp_path_factory):
    out = tmp_path_factory.mktemp("main_bare")
    psf.main(["--input", str(TEST_WORKBOOK), "--model", "bare",
              "--output-dir", str(out), "--no-gui"])
    return out


@pytest.fixture(scope="module")
def main_insulated(tmp_path_factory):
    out = tmp_path_factory.mktemp("main_insulated")
    psf.main(["--input", str(TEST_WORKBOOK), "--model", "insulated",
              "--output-dir", str(out), "--no-gui"])
    return out


@pytest.fixture(scope="module")
def main_both_output(tmp_path_factory, full_both):
    out = tmp_path_factory.mktemp("main_both")
    psf.plot_static(full_both, out)
    psf.write_csv(full_both, out)
    psf.write_summary(full_both, out)
    return out


# ---------------------------------------------------------------
# 1-4. CLI 模型选择
# ---------------------------------------------------------------

def test_cli_accepts_bare():
    args = psf.parse_args(["--input", "x.xlsx", "--model", "bare"])
    assert args.model == "bare"


def test_cli_accepts_insulated():
    args = psf.parse_args(["--input", "x.xlsx", "--model", "insulated"])
    assert args.model == "insulated"


def test_cli_accepts_both():
    args = psf.parse_args(["--input", "x.xlsx", "--model", "both"])
    assert args.model == "both"


def test_invalid_model_rejected():
    with pytest.raises(SystemExit):
        psf.parse_args(["--input", "x.xlsx", "--model", "bogus"])


# ---------------------------------------------------------------
# 5-10. 几何
# ---------------------------------------------------------------

def test_bare_geometry_correct():
    layers = psf.build_geometry("bare")
    assert layers is heat_model.BARE_TOP_COC_LAYERS
    assert len(layers) == 4
    mats = {l.material for l in layers}
    assert "Air" not in mats and "PDMS" not in mats
    assert sum(l.thickness_m for l in layers) == pytest.approx(850e-6,
                                                               abs=0.0)


def test_insulated_geometry_correct():
    layers = psf.build_geometry("insulated")
    assert len(layers) == 6
    mats = {l.material for l in layers}
    assert "Air" in mats and "PDMS" in mats
    assert sum(l.thickness_m for l in layers) == pytest.approx(4050e-6,
                                                               abs=0.0)


def test_air_thickness_correct():
    layers = psf.build_geometry("insulated")
    air = [l for l in layers if l.material == "Air"]
    assert len(air) == 1
    assert air[0].thickness_m == pytest.approx(3000e-6, abs=0.0)


def test_pdms_thickness_correct():
    layers = psf.build_geometry("insulated")
    pdms = [l for l in layers if l.material == "PDMS"]
    assert len(pdms) == 1
    assert pdms[0].thickness_m == pytest.approx(200e-6, abs=0.0)


def test_external_boundary_bare_at_top_coc():
    layers = psf.build_geometry("bare")
    # 裸顶: 最外层 = Top COC 外表面, 无绝缘层
    assert layers[-1].name == "Top COC"
    assert layers[-1].material == "COC"
    assert layers[-1].role == "top_surface"


def test_external_boundary_insulated_at_outer_pdms():
    layers = psf.build_geometry("insulated")
    # 绝缘: 最外层 = Cap PDMS 外表面 (环境边界作用于此, 非 Top COC)
    assert layers[-1].name == "Cap PDMS"
    assert layers[-1].material == "PDMS"
    assert layers[-1].role is None


# ---------------------------------------------------------------
# 11-14. both 模式同输入
# ---------------------------------------------------------------

def test_same_frozen_coc_k_cp_both(full_both):
    assert psf.K_EFF == 0.0675
    assert psf.CP_EFF == 700.0
    assert psf.TAU_TOP == 8.0
    # 两种配置共用同一 K_EFF/CP_EFF 常量构建材料库
    assert "make_convection_radiation_materials(K_EFF, CP_EFF, RHO_COC)" in SRC


def test_same_internal_input_both(full_bare, full_both):
    assert np.allclose(full_both["T_internal"], full_bare["T_internal"])
    assert np.allclose(full_both["t_original"], full_bare["t_original"])


def test_same_initial_field_both(full_bare, full_both):
    assert full_both["initial_internal_C"] == pytest.approx(
        full_bare["initial_internal_C"], abs=0.0)


def test_same_environment_both(full_bare, full_both):
    assert full_both["environment_C"] == pytest.approx(
        full_bare["environment_C"], abs=0.0)
    assert full_both["environment_source"] == \
        full_bare["environment_source"] == \
        "INTERNAL_INITIAL_PROXY_NO_TOP_MEASUREMENT"


# ---------------------------------------------------------------
# 15-16. 样品绝不滞后
# ---------------------------------------------------------------

def test_no_lag_on_bare_sample(full_bare):
    assert "apply_first_order_lag" not in SRC
    assert "lag_augmented" not in SRC
    assert "T_sample_arr" in SRC
    assert full_bare["all_finite"] is True


def test_no_lag_on_insulated_sample(full_insulated):
    assert "apply_first_order_lag" not in SRC
    assert full_insulated["all_finite"] is True


# ---------------------------------------------------------------
# 17-20. 热历史语义
# ---------------------------------------------------------------

def test_start_s_does_not_reset_thermal_state(partial_bare, full_bare):
    # 初始/环境 = 完整源迹线首点, 而不是窗口首点
    assert partial_bare["initial_internal_C"] == pytest.approx(
        full_bare["initial_internal_C"], abs=0.0)
    assert partial_bare["environment_C"] == pytest.approx(
        full_bare["environment_C"], abs=0.0)
    # 窗口首点内部温度 != 初始场 (证明未重置)
    assert partial_bare["T_internal"][0] > partial_bare[
        "initial_internal_C"] + 1.0
    assert partial_bare["thermal_history_preserved"] is True


def test_partial_bare_equals_full_slice(partial_bare, full_bare):
    mask = (full_bare["t_original"] >= 100 - 1e-9) & \
        (full_bare["t_original"] <= 200 + 1e-9)
    m2 = (partial_bare["t_original"] >= 100 - 1e-9) & \
        (partial_bare["t_original"] <= 200 + 1e-9)
    assert np.array_equal(partial_bare["t_original"][m2],
                          full_bare["t_original"][mask])
    diff = np.abs(full_bare["sample_active"][mask]
                  - partial_bare["sample_active"][m2])
    # 容差 = save_dt=0.02 s 下采样相位上界 (远小于模型 RMSE 2.4-3.0 C)
    assert float(np.max(diff)) < 0.05


def test_partial_insulated_equals_full_slice(partial_insulated,
                                             full_insulated):
    mask = (full_insulated["t_original"] >= 100 - 1e-9) & \
        (full_insulated["t_original"] <= 200 + 1e-9)
    m2 = (partial_insulated["t_original"] >= 100 - 1e-9) & \
        (partial_insulated["t_original"] <= 200 + 1e-9)
    assert np.array_equal(partial_insulated["t_original"][m2],
                          full_insulated["t_original"][mask])
    diff = np.abs(full_insulated["sample_active"][mask]
                  - partial_insulated["sample_active"][m2])
    assert float(np.max(diff)) < 0.05


def test_end_s_limits_simulation(partial_bare):
    assert partial_bare["sim_end_s"] == pytest.approx(200.0, abs=1e-9)
    assert partial_bare["last_recorded_s"] > 200.0  # 源数据更长, 未模拟


# ---------------------------------------------------------------
# 21-23. 时间轴
# ---------------------------------------------------------------

def test_original_time_retained(partial_bare):
    assert partial_bare["t_original"][0] == pytest.approx(
        partial_bare["window_start_s"], abs=1e-9)
    assert partial_bare["t_original"][0] > 0.0  # 记录时间, 非行号


def test_simulation_time_referenced_to_true_start(partial_bare, full_bare):
    # 模拟相对时间 = original - 完整源迹线首点
    first = full_bare["first_recorded_s"]
    assert partial_bare["t_sim"][0] == pytest.approx(
        partial_bare["t_original"][0] - first, abs=1e-9)
    assert partial_bare["t_sim"][0] > 0.0  # 不重置为 0
    assert full_bare["t_sim"][0] == 0.0


def test_analysis_time_starts_at_display_start(partial_bare):
    assert partial_bare["t_analysis"][0] == 0.0
    assert partial_bare["t_analysis"][-1] == pytest.approx(
        partial_bare["duration_s"], abs=1e-9)


# ---------------------------------------------------------------
# 24-27. 静态输出文件
# ---------------------------------------------------------------

def test_bare_png_created(main_bare):
    png = main_bare / "sample_temperature_prediction_bare.png"
    assert png.is_file() and png.stat().st_size > 0


def test_insulated_png_created(main_insulated):
    png = main_insulated / "sample_temperature_prediction_insulated.png"
    assert png.is_file() and png.stat().st_size > 0


def test_both_comparison_png_created(main_both_output):
    png = main_both_output / "sample_temperature_bare_vs_insulated.png"
    assert png.is_file() and png.stat().st_size > 0


def test_pdfs_created(main_bare, main_insulated, main_both_output):
    assert (main_bare / "sample_temperature_prediction_bare.pdf").is_file()
    assert (main_insulated /
            "sample_temperature_prediction_insulated.pdf").is_file()
    assert (main_both_output /
            "sample_temperature_bare_vs_insulated.pdf").is_file()


# ---------------------------------------------------------------
# 28-29. CSV / summary
# ---------------------------------------------------------------

def test_csv_schema_all_modes(main_bare, main_insulated, main_both_output):
    bare = pd.read_csv(main_bare / "sample_temperature_prediction_bare.csv")
    assert set(bare.columns) == {
        "original_time_s", "simulation_time_s", "analysis_time_s",
        "measured_internal_C", "predicted_sample_bare_C",
        "predicted_top_bare_raw_C"}
    ins = pd.read_csv(
        main_insulated / "sample_temperature_prediction_insulated.csv")
    assert set(ins.columns) == {
        "original_time_s", "simulation_time_s", "analysis_time_s",
        "measured_internal_C", "predicted_sample_insulated_C",
        "predicted_topCOC_air_interface_C", "predicted_outer_PDMS_C"}
    both = pd.read_csv(
        main_both_output / "sample_temperature_bare_vs_insulated.csv")
    assert set(both.columns) == {
        "original_time_s", "simulation_time_s", "analysis_time_s",
        "measured_internal_C", "predicted_sample_bare_C",
        "predicted_sample_insulated_C",
        "delta_sample_insulated_minus_bare_C",
        "predicted_top_bare_raw_C", "predicted_topCOC_insulated_C",
        "predicted_outer_PDMS_insulated_C"}


def test_summary_generated_all_modes(main_bare, main_insulated,
                                     main_both_output):
    for d in (main_bare, main_insulated, main_both_output):
        s = d / "sample_temperature_summary.txt"
        assert s.is_file() and s.stat().st_size > 0


# ---------------------------------------------------------------
# 30. both 模式 delta
# ---------------------------------------------------------------

def test_both_mode_delta_correct(full_both):
    assert np.allclose(full_both["delta_sample_ins_minus_bare"],
                       full_both["sample_insulated"]
                       - full_both["sample_bare"])
    c = full_both["comparison"]
    assert c["sample_max_increase_C"] == pytest.approx(
        full_both["stats_insulated"]["sample_max_C"]
        - full_both["stats_bare"]["sample_max_C"], abs=1e-12)
    assert c["sample_mean_increase_C"] == pytest.approx(
        full_both["stats_insulated"]["sample_mean_C"]
        - full_both["stats_bare"]["sample_mean_C"], abs=1e-12)
    assert c["sample_median_increase_C"] == pytest.approx(
        full_both["stats_insulated"]["sample_median_C"]
        - full_both["stats_bare"]["sample_median_C"], abs=1e-12)
    assert c["max_instantaneous_delta_C"] == pytest.approx(
        float(np.max(full_both["delta_sample_ins_minus_bare"])), abs=1e-12)
    # 绝缘应升高样品温度 (物理趋势; 用实际结果验证 delta > 0)
    assert c["sample_max_increase_C"] > 0.0


# ---------------------------------------------------------------
# 31-34. 交互复用 / 样品阈值
# ---------------------------------------------------------------

def test_draggable_hlines_reused():
    assert "from thermal_model.utilities.draggable_hlines import" in SRC
    assert callable(psf.find_intersections)
    assert isinstance(psf.DraggableHLine, type)
    t = np.array([0.0, 1.0, 2.0])
    T = np.array([0.0, 1.0, 2.0])
    assert psf.find_intersections(t, T, 0.5) == pytest.approx([0.5])


def test_bare_threshold_timing_references_sample():
    s = psf.compute_threshold_timing(np.array([0.0, 1.0]),
                                     np.array([80.0, 90.0]), 85.0)
    assert s["applied_to"] == "PREDICTED SAMPLE"
    # 单模式交互线 y_data = 该模式样品
    assert 'y_data = r["sample_active"]' in SRC


def test_insulated_threshold_timing_references_sample():
    # both 模式公共阈值必须同时持有两条样品曲线
    assert 'y_bare=r["sample_bare"]' in SRC
    assert 'y_ins=r["sample_insulated"]' in SRC
    assert issubclass(psf.BothSampleThresholdHLine, psf.DraggableHLine)


def test_both_common_threshold_reports_both_durations():
    fig, ax = plt.subplots()
    t = np.linspace(0.0, 10.0, 51)
    bare = 60.0 + 20.0 * np.sin(t)
    ins = bare + 2.0
    line = psf.BothSampleThresholdHLine(
        ax, y_init=80.0, x_data=t, y_bare=bare, y_ins=ins,
        color="darkorange", label_prefix="Threshold")
    txt = line.stats_text.get_text()
    assert "Threshold:" in txt
    assert "Bare sample >= threshold:" in txt
    assert "Insulated sample >= threshold:" in txt
    plt.close(fig)


# ---------------------------------------------------------------
# 35. 时间戳感知
# ---------------------------------------------------------------

def test_timestamp_aware_threshold_retained():
    t = np.array([0.0, 1.0, 3.0, 4.0])   # 非均匀 dt
    T = np.array([80.0, 90.0, 90.0, 85.0])
    s = psf.compute_threshold_timing(t, T, 85.0)
    assert s["time_above_s"] == pytest.approx(3.5, abs=1e-9)
    assert s["time_above_s"] != 3.0  # 绝不是 3 点 x 1 s


# ---------------------------------------------------------------
# 36-37. 无拟合 / 无时移
# ---------------------------------------------------------------

def test_no_parameter_fitting():
    assert "curve_fit" not in SRC
    assert "minimize" not in SRC
    assert "least_squares" not in SRC
    assert "scipy.optimize" not in SRC
    assert "grid_search" not in SRC.lower()


def test_no_time_shift_optimization():
    assert "correlate" not in SRC
    assert "xcorr" not in SRC
    assert "time_shift" not in SRC


# ---------------------------------------------------------------
# 38-39. 科学状态文档
# ---------------------------------------------------------------

def test_insulated_documented_as_forward_extension_not_validated(
        main_insulated):
    # 否定形式声明: 尚未针对实测绝缘 Top COC 做独立验证 (源码多行拼接)
    assert "has not yet been " in SRC
    assert "independently validated against measured insulated Top COC" in SRC
    # 不存在"已经独立验证"的正面断言
    assert "was independently validated" not in SRC
    s = (main_insulated / "sample_temperature_summary.txt").read_text(
        encoding="utf-8")
    assert "forward extension" in s
    assert "not yet been independently validated" in s


def test_bare_validation_status_documented(main_bare):
    assert "externally validated" in SRC
    assert "1.84 C" in SRC
    s = (main_bare / "sample_temperature_summary.txt").read_text(
        encoding="utf-8")
    assert "externally validated" in s
    assert "0.6368" in s


# ---------------------------------------------------------------
# 40. 权威文件未修改 / 输入未修改
# ---------------------------------------------------------------

def test_no_authoritative_model_files_modified():
    for f in FROZEN_FILES:
        assert _sha(PROJECT_ROOT / f) == HASHES_BEFORE[f], f


def test_input_workbook_unchanged():
    assert _sha(TEST_WORKBOOK) == WORKBOOK_HASH_BEFORE
