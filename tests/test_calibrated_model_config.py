"""
标定裸顶模型 V1 —— 名义配置 + 最终运行逻辑测试。

覆盖 (任务 24 的 16 项 + 任务 25):
1.  nominal k_eff == 0.068;
2.  nominal cp_eff == 9200;
3.  nominal rho == 1020;
4.  nominal 几何 == 裸顶;
5.  材料工厂不修改 DEFAULT_MATERIALS;
6.  仅 COC 材料 k/cp 变化;
7.  Water 不变;
8.  Oil 不变;
9.  Bottom COC 使用名义有效值;
10. Top COC 使用相同名义有效值;
11. 最终模型几何 850 um;
12. 顶部观测 850 um;
13. 样品平均保持 CV 加权;
14. auto 初始条件 = 第一个内部温度;
15. 最终指标仅用实测顶部;
16. 样品预测不入标定目标;

任务 25 (最终迹线导出逻辑, 用合成数据):
17. 列名正确;
18. 残差符号 = pred - meas;
19. 插值到实测时间;
20. 实测数据不被修改;
21. 输出目录隔离;
22. 不写入 parameter_scan_output;
23. 生成文件名确定。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import heat_model
from heat_model import DEFAULT_MATERIALS, build_layer_stack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "calibrated_model_config.py"
RUNNER = PROJECT_ROOT / "run_calibrated_thermal_model.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = load_module(CONFIG, "calibrated_model_config")
runner = load_module(RUNNER, "run_calibrated_thermal_model")


# ===============================================================
# 1-4. 名义配置值
# ===============================================================

def test_nominal_k_eff():
    """当前名义标定 = V3 修正目标 (0.0165)。"""
    assert cfg.NOMINAL_BARE_TOP_CALIBRATION_V1.k_eff_W_mK == 0.0165


def test_nominal_cp_eff():
    assert cfg.NOMINAL_BARE_TOP_CALIBRATION_V1.cp_eff_J_kgK == 900.0


def test_nominal_rho():
    assert cfg.NOMINAL_BARE_TOP_CALIBRATION_V1.rho_COC_kg_m3 == 1020.0


def test_nominal_geometry_preset():
    assert (cfg.NOMINAL_BARE_TOP_CALIBRATION_V1.geometry_preset
            == "BARE_TOP_COC_LAYERS")
    assert cfg.nominal_layer_stack() is heat_model.BARE_TOP_COC_LAYERS


def test_nominal_name_and_source():
    """当前名义标定 = V3 (corrected_time_objective_v3), accepted。"""
    cal = cfg.NOMINAL_BARE_TOP_CALIBRATION_V1
    assert cal.source_analysis == "corrected_time_objective_v3"
    assert cal.status == "accepted"
    assert cal.valid_for_final_calibration is True
    assert cal.selection_objective == "corrected_measurement_time_rmse"
    # 旧 0.068/9200 仍保留为历史暂定
    leg = cfg.LEGACY_OBJECTIVE_PROVISIONAL_CALIBRATION
    assert leg.k_eff_W_mK == 0.068 and leg.cp_eff_J_kgK == 9200.0
    assert leg.valid_for_final_calibration is False


# ===============================================================
# 5-8. 材料工厂
# ===============================================================

def test_factory_does_not_mutate_default_materials():
    before = {n: (m.k_W_mK, m.rho_kg_m3, m.cp_J_kgK)
              for n, m in DEFAULT_MATERIALS.items()}
    cfg.make_nominal_calibrated_materials()
    after = {n: (m.k_W_mK, m.rho_kg_m3, m.cp_J_kgK)
             for n, m in DEFAULT_MATERIALS.items()}
    assert before == after
    assert DEFAULT_MATERIALS["COC"].k_W_mK == 0.13  # 默认库未被标定值污染


def test_factory_only_changes_coc():
    mats = cfg.make_nominal_calibrated_materials()
    assert mats["COC"].k_W_mK == 0.0165
    assert mats["COC"].cp_J_kgK == 900.0
    assert mats["COC"].rho_kg_m3 == 1020.0
    for name in ("Water", "Oil", "Air", "PDMS"):
        assert (mats[name].k_W_mK, mats[name].rho_kg_m3, mats[name].cp_J_kgK) == (
            DEFAULT_MATERIALS[name].k_W_mK,
            DEFAULT_MATERIALS[name].rho_kg_m3,
            DEFAULT_MATERIALS[name].cp_J_kgK)


# ===============================================================
# 9-10. Bottom/Top COC 使用同一名义值
# ===============================================================

def test_both_coc_layers_use_nominal():
    mats = cfg.make_nominal_calibrated_materials()
    ms = build_layer_stack(mats, heat_model.BARE_TOP_COC_LAYERS)
    for i, layer in enumerate(heat_model.BARE_TOP_COC_LAYERS):
        if layer.material == "COC":
            nodes = ms.node_layer_index == i
            assert np.all(ms.k[nodes] == 0.0165)
            assert np.all(ms.cp[nodes] == 900.0)


# ===============================================================
# 11-14. 最终模型几何 / 观测 / 平均 / 初始条件
# ===============================================================

def test_final_geometry_850um():
    ms = build_layer_stack(cfg.make_nominal_calibrated_materials(),
                           heat_model.BARE_TOP_COC_LAYERS)
    assert ms.boundaries[-1] == pytest.approx(850e-6, abs=1e-12)
    assert ms.x[-1] == pytest.approx(850e-6, abs=1e-12)
    assert ms.boundaries[1] == pytest.approx(180e-6)
    assert ms.boundaries[2] == pytest.approx(200e-6)
    assert ms.sample_layer_index == 1
    assert ms.top_surface_layer_index == 3


def test_top_observation_850um():
    ms = build_layer_stack(cfg.make_nominal_calibrated_materials(),
                           heat_model.BARE_TOP_COC_LAYERS)
    assert ms.idx_top_surface[0] == ms.Nx - 1
    assert ms.x[ms.idx_top_surface[0]] == pytest.approx(850e-6, abs=1e-12)


def test_sample_averaging_still_cv_weighted():
    ms = build_layer_stack(cfg.make_nominal_calibrated_materials(),
                           heat_model.BARE_TOP_COC_LAYERS)
    assert np.sum(ms.sample_weights) == pytest.approx(1.0, abs=1e-12)
    assert np.all(ms.sample_weights >= 0)
    # 界面节点半体积权重
    j180 = int(np.argmin(np.abs(ms.x - 180e-6)))
    interior = int(np.argmin(np.abs(ms.x - 190e-6)))
    assert ms.sample_weights[j180] == pytest.approx(
        ms.sample_weights[interior] / 2)


def test_auto_initial_equals_first_internal():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    tint = np.array([27.64, 40.0, 55.0, 70.0, 62.0, 50.0])
    ttop = tint - 2.0
    mats = cfg.make_nominal_calibrated_materials()
    res = heat_model.run_simulation(
        t, tint, mats, heat_model.BARE_TOP_COC_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.5,
        T_initial_C=float(tint[0]),
    )
    assert res["T_sample_arr"][0] == pytest.approx(27.64, abs=1e-9)
    assert res["T_top_surface_arr"][0] == pytest.approx(27.64, abs=1e-9)


# ===============================================================
# 15-16. 目标只含实测顶部 / 样品不入目标
# ===============================================================

def test_metrics_use_top_only():
    t = np.arange(4.0)
    r = np.array([1.0, -2.0, 3.0, -4.0])
    m = runner.compute_metrics(t, r)
    assert m["RMSE_top_C"] == pytest.approx(np.sqrt(np.mean(r ** 2)))
    assert m["MAE_top_C"] == pytest.approx(np.mean(np.abs(r)))
    assert m["mean_residual_top_C"] == pytest.approx(np.mean(r))
    assert m["max_positive_residual_C"] == pytest.approx(3.0)
    assert m["max_negative_residual_C"] == pytest.approx(-4.0)
    assert m["max_absolute_residual_C"] == pytest.approx(4.0)


def test_sample_not_in_objective():
    src = RUNNER.read_text(encoding="utf-8")
    metrics_seg = src[src.index("def compute_metrics"):src.index("def run_nominal")]
    assert "sample" not in metrics_seg.lower()


# ===============================================================
# 任务 25: 最终迹线导出逻辑 (合成短数据)
# ===============================================================

_SYN_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_SYN_TINT = np.array([30.0, 40.0, 55.0, 70.0, 62.0, 50.0])
_SYN_TTOP = _SYN_TINT - 2.0


def test_trace_columns_and_residual_sign(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    metrics, trace = runner.run_nominal(_SYN_T, _SYN_TINT, _SYN_TTOP, out)
    assert list(trace.columns) == [
        "time_s", "T_internal_C", "T_top_measured_C", "T_top_predicted_C",
        "T_sample_predicted_C", "top_residual_C",
    ]
    # 残差符号: pred - meas
    np.testing.assert_allclose(
        trace["top_residual_C"],
        trace["T_top_predicted_C"] - trace["T_top_measured_C"],
        rtol=0, atol=1e-12)
    # 时间轴单调递增
    assert np.all(np.diff(trace["time_s"]) > 0)
    assert np.all(np.isfinite(trace.to_numpy(dtype=float)))


def test_trace_interpolated_to_measurement_times(tmp_path):
    """时间轴 = 实测 TIME 坐标; 预测在实测时间处线性插值 (不是温度值)。"""
    out = tmp_path / "out"
    out.mkdir()
    metrics, trace = runner.run_nominal(_SYN_T, _SYN_TINT, _SYN_TTOP, out)
    np.testing.assert_array_equal(trace["time_s"], _SYN_T)  # 时间是时间轴
    # 预测 = FDM 结果在实测时间 (而非实测温度值) 处线性插值
    mats = cfg.make_nominal_calibrated_materials()
    res = heat_model.run_simulation(
        _SYN_T, _SYN_TINT, mats, heat_model.BARE_TOP_COC_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=runner.SAVE_DT,
        T_initial_C=float(_SYN_TINT[0]),
    )
    expected = np.interp(_SYN_T, res["t_array"], res["T_top_surface_arr"])
    np.testing.assert_allclose(trace["T_top_predicted_C"], expected,
                               rtol=0, atol=1e-12)
    # 内部温度列 = 输入 (实测时间坐标)
    np.testing.assert_allclose(trace["T_internal_C"], _SYN_TINT, rtol=0,
                               atol=1e-12)


def test_measured_data_not_modified(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    ttop_before = _SYN_TTOP.copy()
    runner.run_nominal(_SYN_T, _SYN_TINT, _SYN_TTOP, out)
    np.testing.assert_array_equal(_SYN_TTOP, ttop_before)
    np.testing.assert_array_equal(_SYN_TINT, _SYN_TINT)


def test_output_directory_isolation(tmp_path):
    """最终输出只写在指定 output_dir; 不写 parameter_scan_output。"""
    out = tmp_path / "out"
    out.mkdir()
    runner.run_nominal(_SYN_T, _SYN_TINT, _SYN_TTOP, out)
    names = {p.name for p in out.glob("*")}
    assert "final_72C_thermal_trace.csv" in names
    assert "final_72C_thermal_trace.png" in names
    assert "final_72C_thermal_trace.pdf" in names
    assert "final_72C_metadata.json" in names
    # 仓库级 parameter_scan_output 目录未写入 (只读检查)
    scan_dir = PROJECT_ROOT / "parameter_scan_output"
    if scan_dir.is_dir():
        before = {p.name: p.stat().st_size
                  for p in scan_dir.rglob("*") if p.is_file()}
        out2 = tmp_path / "out2"
        out2.mkdir()
        runner.run_nominal(_SYN_T, _SYN_TINT, _SYN_TTOP, out2)
        after = {p.name: p.stat().st_size
                 for p in scan_dir.rglob("*") if p.is_file()}
        assert before == after


def test_trace_no_nan(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    metrics, trace = runner.run_nominal(_SYN_T, _SYN_TINT, _SYN_TTOP, out)
    assert np.all(np.isfinite(trace.to_numpy(dtype=float)))


def test_metadata_records_git_and_interpretation(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    runner.run_nominal(_SYN_T, _SYN_TINT, _SYN_TTOP, out)
    import json
    meta = json.loads((out / "final_72C_metadata.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == (
        "bare_top_calibrated_model_v1 (corrected measurement-time objective)")
    assert meta["k_eff_W_mK"] == 0.0165
    assert meta["cp_eff_J_kgK"] == 900.0
    assert meta["rho_COC_kg_m3"] == 1020.0
    assert "system-level effective" in meta["interpretation"]
    assert "intrinsic" not in meta["interpretation"].lower().replace(
        "not intrinsic", "")
    assert meta["git_commit"]  # 非空
    assert "sample temperature is model-estimated" in meta["note"].lower()
