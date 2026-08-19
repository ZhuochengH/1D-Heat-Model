"""
V3 修正时间目标标定 + 时间轴插值回归保护测试。

覆盖 (任务 30 的 15 项):
1.  时间插值查询轴 = 实测时间;
2.  实测温度值不能作为插值查询坐标 (合成回归: 旧 bug 必失败);
3.  0.068/9200 修正目标参考指标 ~7.4345 在容差内;
4.  V3 使用修正目标;
5.  V3 输出目录隔离;
6.  V1 不变 (只读检查);
7.  V2 不变 (只读检查);
8.  历史 notice 文件生成;
9.  legacy 配置显式标记 provisional;
10. 旧名义别名不能生成最终标定 (valid_for_final_calibration=False);
11. V3 粗网格正确;
12. V3 细网格逻辑正确;
13. T_sample 不入目标;
14. 无额外自由参数;
15. 最终迹线残差符号正确。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import heat_model
from heat_model import DEFAULT_MATERIALS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN = PROJECT_ROOT / "scan_effective_thermal_parameters.py"
CONFIG = PROJECT_ROOT / "calibrated_model_config.py"
RUNNER = PROJECT_ROOT / "run_calibrated_thermal_model.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scan = load_module(SCAN, "scan_v3")
cfg = load_module(CONFIG, "cfg_v3")
runner = load_module(RUNNER, "runner_v3")

_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_TINT = np.array([30.0, 40.0, 55.0, 70.0, 62.0, 50.0])
_TTOP = _TINT - 2.0


# ===============================================================
# 1-2. 时间轴插值回归保护
# ===============================================================

def test_authoritative_helper_queries_measurement_time():
    """合成: FDM_time=[0..60], FDM_signal=time -> 在测量时间处应为 [10,20,30]。"""
    fdm_time = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    fdm_signal = fdm_time  # 信号 = 时间本身
    meas_time = np.array([10.0, 20.0, 30.0])
    meas_temp = np.array([40.0, 50.0, 60.0])  # 温度值 (绝不能作为查询轴)
    out = scan.sample_prediction_at_measurement_times(
        meas_time, fdm_time, fdm_signal)
    np.testing.assert_array_equal(out, [10.0, 20.0, 30.0])
    # 旧 bug 语义 (用温度值查询) 会得到 [40,50,60] —— 回归证明不同
    assert not np.allclose(out, meas_temp)


def test_authoritative_helper_validates_monotonic():
    with pytest.raises(ValueError, match="严格单调递增"):
        scan.sample_prediction_at_measurement_times(
            np.array([10.0, 5.0, 30.0]), np.arange(61.0), np.arange(61.0))
    with pytest.raises(ValueError, match="严格单调递增"):
        scan.sample_prediction_at_measurement_times(
            np.array([10.0, 20.0, 30.0]), np.arange(61.0)[::-1],
            np.arange(61.0))


def test_scan_objective_uses_measurement_time():
    """evaluate_point 目标: 查询轴 = 实测时间 (合成协议 t 即测量时间)。"""
    row = scan.evaluate_point(0.14, 1400.0, _T, _TINT, _TTOP)
    mats = scan.make_candidate_materials(0.14, 1400.0)
    res = heat_model.run_simulation(
        _T, _TINT, mats, heat_model.BARE_TOP_COC_LAYERS,
        h_conv=scan.H_CONV, T_air_ambient=scan.T_AMB, save_dt=scan.SAVE_DT,
        T_initial_C=float(_TINT[0]),
    )
    pred = scan.sample_prediction_at_measurement_times(
        _T, res["t_array"], res["T_top_surface_arr"])
    expected = float(np.sqrt(np.mean((pred - _TTOP) ** 2)))
    assert row["RMSE_C"] == pytest.approx(expected, abs=1e-10)


# ===============================================================
# 3. 0.068/9200 修正目标参考
# ===============================================================

def test_legacy_point_corrected_metrics():
    """0.068/9200 修正目标指标 ≈ RMSE 7.4345 / MAE 6.0083 / mean +2.2075。"""
    import pandas as pd
    df = pd.read_csv(
        PROJECT_ROOT / "temperature_alignment_output" / "72C"
        / "aligned_internal_top_temperature.csv")
    t = df.time_s.to_numpy(); tint = df.T_internal_interpolated_C.to_numpy()
    ttop = df.T_top_measured_C.to_numpy()
    row = scan.evaluate_point(0.068, 9200.0, t, tint, ttop)
    assert row["RMSE_C"] == pytest.approx(7.4345, abs=0.01)
    assert row["MAE_C"] == pytest.approx(6.0083, abs=0.01)
    assert row["mean_residual_C"] == pytest.approx(2.2075, abs=0.01)


# ===============================================================
# 4-5. V3 目标 / 输出隔离
# ===============================================================

def test_v3_grid_constants():
    assert scan.V3_K_GRID == [0.005, 0.008, 0.012, 0.018, 0.027, 0.040,
                              0.060, 0.080, 0.120, 0.160, 0.200, 0.240]
    assert scan.V3_CP_GRID == [800.0, 1200.0, 1800.0, 2600.0, 4000.0,
                               6000.0, 8000.0, 10000.0]
    assert len(scan.product_grid(scan.V3_K_GRID, scan.V3_CP_GRID)) == 96


def test_v3_stage_writes_only_into_v3_dir(tmp_path):
    out = tmp_path / "v3out"
    out.mkdir()
    scan.run_stage_points([(0.005, 800.0), (0.008, 1200.0)],
                          _T, _TINT, _TTOP, "v3_coarse_scan.csv", out, "t")
    assert (out / "v3_coarse_scan.csv").is_file()
    # 仓库 parameter_scan_output 无新写入 (只读检查)
    ps = PROJECT_ROOT / "parameter_scan_output"
    if ps.is_dir():
        before = {p.name: p.stat().st_size
                  for p in ps.rglob("*") if p.is_file()}
        out2 = tmp_path / "out2"; out2.mkdir()
        scan.run_stage_points([(0.012, 1800.0)], _T, _TINT, _TTOP,
                              "v3_coarse_scan.csv", out2, "t")
        after = {p.name: p.stat().st_size
                 for p in ps.rglob("*") if p.is_file()}
        assert before == after


# ===============================================================
# 6-8. V1/V2 不变 + notice 文件
# ===============================================================

def test_notice_files_exist():
    v1_notice = (PROJECT_ROOT / "parameter_scan_output" / "72C"
                 / "LEGACY_OBJECTIVE_NOTICE_V1.txt")
    v2_notice = (PROJECT_ROOT / "parameter_scan_output" / "72C"
                 / "system_effective_extended_v2"
                 / "LEGACY_OBJECTIVE_NOTICE_V2.txt")
    prov_notice = (PROJECT_ROOT / "calibrated_model_output" / "72C_nominal_v1"
                   / "PROVISIONAL_LEGACY_SELECTED_PARAMS_NOTICE.txt")
    assert v1_notice.is_file()
    assert v2_notice.is_file()
    assert prov_notice.is_file()
    v1_txt = v1_notice.read_text(encoding="utf-8")
    v2_txt = v2_notice.read_text(encoding="utf-8")
    prov_txt = prov_notice.read_text(encoding="utf-8")
    assert "NOT VALID FOR FINAL CALIBRATION" in v1_txt
    assert "NOT VALID FOR FINAL CALIBRATION" in v2_txt
    assert "NOT FINAL" in prov_txt and "HISTORICAL" in prov_txt


def test_v1_v2_historical_files_unchanged_since_checkpoint():
    """V1/V2 数值文件保持 (完整性在任务运行前后用 SHA256 清单校验; 此处抽查)。"""
    v2_comb = (PROJECT_ROOT / "parameter_scan_output" / "72C"
               / "system_effective_extended_v2"
               / "extended_combined_scan.csv")
    df = pd.read_csv(v2_comb)
    row = df[(df.k_eff_W_mK == 0.068) & (df.cp_eff_J_kgK == 9200.0)]
    assert len(row) == 1
    assert row.iloc[0]["RMSE_C"] == pytest.approx(4.744872647981289, abs=1e-9)


# ===============================================================
# 9-10. legacy 配置标记
# ===============================================================

def test_legacy_config_explicitly_provisional():
    cal = cfg.LEGACY_OBJECTIVE_PROVISIONAL_CALIBRATION
    assert cal.k_eff_W_mK == 0.068
    assert cal.cp_eff_J_kgK == 9200.0
    assert cal.status == "historical_provisional"
    assert cal.valid_for_final_calibration is False
    assert cal.selection_objective == "legacy_temperature_as_time_query"


def test_nominal_alias_is_v3_accepted():
    cal = cfg.NOMINAL_BARE_TOP_CALIBRATION_V1
    assert cal is not cfg.LEGACY_OBJECTIVE_PROVISIONAL_CALIBRATION
    assert cal.valid_for_final_calibration is True
    assert cal.k_eff_W_mK == 0.0165 and cal.cp_eff_J_kgK == 900.0


def test_legacy_factory_does_not_mutate_defaults():
    before = {n: (m.k_W_mK, m.rho_kg_m3, m.cp_J_kgK)
              for n, m in DEFAULT_MATERIALS.items()}
    mats = cfg.make_nominal_calibrated_materials()
    after = {n: (m.k_W_mK, m.rho_kg_m3, m.cp_J_kgK)
             for n, m in DEFAULT_MATERIALS.items()}
    assert before == after
    assert mats["COC"].k_W_mK == 0.0165  # V3 接受值
    assert DEFAULT_MATERIALS["COC"].k_W_mK == 0.13


# ===============================================================
# 11-12. V3 网格 / 细网格逻辑
# ===============================================================

def test_v3_fine_skipped_on_boundary():
    assert not scan.is_interior(0.005, 4000.0, scan.V3_K_GRID, scan.V3_CP_GRID)
    assert not scan.is_interior(0.018, 10000.0, scan.V3_K_GRID, scan.V3_CP_GRID)


def test_v3_fine_grid_from_neighbors_11x11():
    k_grid, cp_grid = scan.fine_grid_from_neighbors(
        scan.V3_K_GRID, scan.V3_CP_GRID, 0.018, 4000.0, n=11,
        k_limits=scan.V3_K_LIMITS, cp_limits=scan.V3_CP_LIMITS)
    assert len(k_grid) == 11
    assert len(cp_grid) == 11
    assert k_grid[0] == pytest.approx(0.012) and k_grid[-1] == pytest.approx(0.027)
    assert cp_grid[0] == pytest.approx(2600.0) and cp_grid[-1] == pytest.approx(6000.0)
    assert len(scan.product_grid(k_grid, cp_grid)) == 121


def test_v3_limits():
    assert scan.V3_K_LIMITS == (0.002, 0.35)
    assert scan.V3_CP_LIMITS == (500.0, 15000.0)


# ===============================================================
# 13-14. 目标仅含顶部 / 无额外参数
# ===============================================================

def test_sample_excluded_from_objective():
    src = SCAN.read_text(encoding="utf-8")
    seg = src[src.index("def evaluate_point"):src.index("def evaluate_point_safe")]
    assert "T_sample" not in seg


def test_no_extra_free_parameters():
    src = SCAN.read_text(encoding="utf-8")
    for token in ("h_conv_opt", "tau_fit", "offset_fit", "scipy.optimize"):
        assert token not in src


# ===============================================================
# 15. 最终迹线残差符号
# ===============================================================

def test_final_trace_residual_sign(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    metrics, trace = runner.run_nominal(_T, _TINT, _TTOP, out)
    np.testing.assert_allclose(
        trace["top_residual_C"],
        trace["T_top_predicted_C"] - trace["T_top_measured_C"],
        rtol=0, atol=1e-12)
