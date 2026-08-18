"""
直接内置传感器边界 FDM 对比模型测试 (sample and internal sensor T in one plot.py)。

覆盖 (任务 17 的 11 项):
1.  真实 Excel 风格 Zone 1 列加载 (含空格折叠);
2.  前 300 行截断与参考脚本行为一致;
3.  时间处理保留 Relative time(s);
4.  T_internal 正确重采样到 FDM 时间轴;
5.  直接边界精确等于重采样后的 T_internal;
6.  无稳态校准应用;
7.  无 tau 滤波应用;
8.  FDM 接收 T_internal_fdm 作为边界;
9.  改变校准参数 (外部) 无法改变直接模型边界;
10. 直接模型输出数组保持对齐;
11. 样品对比指标函数在合成数据上结果正确。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "sample and internal sensor T in one plot.py"

REAL_XLSX = (
    PROJECT_ROOT.parent / "Calibration"
    / "08.12 pm_DOE 11 faster_zone1_temperature_analysis.xlsx"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "direct_internal_fdm", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = None


@pytest.fixture(scope="module")
def m():
    global mod
    if mod is None:
        mod = load_module()
    return mod


# ---------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------

def _make_protocol_xlsx(path, column="Zone 1 Avg (°C)", values=None,
                        with_rel_time=False):
    """构造类真实文件: 单空格列名 + Relative time(s) 时间列。"""
    if values is None:
        values = [30.0, 40.0, 50.0, 40.0, 30.0]
    df = pd.DataFrame({column: values})
    if with_rel_time:
        df.insert(0, "Relative time(s)",
                  np.cumsum([0.0] + [1.009] * (len(values) - 1)))
    df.to_excel(path, index=False)
    return path


# ===============================================================
# 1. 真实 Excel 风格 Zone 1 列加载
# ===============================================================

def test_load_real_style_column_single_space(m, tmp_path):
    """单空格列 'Zone 1 Avg (°C)' 经空格折叠匹配默认三空格列名。"""
    p = _make_protocol_xlsx(tmp_path / "p.xlsx", column="Zone 1 Avg (°C)")
    t, T_int, resolved = m.load_protocol_from_excel(
        p, column="Zone 1   Avg (°C)"
    )
    assert resolved == "Zone 1 Avg (°C)"
    np.testing.assert_allclose(T_int, [30, 40, 50, 40, 30])


def test_load_real_file_if_present(m):
    """若真实 Excel 存在, 验证可加载 (开发环境跳过)。"""
    if not REAL_XLSX.is_file():
        pytest.skip("真实 Excel 文件不存在")
    t, T_int, resolved = m.load_protocol_from_excel(
        str(REAL_XLSX), column="Zone 1   Avg (°C)",
        sheet="Extracted_Data", time_col="Relative time(s)"
    )
    assert len(t) > 100
    assert "Zone 1" in resolved
    assert np.all(np.diff(t) > 0)


def test_load_protocol_missing_column_raises(m, tmp_path):
    p = _make_protocol_xlsx(tmp_path / "p.xlsx")
    with pytest.raises(KeyError):
        m.load_protocol_from_excel(p, column="No Such Column")


# ===============================================================
# 2. 前 300 行截断 (与参考脚本行为一致)
# ===============================================================

def test_truncate_keeps_first_300_rows(m):
    rng = np.random.default_rng(1)
    t = np.arange(400, dtype=float)
    T = 25.0 + rng.normal(0, 0.1, 400)
    t_cut, T_cut, n_used = m.truncate_protocol(t, T, max_rows=300)
    assert n_used == 300
    np.testing.assert_allclose(t_cut, t[:300])
    np.testing.assert_allclose(T_cut, T[:300])


def test_truncate_preserves_pairs(m):
    t = np.array([0.0, 1.009, 2.074, 3.122])
    T = np.array([22.1, 22.2, 22.3, 22.4])
    t_cut, T_cut, n_used = m.truncate_protocol(t, T, max_rows=2)
    assert n_used == 2
    np.testing.assert_allclose(t_cut, [0.0, 1.009])
    np.testing.assert_allclose(T_cut, [22.1, 22.2])


# ===============================================================
# 3. 时间处理保留 Relative time(s)
# ===============================================================

def test_relative_time_preserved(m, tmp_path):
    p = _make_protocol_xlsx(tmp_path / "p.xlsx", with_rel_time=True)
    t, T_int, _ = m.load_protocol_from_excel(
        p, column="Zone 1 Avg (°C)", time_col="Relative time(s)"
    )
    # 非均匀时间轴 (1.009 s 间隔) 被保留, 不假设 1 Hz
    assert t[1] == pytest.approx(1.009)
    assert t[4] == pytest.approx(4.036, abs=1e-9)
    assert len(t) == 5


def test_no_time_col_generates_by_dt(m, tmp_path):
    p = _make_protocol_xlsx(tmp_path / "p.xlsx")
    t, _, _ = m.load_protocol_from_excel(p, column="Zone 1 Avg (°C)")
    np.testing.assert_allclose(t, [0, 1, 2, 3, 4])


# ===============================================================
# 4+5. 直接脚本复用共享求解器 (无独立重采样实现)
# ===============================================================

def test_direct_script_has_no_independent_resample(m):
    """直接脚本不再定义 resample_to_fdm_time; 插值由共享求解器负责。"""
    assert not hasattr(m, "resample_to_fdm_time")


def test_direct_script_uses_shared_solver(m):
    """直接脚本调用 heat_model.run_simulation, 边界为 T_internal。"""
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "heat_model.run_simulation" in src
    assert "bottom_temperature_C=T_internal" in src


# ===============================================================
# 6+7. 无稳态校准 / 无 tau 滤波
# ===============================================================

def test_no_calibration_functions_in_module(m):
    """模块不应包含校准/动态表面模型函数或参数。"""
    assert not hasattr(m, "apply_surface_calibration")
    assert not hasattr(m, "apply_dynamic_surface_model")
    assert not hasattr(m, "load_surface_calibration")
    assert not hasattr(m, "validate_tau_eff")


def test_no_calibration_or_tau_references_in_source(m):
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    # 不允许出现校准方程或 tau 滤波 (注释说明除外, 检查可执行代码关键字)
    assert "0.950490" not in src
    assert "1.811586" not in src
    assert "tau_eff" not in src.replace("tau", "tau")  # 全文件无 tau_eff
    assert "--tau-eff" not in src
    assert "--calibration" not in src


# ===============================================================
# 8. 直接脚本不再包含重复 FDM 实现
# ===============================================================

def test_direct_script_has_no_duplicate_fdm(m):
    """直接脚本不含材料常量 / 网格构造 / FDM 系数 / 时间循环。"""
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    # 材料常量
    for tok in ("rho_coc", "k_coc", "cp_coc", "rho_pdms", "k_pdms"):
        assert tok not in src, tok
    # 几何 / 网格 (层厚度常量与网格构造)
    for tok in ("L_coc_bot", "L_sample", "L_oil", "L_coc_top", "L_air",
                "L_pdms", "def make_layer", "dx_fine", "dx_air", "dx_pdms"):
        assert tok not in src, tok
    # FDM 系数与时间循环
    for tok in ("k_half = 2 * k[:-1]", "c_c * T[1:-1]", "bc_A * T[-2]",
                "T[0]    = T_internal_fdm[n]"):
        assert tok not in src, tok


# ===============================================================
# 9. 外部校准参数无法改变直接模型边界
# ===============================================================

def test_boundary_independent_of_external_calibration(m):
    """CLI 无校准参数; 直接边界只依赖 T_internal。"""
    args = m.parse_args(["--protocol-xlsx", "dummy.xlsx"])
    assert not hasattr(args, "calibration_a")
    assert not hasattr(args, "calibration_b")
    assert not hasattr(args, "tau_eff")


# ===============================================================
# 10. 输出数组对齐 (共享求解器结果)
# ===============================================================

def test_output_arrays_aligned(m):
    """共享求解器返回的下采样数组长度一致且对应同一时间轴。"""
    import heat_model
    t_proto = np.arange(11, dtype=float)
    T_int = 25.0 + np.sin(t_proto)
    result = heat_model.run_simulation(
        time_s=t_proto, bottom_temperature_C=T_int,
        materials=heat_model.DEFAULT_MATERIALS,
        layers=heat_model.DEFAULT_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    n = len(result["t_array"])
    assert len(result["T_bottom_arr"]) == n
    assert len(result["T_sample_arr"]) == n
    assert len(result["T_top_arr"]) == n
    assert n > 0


# ===============================================================
# 11. 样品对比指标函数 (合成数据)
# ===============================================================

def test_compute_sample_comparison_metrics_known_values(m):
    t = np.array([0.0, 1.0, 2.0, 3.0])
    direct = np.array([31.0, 32.0, 33.0, 34.0])
    calibrated = np.array([30.0, 31.0, 32.0, 33.0])
    metrics = m.compute_sample_comparison_metrics(t, direct, calibrated)
    assert metrics["n"] == 4
    assert metrics["mean_signed"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["max_abs"] == pytest.approx(1.0)
    assert metrics["max_abs_time"] == pytest.approx(0.0)
    assert metrics["calibrated_max"] == pytest.approx(33.0)
    assert metrics["direct_max"] == pytest.approx(34.0)
    assert metrics["max_difference"] == pytest.approx(1.0)


def test_compute_sample_comparison_metrics_nonuniform(m):
    t = np.array([0.0, 5.0, 10.0])
    direct = np.array([50.0, 52.0, 54.0])
    calibrated = np.array([50.0, 49.0, 48.0])
    metrics = m.compute_sample_comparison_metrics(t, direct, calibrated)
    d = np.array([0.0, 3.0, 6.0])
    assert metrics["mean_signed"] == pytest.approx(d.mean())
    assert metrics["mae"] == pytest.approx(np.abs(d).mean())
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean(d ** 2)))
    assert metrics["max_abs"] == pytest.approx(6.0)
    assert metrics["max_abs_time"] == pytest.approx(10.0)
    # calibrated_max = max([50,49,48]) = 50; direct_max = max([50,52,54]) = 54
    assert metrics["calibrated_max"] == pytest.approx(50.0)
    assert metrics["direct_max"] == pytest.approx(54.0)
    assert metrics["max_difference"] == pytest.approx(4.0)


def test_compute_metrics_nan_handling(m):
    t = np.array([0.0, 1.0, 2.0])
    direct = np.array([30.0, np.nan, 32.0])
    calibrated = np.array([28.0, 29.0, 30.0])
    metrics = m.compute_sample_comparison_metrics(t, direct, calibrated)
    assert metrics["n"] == 2  # NaN 被排除
    assert np.isfinite(metrics["mean_signed"])


def test_align_and_compare_writes_outputs(m, tmp_path):
    """对比 CSV + 图 + 指标在合成输入上正确生成。"""
    out_dir = tmp_path / "cmp"
    out_dir.mkdir()
    direct_csv = tmp_path / "direct.csv"
    cal_csv = tmp_path / "cal.csv"
    t = np.arange(0, 10.1, 1.0)
    pd.DataFrame({"time_s": t, "T_internal_boundary_C": t + 30,
                  "T_sample_direct_C": t + 31}).to_csv(direct_csv, index=False)
    pd.DataFrame({"time_s": t, "T_internal_C": t + 30,
                  "T_surface_equilibrium_C": t + 30.5,
                  "T_surface_dynamic_C": t + 30.5,
                  "T_sample_C": t + 30}).to_csv(cal_csv, index=False)

    metrics, cmp_df = m.align_and_compare(direct_csv, cal_csv, out_dir)
    assert metrics is not None
    assert cmp_df is not None
    assert (out_dir / "sample_temperature_model_comparison.csv").is_file()
    assert (out_dir / "sample_temperature_model_comparison.png").is_file()
    # 差异 = direct - calibrated = 1.0
    assert metrics["mean_signed"] == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(
        cmp_df["sample_difference_direct_minus_calibrated_C"], 1.0, atol=1e-6
    )


def test_align_and_compare_missing_files_returns_none(m, tmp_path):
    metrics, cmp_df = m.align_and_compare(
        tmp_path / "no_direct.csv", tmp_path / "no_cal.csv", tmp_path
    )
    assert metrics is None and cmp_df is None
