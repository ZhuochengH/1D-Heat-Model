"""
底部边界解耦 / 单一求解器架构测试 (heat_model.run_simulation)。

覆盖 (任务 20 的核心项):
1.  run_simulation 接受显式 bottom_temperature_C;
2.  求解器施加的底部边界 == 提供的迹线插值到 FDM 网格;
3.  直接模式不施加 a/b 校准;
4.  直接模式不施加 tau 滤波;
5.  校准预处理在求解器之外;
6.  校准模式把准备好的表面迹线交给同一求解器;
7.  校准模式 旧 vs 新 数值回归 (见 test_material_layer_configuration);
8.  直接模式 旧 vs 新 数值回归 (本文件);
12. 两种模式复用 heat_model.run_simulation;
13/14. 修改中心 COC k/cp 更新所有 COC 层;
15. 未来直接边界拟合 API 可用;
    + 边界时间序列校验 (长度/有限性/单调)。
"""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from thermal_model.core import fv_reference
from thermal_model.core import heat_model
from thermal_model.core.heat_model import DEFAULT_LAYERS, DEFAULT_MATERIALS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATED_SCRIPT = PROJECT_ROOT / "workflows/diagnostics/sample and heater T in one plot.py"
DIRECT_SCRIPT = PROJECT_ROOT / "workflows/diagnostics/sample and internal sensor T in one plot.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cal = None


@pytest.fixture(scope="module")
def cal():
    global _cal
    if _cal is None:
        _cal = _load(CALIBRATED_SCRIPT, "calibrated_fdm_iface")
    return _cal


_SYNTH_T = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
_SYNTH_TINT = np.array([30.0, 95.0, 95.0, 40.0, 40.0, 60.0])


# ===============================================================
# 1. run_simulation 接受显式 bottom_temperature_C
# ===============================================================

def test_run_simulation_signature():
    params = list(inspect.signature(heat_model.run_simulation).parameters)
    assert "time_s" in params
    assert "bottom_temperature_C" in params
    assert "materials" in params
    assert "layers" in params
    # 求解器不含校准 / 动态模型 / 协议解析参数
    for forbidden in ("a", "b", "tau_eff", "T_internal", "t_protocol",
                      "calibration_file", "protocol_xlsx"):
        assert forbidden not in params, forbidden


def test_run_simulation_accepts_explicit_bottom_temperature():
    result = heat_model.run_simulation(
        time_s=_SYNTH_T, bottom_temperature_C=_SYNTH_TINT,
        materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    for key in ("time_fdm", "bottom_temperature_fdm", "t_array",
                "T_bottom_arr", "T_sample_arr", "T_top_arr", "dt", "Nt"):
        assert key in result, key


# ===============================================================
# 2. 底部边界 == 提供的迹线插值到 FDM 网格
# ===============================================================

def test_bottom_boundary_equals_interpolated_supply():
    result = heat_model.run_simulation(
        time_s=_SYNTH_T, bottom_temperature_C=_SYNTH_TINT,
        materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    expected = np.interp(result["time_fdm"], _SYNTH_T, _SYNTH_TINT)
    np.testing.assert_array_equal(result["bottom_temperature_fdm"], expected)


def test_bottom_boundary_imposed_exactly_on_fdm_grid():
    """当迹线已在 FDM 网格上时, 插值为恒等 (不引入误差)。"""
    mesh, dt = heat_model.compute_stable_dt(DEFAULT_MATERIALS, DEFAULT_LAYERS)
    t_total = float(_SYNTH_T[-1]) - float(_SYNTH_T[0])
    Nt = int(t_total / dt) + 1
    time_fdm = np.linspace(0.0, t_total, Nt)
    trace = 30.0 + 2.0 * np.sin(time_fdm)
    result = heat_model.run_simulation(
        time_s=time_fdm, bottom_temperature_C=trace,
        materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    np.testing.assert_array_equal(result["bottom_temperature_fdm"], trace)


# ===============================================================
# 3+4. 直接模式无 a/b 校准、无 tau 滤波
# ===============================================================

def test_direct_mode_applies_no_calibration_or_tau():
    """求解器签名不含 a/b/tau; 底部边界就是插值后的 T_internal。"""
    a, b, tau = 0.95, 1.8, 7.3
    result = heat_model.run_simulation(
        time_s=_SYNTH_T, bottom_temperature_C=_SYNTH_TINT,
        materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    # 若施加了校准, 边界会是 a*T_internal+b; 这里必须等于原始插值
    expected = np.interp(result["time_fdm"], _SYNTH_T, _SYNTH_TINT)
    np.testing.assert_array_equal(result["bottom_temperature_fdm"], expected)
    # 不可能等于校准后的边界 (排除静默转换)
    calibrated = a * expected + b
    assert not np.allclose(result["bottom_temperature_fdm"], calibrated)
    # tau 滤波会产生平滑滞后; 恒温段末端应精确等于 T_internal 而非被平滑
    assert result["bottom_temperature_fdm"][-1] == pytest.approx(
        _SYNTH_TINT[-1], abs=1e-9
    )


# ===============================================================
# 5. 校准预处理在求解器之外
# ===============================================================

def test_calibration_functions_not_in_solver():
    assert not hasattr(heat_model, "apply_surface_calibration")
    assert not hasattr(heat_model, "apply_dynamic_surface_model")
    assert not hasattr(heat_model, "prepare_fdm_boundary")


def test_calibration_functions_in_calibrated_script(cal):
    assert hasattr(cal, "apply_surface_calibration")
    assert hasattr(cal, "apply_dynamic_surface_model")
    assert hasattr(cal, "prepare_fdm_boundary")


# ===============================================================
# 6+12. 两种模式复用同一求解器
# ===============================================================

def test_both_scripts_use_shared_solver():
    cal_src = CALIBRATED_SCRIPT.read_text(encoding="utf-8")
    direct_src = DIRECT_SCRIPT.read_text(encoding="utf-8")
    assert "heat_model.run_simulation" in cal_src
    assert "heat_model.run_simulation" in direct_src
    # 校准版把准备好的动态表面迹线作为底部边界
    assert "bottom_temperature_C=T_surface_fdm" in cal_src
    # 直接版把 T_internal 作为底部边界
    assert "bottom_temperature_C=T_internal" in direct_src


def test_two_prepared_traces_share_one_solver():
    """MODE A 与 MODE B 只是底部边界数组不同, 走同一求解器。"""
    trace_a = _SYNTH_TINT
    trace_b = np.array([40.0, 60.0, 80.0, 70.0, 50.0, 45.0])
    ra = heat_model.run_simulation(
        time_s=_SYNTH_T, bottom_temperature_C=trace_a,
        materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS, save_dt=0.1,
    )
    rb = heat_model.run_simulation(
        time_s=_SYNTH_T, bottom_temperature_C=trace_b,
        materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS, save_dt=0.1,
    )
    # 同一网格 / dt / 时间轴
    assert ra["dt"] == rb["dt"]
    np.testing.assert_array_equal(ra["time_fdm"], rb["time_fdm"])
    # 边界不同 -> 样品温度不同
    assert not np.allclose(ra["T_sample_arr"], rb["T_sample_arr"])


# ===============================================================
# 13/14. 中心修改 COC k/cp 更新所有 COC 层
# ===============================================================

def test_central_coc_k_updates_both_coc_layers():
    mats = heat_model.copy_default_materials()
    mats["COC"].k_W_mK = 0.3  # 与默认 0.13 不同
    ms = heat_model.build_layer_stack(mats, DEFAULT_LAYERS)
    coc_nodes = np.isin(ms.node_layer_index, [0, 3])  # Bottom COC + Top COC
    assert np.all(ms.k[coc_nodes] == 0.3)


def test_central_coc_cp_updates_both_coc_layers():
    mats = heat_model.copy_default_materials()
    mats["COC"].cp_J_kgK = 2500.0  # 与默认 1800 不同
    ms = heat_model.build_layer_stack(mats, DEFAULT_LAYERS)
    coc_nodes = np.isin(ms.node_layer_index, [0, 3])
    assert np.all(ms.cp[coc_nodes] == 2500.0)


# ===============================================================
# 15. 未来直接边界拟合 API
# ===============================================================

def test_future_fitting_interface():
    """k_eff/cp_eff 拟合所需的精确调用接口: 直接 T_internal 边界 + 中心改 COC。"""
    candidate_k, candidate_cp = 0.22, 2100.0
    materials = heat_model.copy_default_materials()
    materials["COC"].k_W_mK = candidate_k
    materials["COC"].cp_J_kgK = candidate_cp

    result = heat_model.run_simulation(
        time_s=_SYNTH_T,
        bottom_temperature_C=_SYNTH_TINT,   # 直接 T_internal, 无校准
        materials=materials,
        layers=DEFAULT_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )

    # 候选 COC 参数传播到所有 COC 层
    ms = heat_model.build_layer_stack(materials, DEFAULT_LAYERS)
    coc_nodes = np.isin(ms.node_layer_index, [0, 3])
    assert np.all(ms.k[coc_nodes] == candidate_k)
    assert np.all(ms.cp[coc_nodes] == candidate_cp)

    # 底部边界 = 直接 T_internal (未施加表面校准)
    expected = np.interp(result["time_fdm"], _SYNTH_T, _SYNTH_TINT)
    np.testing.assert_array_equal(result["bottom_temperature_fdm"], expected)

    # 修改 COC k 确实改变了结果 (证明参数被求解器实际使用)
    ref = heat_model.run_simulation(
        time_s=_SYNTH_T, bottom_temperature_C=_SYNTH_TINT,
        materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    assert not np.allclose(result["T_sample_arr"], ref["T_sample_arr"])


# ===============================================================
# 边界时间序列校验
# ===============================================================

def test_boundary_trace_length_mismatch_raises():
    with pytest.raises(ValueError, match="长度必须一致"):
        heat_model.run_simulation(
            time_s=np.arange(5.0),
            bottom_temperature_C=np.arange(4.0),
            materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS,
        )


def test_boundary_trace_non_monotonic_time_raises():
    with pytest.raises(ValueError, match="单调递增"):
        heat_model.run_simulation(
            time_s=np.array([0.0, 2.0, 1.5, 3.0]),
            bottom_temperature_C=np.array([30.0, 40.0, 50.0, 60.0]),
            materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS,
        )


def test_boundary_trace_non_finite_raises():
    with pytest.raises(ValueError, match="有限数值"):
        heat_model.run_simulation(
            time_s=np.array([0.0, 1.0, 2.0]),
            bottom_temperature_C=np.array([30.0, np.nan, 50.0]),
            materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS,
        )


def test_boundary_trace_too_few_points_raises():
    with pytest.raises(ValueError, match="至少需要 2 个点"):
        heat_model.run_simulation(
            time_s=np.array([0.0]),
            bottom_temperature_C=np.array([30.0]),
            materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS,
        )


# ===============================================================
# 8. 直接模式 旧 vs 新 数值回归 (逐字复刻旧直接求解器)
# ===============================================================

def _reference_old_direct_run(t_protocol, T_internal, h_conv=5.0,
                              T_air_ambient=25.0, save_dt=0.1):
    """重构前 direct-internal 脚本第 7-8 节的逐字复刻 (历史基线)。

    界面物理修正后, 修正版求解器与此历史基线不再逐位等价
    (区间导热 / 界面体积热容已修正, 见 test_interface_discretization.py)。
    """
    mesh = heat_model.build_layer_stack(DEFAULT_MATERIALS, DEFAULT_LAYERS)
    k = mesh.k
    h = mesh.h
    rho = mesh.rho
    cp = mesh.cp
    Nx = mesh.Nx
    idx_sample = mesh.idx_sample

    k_half = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])
    h_m = h[:-1]
    h_p = h[1:]
    k_m = k_half[:-1]
    k_p = k_half[1:]
    rho_int = rho[1:-1]
    cp_int = cp[1:-1]

    dt_stable = rho_int * cp_int * (h_m + h_p) / (2 * (k_p / h_p + k_m / h_m))
    dt = np.min(dt_stable) * 0.9

    t_total = float(t_protocol[-1]) - float(t_protocol[0])
    Nt = int(t_total / dt) + 1
    time_fdm = np.linspace(0.0, t_total, Nt)

    T_internal_fdm = np.interp(time_fdm, t_protocol, T_internal)

    fac = 2 * dt / ((h_m + h_p) * rho_int * cp_int)
    c_p = fac * k_p / h_p
    c_m = fac * k_m / h_m
    c_c = 1.0 - c_p - c_m

    bc_A = (k[-1] / h[-1]) / (k[-1] / h[-1] + h_conv)
    bc_B = h_conv * T_air_ambient / (k[-1] / h[-1] + h_conv)

    T = np.ones(Nx) * 25.0
    save_interval = max(1, int(save_dt / dt))
    plot_times = []
    plot_T_bottom = []
    plot_T_sample = []
    plot_T_top = []

    for n in range(Nt):
        if n % save_interval == 0:
            plot_times.append(time_fdm[n])
            plot_T_bottom.append(T_internal_fdm[n])
            plot_T_sample.append(np.mean(T[idx_sample]))
            plot_T_top.append(T[-1])

        T[0]    = T_internal_fdm[n]
        T[1:-1] = c_c * T[1:-1] + c_m * T[:-2] + c_p * T[2:]
        T[-1]   = bc_A * T[-2] + bc_B

    return {
        "t_array": np.array(plot_times),
        "T_bottom_arr": np.array(plot_T_bottom),
        "T_sample_arr": np.array(plot_T_sample),
        "T_top_arr": np.array(plot_T_top),
        "dt": dt,
        "Nt": Nt,
    }


def test_direct_internal_numerical_regression():
    """修正版 FV 求解器与独立修正版参考逐位一致;
    旧方案 (_reference_old_direct_run) 为历史基线, 结果有意不同。"""
    new = heat_model.run_simulation(
        time_s=_SYNTH_T, bottom_temperature_C=_SYNTH_TINT,
        materials=DEFAULT_MATERIALS, layers=DEFAULT_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    ref = fv_reference.corrected_run(
        DEFAULT_MATERIALS, DEFAULT_LAYERS, _SYNTH_T, _SYNTH_TINT,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )

    assert new["dt"] == pytest.approx(ref["dt"], rel=0, abs=0)
    assert new["Nt"] == ref["Nt"]
    np.testing.assert_array_equal(new["t_array"], ref["t_array"])

    tol = 1e-8
    np.testing.assert_allclose(new["T_bottom_arr"], ref["T_bottom_arr"],
                               rtol=0, atol=tol)
    np.testing.assert_allclose(new["T_sample_arr"], ref["T_sample_arr"],
                               rtol=0, atol=tol)
    np.testing.assert_allclose(new["T_top_arr"], ref["T_outer_surface_arr"],
                               rtol=0, atol=tol)

    # 历史基线保留在 _reference_old_direct_run; 界面修正后与旧方案有意不同
    old = fv_reference.old_run(
        DEFAULT_MATERIALS, DEFAULT_LAYERS, _SYNTH_T, _SYNTH_TINT,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    assert np.max(np.abs(new["T_sample_arr"] - old["T_sample_arr"])) > 1e-4
