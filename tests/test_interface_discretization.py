"""
材料界面有限体积修正测试 (区间导热 + 界面体积热容)。

覆盖 (任务 23-29):
  - 区间导热系数 k_face: 同材料区间 / COC|Water / Water|Oil / Oil|COC /
    任意堆叠 / 不在界面区间使用端点调和平均;
  - 界面体积热容 rho_cp: 普通节点 / 等间距界面 / 非等间距界面 / 正定性 /
    界面节点不使用单侧材料全值;
  - 稳定性: 系数非负 + 归一; compute_stable_dt 与 run_simulation 一致;
  - 解析稳态: 裸顶多层串联热阻 (三对角直接求解 + 显式收敛趋势);
  - 高对比堆叠 (旧方案失败 / 新方案通过);
  - 均匀材料不变性 (新旧逐位一致);
  - 裸顶几何保持 (850 um 等);
  - 模块与独立修正版参考逐位一致; 与旧方案明显不同 (物理修正生效)。

注意: 本文件不拟合 k_eff / cp_eff; 不修改 Robin / Dirichlet 方程。
"""

import numpy as np
import pytest

from thermal_model.core import fv_reference
from thermal_model.core import heat_model
from thermal_model.core.heat_model import (
    BARE_TOP_COC_LAYERS,
    DEFAULT_MATERIALS,
    Layer,
    Material,
    build_layer_stack,
)

_UM = 1e-6


# ===============================================================
# 界面区间导热系数 k_face
# ===============================================================

def test_face_k_same_material_interval():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert ms.k_face.size == ms.Nx - 1
    # COC 内部区间 (界面 180 um 之前): k_face = COC k
    inside_coc = ms.x[1:] < 180e-6 - 1e-9
    assert np.all(ms.k_face[inside_coc] == pytest.approx(0.13))


def test_face_k_coc_water_180um():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    j = int(np.argmin(np.abs(ms.x - 180e-6)))
    assert ms.x[j] == pytest.approx(180e-6, abs=1e-12)
    # 左区间 [175,180] um 在 COC 内; 右区间 [180,185] um 在 Water 内
    assert ms.k_face[j - 1] == pytest.approx(0.13)   # COC k
    assert ms.k_face[j] == pytest.approx(0.60)       # Water k


def test_face_k_water_oil_200um():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    j = int(np.argmin(np.abs(ms.x - 200e-6)))
    assert ms.x[j] == pytest.approx(200e-6, abs=1e-12)
    assert ms.k_face[j - 1] == pytest.approx(0.60)   # Water k
    assert ms.k_face[j] == pytest.approx(0.142)      # Oil k


def test_face_k_oil_coc_250um():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    j = int(np.argmin(np.abs(ms.x - 250e-6)))
    assert ms.x[j] == pytest.approx(250e-6, abs=1e-12)
    assert ms.k_face[j - 1] == pytest.approx(0.142)  # Oil k
    assert ms.k_face[j] == pytest.approx(0.13)       # COC k


def test_no_harmonic_mean_at_node_on_interface_faces():
    """界面右区间不得错误使用端点调和平均。"""
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    for b in (180e-6, 200e-6, 250e-6):
        j = int(np.argmin(np.abs(ms.x - b)))
        k_harm = 2 * ms.k[j] * ms.k[j + 1] / (ms.k[j] + ms.k[j + 1])
        assert not np.isclose(ms.k_face[j], k_harm)


def test_face_k_arbitrary_stack():
    mats = {
        "A": Material("A", k_W_mK=0.5, rho_kg_m3=100.0, cp_J_kgK=1000.0),
        "B": Material("B", k_W_mK=3.0, rho_kg_m3=200.0, cp_J_kgK=2000.0),
        "C": Material("C", k_W_mK=0.05, rho_kg_m3=300.0, cp_J_kgK=3000.0),
    }
    layers = [
        Layer("L1", "A", 20e-6, cells=4),
        Layer("L2", "B", 30e-6, cells=3),
        Layer("L3", "C", 40e-6, cells=2),
    ]
    ms = build_layer_stack(mats, layers)
    # 区间 [x_i, x_{i+1}] 所属层 = 右端节点所属层
    for i in range(ms.Nx - 1):
        li = ms.node_layer_index[i + 1]
        assert ms.k_face[i] == pytest.approx(
            mats[layers[li].material].k_W_mK
        )


# ===============================================================
# 界面体积热容 rho_cp
# ===============================================================

def test_rho_cp_ordinary_node():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    j_coc = int(np.argmin(np.abs(ms.x - 90e-6)))    # COC 内部
    j_w = int(np.argmin(np.abs(ms.x - 190e-6)))     # Water 内部
    assert ms.rho_cp[j_coc] == pytest.approx(1020.0 * 1800.0)
    assert ms.rho_cp[j_w] == pytest.approx(1000.0 * 4180.0)


def test_rho_cp_equal_spacing_interface():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    rc_coc = 1020.0 * 1800.0
    rc_water = 1000.0 * 4180.0
    rc_oil = 876.0 * 1962.0
    assert ms.rho_cp[36] == pytest.approx((rc_coc + rc_water) / 2)
    assert ms.rho_cp[40] == pytest.approx((rc_water + rc_oil) / 2)
    assert ms.rho_cp[50] == pytest.approx((rc_oil + rc_coc) / 2)


def test_rho_cp_unequal_spacing_interface():
    mats = {
        "A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0),
        "B": Material("B", k_W_mK=2.0, rho_kg_m3=200.0, cp_J_kgK=2000.0),
    }
    # 左层 30 um cells=3 -> h=10 um; 右层 40 um cells=2 -> h=20 um
    layers = [Layer("L", "A", 30e-6, cells=3), Layer("R", "B", 40e-6, cells=2)]
    ms = build_layer_stack(mats, layers)
    j = int(np.argmin(np.abs(ms.x - 30e-6)))
    assert ms.x[j] == pytest.approx(30e-6, abs=1e-12)
    rcA = 100.0 * 1000.0
    rcB = 200.0 * 2000.0
    hL = ms.h[j - 1]
    hR = ms.h[j]
    assert hL == pytest.approx(10e-6)
    assert hR == pytest.approx(20e-6)
    assert ms.rho_cp[j] == pytest.approx(
        (rcA * hL + rcB * hR) / (hL + hR)
    )


def test_rho_cp_positive_finite():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert np.all(np.isfinite(ms.rho_cp))
    assert np.all(ms.rho_cp > 0)


def test_interface_node_not_full_left_material_capacity():
    """界面节点不得整段使用左侧材料热容 (除非两侧确为同一材料)。"""
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert ms.rho_cp[36] != pytest.approx(1020.0 * 1800.0)   # 纯 COC
    assert ms.rho_cp[36] != pytest.approx(1000.0 * 4180.0)   # 纯 Water


# ===============================================================
# 稳定性
# ===============================================================

def _coefficients(mesh, dt):
    k_face = mesh.k_face
    h = mesh.h
    rho_cp = mesh.rho_cp
    h_m = h[:-1]
    h_p = h[1:]
    k_m = k_face[:-1]
    k_p = k_face[1:]
    rc_int = rho_cp[1:-1]
    fac = 2 * dt / ((h_m + h_p) * rc_int)
    c_p = fac * k_p / h_p
    c_m = fac * k_m / h_m
    c_c = 1.0 - c_p - c_m
    return c_m, c_c, c_p


def test_corrected_coefficients_nonnegative_and_sum_one():
    for layers in (BARE_TOP_COC_LAYERS, heat_model.LEGACY_INSULATED_LAYERS):
        mesh, dt = heat_model.compute_stable_dt(DEFAULT_MATERIALS, layers)
        c_m, c_c, c_p = _coefficients(mesh, dt)
        assert np.all(c_m >= 0)
        assert np.all(c_p >= 0)
        assert np.all(c_c >= 0)
        np.testing.assert_allclose(c_c + c_m + c_p, 1.0, rtol=0, atol=1e-12)


def test_stable_dt_consistent_between_solver_and_helper():
    t = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    Tint = np.array([30.0, 95.0, 95.0, 40.0, 40.0, 60.0])
    mesh, dt = heat_model.compute_stable_dt(DEFAULT_MATERIALS,
                                            BARE_TOP_COC_LAYERS)
    res = heat_model.run_simulation(
        t, Tint, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, save_dt=0.1,
    )
    assert res["dt"] == dt
    assert res["mesh"] is not None
    # run_simulation 与 compute_stable_dt 使用同一套修正系数 (mesh.k_face/rho_cp)
    np.testing.assert_array_equal(res["mesh"].k_face, mesh.k_face)
    np.testing.assert_array_equal(res["mesh"].rho_cp, mesh.rho_cp)


# ===============================================================
# 解析稳态基准
# ===============================================================

def _steady_solve(mesh, materials, layers, Tb, h_conv, T_amb):
    """修正版稳态 FV 三对角直接求解 (等价于显式格式收敛到稳态)。"""
    n = mesh.Nx
    k_face = mesh.k_face
    h = mesh.h
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0, 0] = 1.0
    b[0] = Tb
    for j in range(1, n - 1):
        k_w = k_face[j - 1]
        k_e = k_face[j]
        h_w = h[j - 1]
        h_e = h[j]
        A[j, j - 1] = k_w / h_w
        A[j, j] = -(k_w / h_w + k_e / h_e)
        A[j, j + 1] = k_e / h_e
    k_top = k_face[-1]
    h_top = h[-1]
    A[n - 1, n - 2] = k_top / h_top
    A[n - 1, n - 1] = -(k_top / h_top + h_conv)
    b[n - 1] = -h_conv * T_amb
    return np.linalg.solve(A, b)


def _steady_solve_old(mesh, materials, layers, Tb, h_conv, T_amb):
    """旧界面处理 (端点调和平均) 的稳态三对角解 (仅用于对比)。"""
    n = mesh.Nx
    k_node = mesh.k
    k_half = 2 * k_node[:-1] * k_node[1:] / (k_node[:-1] + k_node[1:])
    h = mesh.h
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0, 0] = 1.0
    b[0] = Tb
    for j in range(1, n - 1):
        k_w = k_half[j - 1]
        k_e = k_half[j]
        h_w = h[j - 1]
        h_e = h[j]
        A[j, j - 1] = k_w / h_w
        A[j, j] = -(k_w / h_w + k_e / h_e)
        A[j, j + 1] = k_e / h_e
    k_top = k_node[-1]
    h_top = h[-1]
    A[n - 1, n - 2] = k_top / h_top
    A[n - 1, n - 1] = -(k_top / h_top + h_conv)
    b[n - 1] = -h_conv * T_amb
    return np.linalg.solve(A, b)


def test_analytical_steady_bare_top():
    """裸顶多层堆叠: 修正版稳态数值 == 解析串联热阻 (界面温度 + q)。"""
    Tb, h_conv, T_amb = 90.0, 5.0, 25.0
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    T_new = _steady_solve(ms, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
                          Tb, h_conv, T_amb)
    q, pos, T_ana = fv_reference.analytical_steady(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, Tb, h_conv, T_amb
    )
    # 界面 + 顶表面 (180/200/250/850 um)
    for x_target, T_target in zip(pos[1:], T_ana[1:]):
        j = int(np.argmin(np.abs(ms.x - x_target)))
        assert ms.x[j] == pytest.approx(x_target, abs=1e-12)
        assert T_new[j] == pytest.approx(T_target, abs=1e-6)
    # 底部第一段热流
    q_num = -ms.k_face[0] * (T_new[1] - T_new[0]) / ms.h[0]
    assert q_num == pytest.approx(q, rel=1e-8)


def test_analytical_steady_explicit_trend():
    """显式修正版求解器随时间逼近解析稳态 (60 s 后单调上升且未越界)。"""
    Tb, h_conv, T_amb = 90.0, 5.0, 25.0
    t_proto = np.arange(61.0, dtype=float)
    Tb_arr = np.where(t_proto < 1.0, 25.0, Tb)
    res = heat_model.run_simulation(
        t_proto, Tb_arr, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        h_conv=h_conv, T_air_ambient=T_amb, save_dt=1.0,
    )
    _, _, T_ana = fv_reference.analytical_steady(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, Tb, h_conv, T_amb
    )
    T_top = res["T_outer_surface_arr"]
    # 单调升温至解析稳态值的下方
    assert np.all(np.diff(T_top) >= -1e-9)
    assert T_top[-1] < T_ana[-1]
    assert T_top[-1] > 25.0
    # 60 s 时距稳态还差一部分但方向正确 (τ ≈ 330 s)
    assert T_top[-1] > 0.5 * T_ana[-1]


# ===============================================================
# 高对比堆叠基准
# ===============================================================

_HC_MATS = {
    "A": Material("A", k_W_mK=0.1, rho_kg_m3=1000.0, cp_J_kgK=1000.0),
    "B": Material("B", k_W_mK=1000.0, rho_kg_m3=1000.0, cp_J_kgK=1000.0),
    "C": Material("C", k_W_mK=0.01, rho_kg_m3=1000.0, cp_J_kgK=1000.0),
}
_HC_LAYERS = [
    Layer("A", "A", 100e-6, cells=4),
    Layer("B", "B", 50e-6, cells=2),
    Layer("C", "C", 200e-6, cells=2),
]


def test_high_contrast_stack_steady():
    """k 对比 10^5 倍: 修正版匹配解析解; 旧方案明显偏离 (该测试在旧方案下失败)。"""
    Tb, h_conv, T_amb = 100.0, 10.0, 25.0
    ms = build_layer_stack(_HC_MATS, _HC_LAYERS)
    T_new = _steady_solve(ms, _HC_MATS, _HC_LAYERS, Tb, h_conv, T_amb)
    T_old = _steady_solve_old(ms, _HC_MATS, _HC_LAYERS, Tb, h_conv, T_amb)
    q, pos, T_ana = fv_reference.analytical_steady(
        _HC_MATS, _HC_LAYERS, Tb, h_conv, T_amb
    )
    old_max_err = 0.0
    for x_target, T_target in zip(pos[1:], T_ana[1:]):
        j = int(np.argmin(np.abs(ms.x - x_target)))
        assert ms.x[j] == pytest.approx(x_target, abs=1e-12)
        # 修正版: 解析级精度
        assert T_new[j] == pytest.approx(T_target, abs=1e-6)
        old_max_err = max(old_max_err, abs(T_old[j] - T_target))
    # 旧方案: 界面/表面温度明显错误 (误差集中在低 k 材料侧的界面)
    assert old_max_err > 0.5


# ===============================================================
# 均匀材料不变性
# ===============================================================

def test_homogeneous_slab_unchanged():
    """单材料均匀板: 界面修正不改变物理 (温度场逐位一致);
    样品观测已改为空间加权平均, 与独立修正版参考一致。"""
    mats = {"COC": Material("COC", k_W_mK=0.13, rho_kg_m3=1020.0, cp_J_kgK=1800.0)}
    layers = [Layer("Slab", "COC", 500e-6, dx_target_m=5e-6, role="sample")]
    t_proto = np.arange(0.0, 20.0, 0.5)
    Tb = 25.0 + 40.0 * (t_proto >= 2.0)
    new = heat_model.run_simulation(
        t_proto, Tb, mats, layers, h_conv=5.0, T_air_ambient=25.0, save_dt=0.5,
    )
    # 独立修正版参考 (含样品空间加权平均) 逐位一致
    ref = fv_reference.corrected_run(
        mats, layers, t_proto, Tb, h_conv=5.0, T_air_ambient=25.0, save_dt=0.5,
    )
    assert new["dt"] == ref["dt"]
    np.testing.assert_allclose(new["T_sample_arr"], ref["T_sample_arr"],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(new["T_outer_surface_arr"],
                               ref["T_outer_surface_arr"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(new["T_final"], ref["T_final"], rtol=0, atol=1e-12)
    # 物理 (温度场) 与旧方案逐位一致: 均匀材料无界面差异
    old = fv_reference.old_run(
        mats, layers, t_proto, Tb, h_conv=5.0, T_air_ambient=25.0, save_dt=0.5,
    )
    assert new["dt"] == old["dt"]
    np.testing.assert_allclose(new["T_final"], old["T_final"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(new["T_outer_surface_arr"],
                               old["T_outer_surface_arr"], rtol=0, atol=1e-12)


# ===============================================================
# 模块 vs 独立修正版参考 (位级一致) + 旧方案差异 (物理修正生效)
# ===============================================================

_SYNTH_T = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
_SYNTH_TINT = np.array([30.0, 95.0, 95.0, 40.0, 40.0, 60.0])


def test_module_matches_corrected_reference():
    new = heat_model.run_simulation(
        _SYNTH_T, _SYNTH_TINT, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        save_dt=0.1,
    )
    ref = fv_reference.corrected_run(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, _SYNTH_T, _SYNTH_TINT,
        save_dt=0.1,
    )
    assert new["dt"] == ref["dt"]
    assert new["Nt"] == ref["Nt"]
    np.testing.assert_array_equal(new["t_array"], ref["t_array"])
    np.testing.assert_allclose(new["T_sample_arr"], ref["T_sample_arr"],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(new["T_outer_surface_arr"],
                               ref["T_outer_surface_arr"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(new["T_final"], ref["T_final"], rtol=0, atol=1e-10)


def test_corrected_differs_from_old_scheme():
    """界面修正必须改变结果 (界面节点物理不同); 不能是纯几何改名。"""
    new = heat_model.run_simulation(
        _SYNTH_T, _SYNTH_TINT, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS,
        save_dt=0.1,
    )
    old = fv_reference.old_run(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, _SYNTH_T, _SYNTH_TINT,
        save_dt=0.1,
    )
    assert np.max(np.abs(new["T_sample_arr"] - old["T_sample_arr"])) > 1e-4
    assert np.max(np.abs(new["T_outer_surface_arr"]
                         - old["T_outer_surface_arr"])) > 1e-6


# ===============================================================
# 瞬态对比 (25 -> 90 °C 阶跃, 60 s): 量化修正影响 (不解释为真值)
# ===============================================================

def test_transient_step_old_vs_corrected_impact():
    t_proto = np.arange(61.0, dtype=float)
    Tb_arr = np.where(t_proto < 1.0, 25.0, 90.0)
    new = heat_model.run_simulation(
        t_proto, Tb_arr, DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, save_dt=1.0,
    )
    old = fv_reference.old_run(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, t_proto, Tb_arr, save_dt=1.0,
    )
    # 时间轴一致 (dt 相同)
    np.testing.assert_array_equal(new["t_array"], old["t_array"])
    d_s = new["T_sample_arr"] - old["T_sample_arr"]
    d_t = new["T_top_surface_arr"] - old["T_top_surface_arr"]
    # 修正影响确实存在且有物理量级
    assert np.max(np.abs(d_s)) > 0.05
    assert np.max(np.abs(d_t)) > 1e-3
    # 两者均物理合理: 在 [25, 90] 范围内
    assert np.all(new["T_sample_arr"] >= 25.0 - 1e-9)
    assert np.all(new["T_sample_arr"] <= 90.0 + 1e-9)
    assert np.all(new["T_top_surface_arr"] >= 25.0 - 1e-9)
    assert np.all(new["T_top_surface_arr"] <= 90.0 + 1e-9)


# ===============================================================
# 裸顶几何保持 (不回归上一任务)
# ===============================================================

def test_bare_top_geometry_preserved():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert ms.boundaries[-1] == pytest.approx(850e-6, abs=1e-12)
    assert ms.x[-1] == pytest.approx(850e-6, abs=1e-12)
    assert ms.sample_layer_index == 1
    assert ms.top_surface_layer_index == 3
    assert ms.idx_top_surface[0] == ms.Nx - 1
    present = {BARE_TOP_COC_LAYERS[i].material
               for i in np.unique(ms.node_layer_index)}
    assert "Air" not in present
    assert "PDMS" not in present


def test_layer_stack_exposes_corrected_arrays():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    # 公开 helper 与 mesh 内数组一致 (单一事实来源)
    kf = heat_model.build_face_conductivity(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, ms.node_layer_index
    )
    rc = heat_model.build_volumetric_capacity(
        DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS, ms.node_layer_index, ms.h
    )
    np.testing.assert_array_equal(kf, ms.k_face)
    np.testing.assert_array_equal(rc, ms.rho_cp)
