"""
材料库 / 层叠结构配置层测试 + 数值回归。

覆盖 (任务 19 的 18 项):
1.  Material 正确存储 k/rho/cp;
2.  负 k 被拒绝;
3.  零/负 rho 被拒绝;
4.  零/负 cp 被拒绝;
5.  Layer 引用有效材料;
6.  未知材料引用报错;
7.  零/负厚度被拒绝;
8.  任意层数支持;
9.  同一材料可出现在多个层;
10. 修改一个材料库条目 -> 所有引用该材料的层自动更新;
11. 自动 k(x)/rho(x)/cp(x) 数组与层分配一致;
12. 界面位置正确;
13. 非均匀网格行为保留;
14. sample 角色被正确找到;
15. 缺失 sample 角色被清晰处理;
16. 多个 sample 角色报错;
17. 默认泛化配置与原层叠结构逐位一致;
18. 数值回归: 修正版 FV 求解器与独立修正版参考逐位一致
    (旧方案为历史基线, 界面物理修正后结果有意不同, 见
    tests/test_interface_discretization.py)。

辅助 (历史基线):
- _reference_old_mesh: 逐字复刻重构前脚本的网格/材料构造;
- _reference_old_run : 逐字复刻重构前脚本的 FDM 求解器主循环 (旧界面方案,
                       仅作历史基线, 不再要求与修正版逐位等价)。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import fv_reference
import heat_model
from heat_model import (
    DEFAULT_LAYERS,
    DEFAULT_MATERIALS,
    Layer,
    Material,
    build_layer_stack,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "sample and heater T in one plot.py"


def load_main_module():
    """加载主脚本模块 (文件名含空格, 用 importlib 按路径加载)。"""
    spec = importlib.util.spec_from_file_location("sample_heater_fdm_cfg", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_main = None


@pytest.fixture(scope="module")
def m():
    global _main
    if _main is None:
        _main = load_main_module()
    return _main


# ===============================================================
# 1. Material 存储 k/rho/cp
# ===============================================================

def test_material_stores_properties():
    mat = Material(name="Glass", k_W_mK=1.0, rho_kg_m3=2500.0, cp_J_kgK=840.0)
    assert mat.name == "Glass"
    assert mat.k_W_mK == 1.0
    assert mat.rho_kg_m3 == 2500.0
    assert mat.cp_J_kgK == 840.0


# ===============================================================
# 2-4. 负 k / 零负 rho / 零负 cp 拒绝
# ===============================================================

@pytest.mark.parametrize(
    "field,bad",
    [
        ("k_W_mK", -0.1),
        ("k_W_mK", 0.0),
        ("rho_kg_m3", 0.0),
        ("rho_kg_m3", -5.0),
        ("cp_J_kgK", 0.0),
        ("cp_J_kgK", -3.0),
        ("rho_kg_m3", np.nan),
        ("cp_J_kgK", np.inf),
    ],
)
def test_nonpositive_or_nonfinite_property_rejected(field, bad):
    kwargs = dict(k_W_mK=1.0, rho_kg_m3=1000.0, cp_J_kgK=1000.0)
    kwargs[field] = bad
    with pytest.raises(ValueError):
        heat_model.validate_material(Material(name="X", **kwargs))


def test_non_numeric_property_rejected():
    with pytest.raises(TypeError):
        heat_model.validate_material(
            Material(name="X", k_W_mK="abc", rho_kg_m3=1000.0, cp_J_kgK=1000.0)
        )


def test_missing_property_raises():
    # dataclass 缺省必填字段 -> TypeError (清晰报错)
    with pytest.raises(TypeError):
        Material(name="X", k_W_mK=1.0, rho_kg_m3=1000.0)


# ===============================================================
# 5-7. Layer 引用 / 未知材料 / 厚度校验
# ===============================================================

def test_layer_references_valid_material():
    layer = Layer(name="Bottom", material="COC", thickness_m=180e-6, dx_target_m=5e-6)
    # 校验不抛异常即通过
    heat_model.validate_layers([layer], DEFAULT_MATERIALS)


def test_unknown_material_reference_raises():
    layer = Layer(name="Bottom", material="Kryptonite",
                  thickness_m=180e-6, dx_target_m=5e-6)
    with pytest.raises(ValueError, match="未知材料"):
        heat_model.validate_layers([layer], DEFAULT_MATERIALS)


@pytest.mark.parametrize("thickness", [0.0, -10e-6])
def test_zero_or_negative_thickness_rejected(thickness):
    layer = Layer(name="L", material="COC", thickness_m=thickness, dx_target_m=5e-6)
    with pytest.raises(ValueError):
        heat_model.validate_layers([layer], DEFAULT_MATERIALS)


def test_layer_requires_mesh_resolution():
    layer = Layer(name="L", material="COC", thickness_m=100e-6)
    with pytest.raises(ValueError, match="cells 或 dx_target_m"):
        heat_model.validate_layers([layer], DEFAULT_MATERIALS)


def test_layer_cells_and_dx_both_given_raises():
    layer = Layer(name="L", material="COC", thickness_m=100e-6,
                  cells=10, dx_target_m=5e-6)
    with pytest.raises(ValueError, match="不能同时"):
        heat_model.validate_layers([layer], DEFAULT_MATERIALS)


# ===============================================================
# 8. 任意层数支持
# ===============================================================

def test_arbitrary_layer_count_supported():
    mats = {
        "A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0),
        "B": Material("B", k_W_mK=2.0, rho_kg_m3=200.0, cp_J_kgK=2000.0),
    }
    for n in (1, 2, 5, 10):
        layers = [
            Layer(name=f"L{i}", material=("A" if i % 2 == 0 else "B"),
                  thickness_m=10e-6, cells=5)
            for i in range(n)
        ]
        ms = build_layer_stack(mats, layers)
        assert ms.boundaries.shape == (n + 1,)
        assert ms.Nx == n * 5 + 1  # 每层 5 个单元 -> 6 节点, 界面共享


# ===============================================================
# 9-10. 同一材料多层层用 / 中心修改自动传播
# ===============================================================

def test_same_material_multiple_layers():
    mats = {
        "A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0),
        "B": Material("B", k_W_mK=2.0, rho_kg_m3=200.0, cp_J_kgK=2000.0),
    }
    layers = [
        Layer(name="A1", material="A", thickness_m=10e-6, cells=10),
        Layer(name="B1", material="B", thickness_m=10e-6, cells=10),
        Layer(name="A2", material="A", thickness_m=10e-6, cells=10),
    ]
    ms = build_layer_stack(mats, layers)
    # A1 与 A2 的节点 k 应相同 (都来自材料 A)
    node_a1 = ms.node_layer_index == 0
    node_a2 = ms.node_layer_index == 2
    assert np.all(ms.k[node_a1] == 1.0)
    assert np.all(ms.k[node_a2] == 1.0)


def test_material_change_propagates_to_all_layers():
    mats = {
        "A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0),
        "B": Material("B", k_W_mK=2.0, rho_kg_m3=200.0, cp_J_kgK=2000.0),
    }
    layers = [
        Layer(name="A1", material="A", thickness_m=10e-6, cells=10),
        Layer(name="B1", material="B", thickness_m=10e-6, cells=10),
        Layer(name="A2", material="A", thickness_m=10e-6, cells=10),
    ]
    ms_before = build_layer_stack(mats, layers)
    # 中心修改材料 A 的 k
    mats["A"].k_W_mK = 3.0
    ms_after = build_layer_stack(mats, layers)
    # 所有引用 A 的节点 (A1 + A2) 都变为 3.0, B 层不变
    node_a = np.isin(ms_after.node_layer_index, [0, 2])
    node_b = ms_after.node_layer_index == 1
    assert np.all(ms_after.k[node_a] == 3.0)
    assert np.all(ms_after.k[node_b] == 2.0)
    assert not np.allclose(ms_before.k[node_a], ms_after.k[node_a])


# ===============================================================
# 11. k/rho/cp 数组与层分配一致
# ===============================================================

def test_property_arrays_match_layer_assignments():
    mats = {
        "A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0),
        "B": Material("B", k_W_mK=2.0, rho_kg_m3=200.0, cp_J_kgK=2000.0),
    }
    layers = [
        Layer(name="A1", material="A", thickness_m=10e-6, cells=10),
        Layer(name="B1", material="B", thickness_m=10e-6, cells=10),
    ]
    ms = build_layer_stack(mats, layers)
    for i, xi in enumerate(ms.x):
        if xi <= 10e-6 + 1e-9:
            assert ms.k[i] == 1.0
            assert ms.rho[i] == 100.0
            assert ms.cp[i] == 1000.0
            assert ms.node_layer_index[i] == 0
        else:
            assert ms.k[i] == 2.0
            assert ms.rho[i] == 200.0
            assert ms.cp[i] == 2000.0
            assert ms.node_layer_index[i] == 1


# ===============================================================
# 12. 界面位置正确
# ===============================================================

def test_interface_positions_correct():
    mats = {"A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0)}
    layers = [
        Layer(name="L1", material="A", thickness_m=10e-6, cells=10),
        Layer(name="L2", material="A", thickness_m=20e-6, cells=10),
        Layer(name="L3", material="A", thickness_m=30e-6, cells=10),
    ]
    ms = build_layer_stack(mats, layers)
    np.testing.assert_allclose(ms.boundaries, [0.0, 10e-6, 30e-6, 60e-6])


# ===============================================================
# 13. 非均匀网格行为保留
# ===============================================================

def test_nonuniform_mesh_preserved():
    ms = build_layer_stack(DEFAULT_MATERIALS, DEFAULT_LAYERS)
    # 关注区 5um, Air 200um, PDMS 50um -> 多种间距
    unique_h = np.unique(np.round(ms.h, 12))
    assert 5e-6 in unique_h
    assert 200e-6 in unique_h
    assert 50e-6 in unique_h
    assert len(unique_h) > 1


# ===============================================================
# 14-16. sample 角色识别
# ===============================================================

def test_sample_role_found():
    ms = build_layer_stack(DEFAULT_MATERIALS, DEFAULT_LAYERS)
    assert ms.sample_layer_index == 1  # PCR Sample 层
    assert ms.idx_sample.size > 0


def test_missing_sample_role_handled():
    mats = {"A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0)}
    layers = [
        Layer(name="L1", material="A", thickness_m=10e-6, cells=10),
        Layer(name="L2", material="A", thickness_m=10e-6, cells=10),
    ]
    ms = build_layer_stack(mats, layers)
    assert ms.sample_layer_index is None
    assert ms.idx_sample.size == 0


def test_multiple_sample_roles_raise():
    mats = {"A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0)}
    layers = [
        Layer(name="L1", material="A", thickness_m=10e-6, cells=10, role="sample"),
        Layer(name="L2", material="A", thickness_m=10e-6, cells=10, role="sample"),
    ]
    with pytest.raises(ValueError, match="多个 role='sample'"):
        build_layer_stack(mats, layers)


# ===============================================================
# 回归基准: 逐字复刻重构前的网格/材料构造与 FDM 主循环
# ===============================================================

def _reference_old_mesh():
    """重构前脚本第 1-2 节的逐字复刻 (回归基准)。"""
    L_coc_bot = 180e-6
    L_sample   = 20e-6
    L_oil      = 50e-6
    L_coc_top  = 600e-6
    L_air      = 3000e-6
    L_pdms     = 200e-6

    L_total = L_coc_bot + L_sample + L_oil + L_coc_top + L_air + L_pdms

    x_coc_bot_end = L_coc_bot
    x_sample_end  = L_coc_bot + L_sample
    x_oil_end     = L_coc_bot + L_sample + L_oil
    x_coc_top_end = L_coc_bot + L_sample + L_oil + L_coc_top
    x_air_end     = x_coc_top_end + L_air

    dx_fine = 5e-6
    dx_air  = 200e-6
    dx_pdms = 50e-6

    def make_layer(x0, x1, dx):
        n = max(1, int(round((x1 - x0) / dx)))
        return np.linspace(x0, x1, n + 1)

    x = np.unique(np.concatenate([
        make_layer(0,             x_coc_bot_end, dx_fine),
        make_layer(x_coc_bot_end, x_sample_end,  dx_fine),
        make_layer(x_sample_end,  x_oil_end,     dx_fine),
        make_layer(x_oil_end,     x_coc_top_end, dx_fine),
        make_layer(x_coc_top_end, x_air_end,     dx_air),
        make_layer(x_air_end,     L_total,       dx_pdms),
    ]))
    Nx = len(x)
    h = np.diff(x)

    rho_coc,  k_coc,  cp_coc  = 1020.0, 0.13,   1800.0
    rho_w,    k_w,    cp_w    = 1000.0, 0.60,   4180.0
    rho_oil,  k_oil,  cp_oil  = 876.0,  0.142,  1962.0
    rho_air,  k_air,  cp_air  = 1.204,  0.0257, 1005.0
    rho_pdms, k_pdms, cp_pdms = 970.0,  0.15,   1460.0

    rho = np.zeros(Nx)
    k   = np.zeros(Nx)
    cp  = np.zeros(Nx)

    for i, xi in enumerate(x):
        if xi <= x_coc_bot_end + 1e-9:
            rho[i], k[i], cp[i] = rho_coc,  k_coc,  cp_coc
        elif xi <= x_sample_end + 1e-9:
            rho[i], k[i], cp[i] = rho_w,    k_w,    cp_w
        elif xi <= x_oil_end + 1e-9:
            rho[i], k[i], cp[i] = rho_oil,  k_oil,  cp_oil
        elif xi <= x_coc_top_end + 1e-9:
            rho[i], k[i], cp[i] = rho_coc,  k_coc,  cp_coc
        elif xi <= x_air_end + 1e-9:
            rho[i], k[i], cp[i] = rho_air,  k_air,  cp_air
        else:
            rho[i], k[i], cp[i] = rho_pdms, k_pdms, cp_pdms

    idx_sample = np.where((x > L_coc_bot) & (x <= L_coc_bot + L_sample + 1e-9))[0]
    return x, h, k, rho, cp, idx_sample


def _reference_old_run(t_protocol, T_internal, a, b, tau_eff, prepare_fdm_boundary,
                       h_conv=5.0, T_air_ambient=25.0, save_dt=0.1):
    """重构前脚本第 4-5 节的逐字复刻 (回归基准)。

    prepare_fdm_boundary 未改动, 从主模块传入, 保证两边边界准备一致。
    """
    x, h, k, rho, cp, idx_sample = _reference_old_mesh()
    Nx = len(x)

    k_half = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])
    h_m = h[:-1]
    h_p = h[1:]
    k_m = k_half[:-1]
    k_p = k_half[1:]
    rho_int = rho[1:-1]
    cp_int  = cp[1:-1]

    dt_stable = rho_int * cp_int * (h_m + h_p) / (2 * (k_p / h_p + k_m / h_m))
    dt = np.min(dt_stable) * 0.9

    t_total = float(t_protocol[-1]) - float(t_protocol[0])
    Nt = int(t_total / dt) + 1
    time_fdm = np.linspace(0.0, t_total, Nt)

    _, _, T_surface_fdm = prepare_fdm_boundary(
        T_internal, t_protocol, a, b, tau_eff, time_fdm
    )

    fac = 2 * dt / ((h_m + h_p) * rho_int * cp_int)
    c_p = fac * k_p / h_p
    c_m = fac * k_m / h_m
    c_c = 1.0 - c_p - c_m

    bc_A = (k[-1] / h[-1]) / (k[-1] / h[-1] + h_conv)
    bc_B = h_conv * T_air_ambient / (k[-1] / h[-1] + h_conv)

    T = np.ones(Nx) * 25.0
    save_interval = max(1, int(save_dt / dt))
    plot_times = []
    plot_T_surface = []
    plot_T_sample = []
    plot_T_top = []

    for n in range(Nt):
        if n % save_interval == 0:
            plot_times.append(time_fdm[n])
            plot_T_surface.append(T_surface_fdm[n])
            plot_T_sample.append(np.mean(T[idx_sample]))
            plot_T_top.append(T[-1])

        T[0]    = T_surface_fdm[n]
        T[1:-1] = c_c * T[1:-1] + c_m * T[:-2] + c_p * T[2:]
        T[-1]   = bc_A * T[-2] + bc_B

    return {
        "t_array": np.array(plot_times),
        "T_surface_arr": np.array(plot_T_surface),
        "T_sample_arr": np.array(plot_T_sample),
        "T_top_arr": np.array(plot_T_top),
        "dt": dt,
        "Nt": Nt,
    }


# ===============================================================
# 17. 默认泛化配置与原层叠结构逐位一致
# ===============================================================

def test_default_materials_reproduce_original():
    assert DEFAULT_MATERIALS["COC"].k_W_mK == 0.13
    assert DEFAULT_MATERIALS["COC"].rho_kg_m3 == 1020.0
    assert DEFAULT_MATERIALS["COC"].cp_J_kgK == 1800.0
    assert DEFAULT_MATERIALS["Water"].k_W_mK == 0.60
    assert DEFAULT_MATERIALS["Water"].rho_kg_m3 == 1000.0
    assert DEFAULT_MATERIALS["Water"].cp_J_kgK == 4180.0
    assert DEFAULT_MATERIALS["Oil"].k_W_mK == 0.142
    assert DEFAULT_MATERIALS["Oil"].rho_kg_m3 == 876.0
    assert DEFAULT_MATERIALS["Oil"].cp_J_kgK == 1962.0
    assert DEFAULT_MATERIALS["Air"].k_W_mK == 0.0257
    assert DEFAULT_MATERIALS["Air"].rho_kg_m3 == 1.204
    assert DEFAULT_MATERIALS["Air"].cp_J_kgK == 1005.0
    assert DEFAULT_MATERIALS["PDMS"].k_W_mK == 0.15
    assert DEFAULT_MATERIALS["PDMS"].rho_kg_m3 == 970.0
    assert DEFAULT_MATERIALS["PDMS"].cp_J_kgK == 1460.0


def test_default_layer_stack_matches_original_structure():
    assert [l.name for l in DEFAULT_LAYERS] == [
        "Bottom COC", "PCR Sample", "Mineral Oil", "Top COC", "Air Gap", "Cap PDMS",
    ]
    assert [l.material for l in DEFAULT_LAYERS] == [
        "COC", "Water", "Oil", "COC", "Air", "PDMS",
    ]
    assert [l.thickness_m for l in DEFAULT_LAYERS] == [
        180e-6, 20e-6, 50e-6, 600e-6, 3000e-6, 200e-6,
    ]
    assert [l.dx_target_m for l in DEFAULT_LAYERS] == [
        5e-6, 5e-6, 5e-6, 5e-6, 200e-6, 50e-6,
    ]
    assert DEFAULT_LAYERS[1].role == "sample"


def test_build_layer_stack_reproduces_original_mesh_bit_exact():
    ref = _reference_old_mesh()
    ms = build_layer_stack(DEFAULT_MATERIALS, DEFAULT_LAYERS)
    np.testing.assert_array_equal(ms.x, ref[0])
    np.testing.assert_array_equal(ms.h, ref[1])
    np.testing.assert_array_equal(ms.k, ref[2])
    np.testing.assert_array_equal(ms.rho, ref[3])
    np.testing.assert_array_equal(ms.cp, ref[4])
    np.testing.assert_array_equal(ms.idx_sample, ref[5])
    # 已知节点数 (原均匀 5um 网格 811 节点 -> 非均匀 190 节点)
    assert ms.Nx == 190


# ===============================================================
# 18. 数值回归: 修正版 FV vs 独立参考 (界面物理已修正, 旧方案不再逐位等价)
# ===============================================================

_SYNTH_T = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
_SYNTH_TINT = np.array([30.0, 95.0, 95.0, 40.0, 40.0, 60.0])


def test_numerical_regression_corrected_fv_vs_reference(m):
    """修正版求解器与独立 FV 参考 (区间导热 + 体积加权热容) 逐位一致;
    旧方案 (调和平均 + 左侧材料热容) 为历史基线, 结果有意不同。"""
    a, b, tau = 0.95, 1.8, 7.3

    # 新的调用方式: 先构造 FDM 时间网格, 预处理底部边界, 再交给共享求解器
    mesh, dt = heat_model.compute_stable_dt(DEFAULT_MATERIALS, DEFAULT_LAYERS)
    t_total = float(_SYNTH_T[-1]) - float(_SYNTH_T[0])
    Nt = int(t_total / dt) + 1
    time_fdm = np.linspace(0.0, t_total, Nt)
    _, _, T_surface_fdm = m.prepare_fdm_boundary(
        _SYNTH_TINT, _SYNTH_T, a, b, tau, time_fdm
    )
    new = heat_model.run_simulation(
        time_s=time_fdm,
        bottom_temperature_C=T_surface_fdm,
        materials=DEFAULT_MATERIALS,
        layers=DEFAULT_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    # 独立修正版参考 (fv_reference.corrected_run): 同样的边界预处理数组
    ref = fv_reference.corrected_run(
        DEFAULT_MATERIALS, DEFAULT_LAYERS, time_fdm, T_surface_fdm,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )

    # dt 与时间点数一致
    assert new["dt"] == pytest.approx(ref["dt"], rel=0, abs=0)
    assert new["Nt"] == ref["Nt"]
    np.testing.assert_array_equal(new["t_array"], ref["t_array"])

    # 温度场 (底部边界/样品/顶部) 严格一致 (修正版位级等价)
    tol = 1e-8
    np.testing.assert_allclose(new["T_bottom_arr"], ref["T_bottom_arr"],
                               rtol=0, atol=tol)
    np.testing.assert_allclose(new["T_sample_arr"], ref["T_sample_arr"],
                               rtol=0, atol=tol)
    np.testing.assert_allclose(new["T_top_arr"], ref["T_outer_surface_arr"],
                               rtol=0, atol=tol)

    # 记录最大绝对差 (供报告; 预期为 0.0)
    max_d_sample = np.max(np.abs(new["T_sample_arr"] - ref["T_sample_arr"]))
    max_d_top = np.max(np.abs(new["T_top_arr"] - ref["T_outer_surface_arr"]))
    assert max_d_sample <= tol
    assert max_d_top <= tol

    # 历史基线: 旧界面方案 (调和平均) 仍保留在 _reference_old_run / fv_reference
    # 界面物理修正后其结果与修正版有意不同 (不再 bit-exact)。
    old = fv_reference.old_run(
        DEFAULT_MATERIALS, DEFAULT_LAYERS, time_fdm, T_surface_fdm,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    assert np.max(np.abs(new["T_sample_arr"] - old["T_sample_arr"])) > 1e-4
