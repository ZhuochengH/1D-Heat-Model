"""
裸顶实验层叠几何 / 顶部观测语义 / Pylance Optional 收窄测试。

覆盖 (任务 22 的 16 项 + 几何/观测专项):
1.  历史 legacy 绝缘预设仍为 6 层 (含 Air + PDMS, 总厚 4050 um);
2.  裸顶实验预设恰好 4 层;
3.  裸顶材料序列 = COC, Water, Oil, COC;
4.  裸顶总厚度 = 850e-6 m (容差内, 180+20+50+600);
5.  裸顶无 Air 层;
6.  裸顶无 PDMS 层;
7.  sample 角色解析到 180-200 um 层;
8.  top_surface 角色解析到 Top COC;
9.  Top COC 外边界 = 850 um (累积边界 0/180/200/250/850 um);
10. 最外节点 = 850 um, 无 Air/PDMS 节点;
11. 裸顶实验顶部观测 == 最外表面 (idx_top_surface == Nx-1);
12. Robin 对流边界施加于 Top COC 外表面 (x = 850 um);
13. legacy 回归数值等价 (DEFAULT_LAYERS 与 LEGACY_INSULATED_LAYERS 逐位一致);
14. 用户仍可构造任意含 Air/PDMS 的层叠;
15. Optional 控制流运行时路径: cells / dx_target / 二者皆无 -> 报错 /
    二者皆有 -> 报错;
16. 实验观测坐标 = 850 um 而非旧坐标 4050 um;
17. 层叠预设注册表 / resolve_layer_stack;
18. 多个 role='top_surface' 报错。

注意: 本文件不做任何 k_eff / cp_eff 拟合, 也不修改 Robin 方程。
"""

import numpy as np
import pytest

import heat_model
from heat_model import (
    BARE_TOP_COC_LAYERS,
    DEFAULT_LAYERS,
    DEFAULT_MATERIALS,
    LEGACY_INSULATED_LAYERS,
    Layer,
    Material,
    build_layer_stack,
)

_UM = 1e-6


# ===============================================================
# 1-2. 层数
# ===============================================================

def test_legacy_stack_has_six_layers():
    assert len(LEGACY_INSULATED_LAYERS) == 6
    assert [l.material for l in LEGACY_INSULATED_LAYERS] == [
        "COC", "Water", "Oil", "COC", "Air", "PDMS",
    ]
    total = sum(l.thickness_m for l in LEGACY_INSULATED_LAYERS)
    assert total == pytest.approx(4050e-6)
    # DEFAULT_LAYERS 是向后兼容别名 (同一对象), 未破坏历史引用
    assert DEFAULT_LAYERS is LEGACY_INSULATED_LAYERS


def test_bare_top_stack_exactly_four_layers():
    assert len(BARE_TOP_COC_LAYERS) == 4


# ===============================================================
# 3-6. 材料 / 厚度 / 无 Air / 无 PDMS
# ===============================================================

def test_bare_top_stack_materials():
    assert [l.material for l in BARE_TOP_COC_LAYERS] == [
        "COC", "Water", "Oil", "COC",
    ]


def test_bare_top_total_thickness_850um():
    thicknesses = [l.thickness_m for l in BARE_TOP_COC_LAYERS]
    assert thicknesses == [180e-6, 20e-6, 50e-6, 600e-6]
    assert sum(thicknesses) == pytest.approx(850e-6)
    # 数值确认: 180 + 20 + 50 + 600 = 850
    assert 180 + 20 + 50 + 600 == 850


def test_bare_top_has_no_air_layer():
    assert not any(l.material == "Air" for l in BARE_TOP_COC_LAYERS)


def test_bare_top_has_no_pdms_layer():
    assert not any(l.material == "PDMS" for l in BARE_TOP_COC_LAYERS)


# ===============================================================
# 7-8. sample / top_surface 角色解析
# ===============================================================

def test_sample_role_resolves_to_180_200um():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert ms.sample_layer_index == 1
    assert ms.boundaries[1] == pytest.approx(180e-6)
    assert ms.boundaries[2] == pytest.approx(200e-6)
    assert ms.idx_sample.size > 0
    # 样品观测节点位于 (180, 200] um (含界面归属规则)
    assert np.all(ms.x[ms.idx_sample] > 180e-6 - 1e-9)
    assert np.all(ms.x[ms.idx_sample] <= 200e-6 + 1e-9)
    # 观测节点全部属于样品层 (Water)
    assert np.all(ms.node_layer_index[ms.idx_sample] == 1)


def test_top_surface_role_resolves_to_top_coc():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert ms.top_surface_layer_index == 3
    assert BARE_TOP_COC_LAYERS[3].name == "Top COC"
    assert BARE_TOP_COC_LAYERS[3].material == "COC"
    assert ms.idx_top_surface.size == 1
    # 顶表面节点属于 Top COC 层
    assert ms.node_layer_index[ms.idx_top_surface[0]] == 3


# ===============================================================
# 9-10. 累积边界 / 最外节点 / 无 Air/PDMS 节点
# ===============================================================

def test_bare_top_cumulative_boundaries():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    np.testing.assert_allclose(
        ms.boundaries, [0.0, 180e-6, 200e-6, 250e-6, 850e-6],
        rtol=0, atol=1e-12,
    )


def test_outermost_node_is_850um_and_final_material_is_coc():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert ms.x[-1] == pytest.approx(850e-6, abs=1e-12)
    assert ms.boundaries[-1] == pytest.approx(850e-6, abs=1e-12)
    # 无 Air / PDMS 节点
    present = {BARE_TOP_COC_LAYERS[i].material for i in np.unique(
        ms.node_layer_index
    )}
    assert "Air" not in present
    assert "PDMS" not in present
    # 最外节点材料 = COC
    assert BARE_TOP_COC_LAYERS[ms.node_layer_index[-1]].material == "COC"
    # 裸顶网格 = 均匀 5 um (关注区分辨率), Nx = 850/5 + 1
    assert ms.Nx == 171
    np.testing.assert_allclose(ms.h, 5e-6, rtol=0, atol=1e-15)


# ===============================================================
# 11. 实验顶部观测 == 最外表面
# ===============================================================

def test_experimental_top_observation_equals_outer_surface():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert ms.idx_top_surface[0] == ms.Nx - 1  # 顶表面观测 == x[-1]

    t = np.arange(5.0, dtype=float)
    Tb = 30.0 + t
    res = heat_model.run_simulation(
        time_s=t, bottom_temperature_C=Tb,
        materials=DEFAULT_MATERIALS, layers=BARE_TOP_COC_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    # T_outer_surface_arr == T_top_arr (别名) == T_top_surface_arr (角色观测)
    np.testing.assert_array_equal(res["T_outer_surface_arr"], res["T_top_arr"])
    assert res["T_top_surface_arr"].size == res["T_outer_surface_arr"].size
    np.testing.assert_allclose(
        res["T_top_surface_arr"], res["T_outer_surface_arr"],
        rtol=0, atol=1e-12,
    )


# ===============================================================
# 12. Robin 边界施加于 Top COC 外表面
# ===============================================================

def test_robin_applied_at_top_coc_outer_surface():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    # 顶表面观测节点 == 最外节点 == Robin 更新目标节点
    assert ms.idx_top_surface[0] == ms.Nx - 1
    assert ms.node_layer_index[-1] == 3          # Top COC
    assert ms.top_surface_layer_index == 3
    assert ms.x[-1] == pytest.approx(850e-6, abs=1e-12)
    # Robin 面之后没有任何材料节点 (裸顶, 无 Air/PDMS 盖帽)
    assert ms.idx_top_surface[0] == ms.Nx - 1

    # 物理合理性 (不解释精度): 底部恒温 90 °C 长时间对流,
    # 顶表面应低于底部且高于环境; 外表面应低于样品层。
    t = np.linspace(0.0, 200.0, 2001)
    Tb = np.full_like(t, 90.0)
    res = heat_model.run_simulation(
        time_s=t, bottom_temperature_C=Tb,
        materials=DEFAULT_MATERIALS, layers=BARE_TOP_COC_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=1.0,
    )
    T_top = res["T_outer_surface_arr"]
    T_sample = res["T_sample_arr"]
    # 初始条件 = 25.0 °C, 故用 >= / <= 并留数值容差
    assert np.all(T_top >= 25.0 - 1e-9)
    assert np.all(T_top <= 90.0 + 1e-9)
    assert T_top[-1] <= T_sample[-1] + 1e-9


# ===============================================================
# 13. legacy 回归数值等价
# ===============================================================

def test_legacy_regression_numerically_equivalent():
    # 别名对象一致
    assert DEFAULT_LAYERS is LEGACY_INSULATED_LAYERS
    ms_new = build_layer_stack(DEFAULT_MATERIALS, LEGACY_INSULATED_LAYERS)
    ms_old = build_layer_stack(DEFAULT_MATERIALS, DEFAULT_LAYERS)
    for field in ("x", "h", "k", "rho", "cp", "idx_sample", "boundaries",
                  "node_layer_index"):
        np.testing.assert_array_equal(getattr(ms_new, field),
                                      getattr(ms_old, field))
    assert ms_new.Nx == 190  # 与原脚本已知节点数一致 (历史 6 层网格不变)

    # 求解结果: 新输出键存在, 别名一致; 旧配置无 top_surface 角色
    t = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    Tint = np.array([30.0, 95.0, 95.0, 40.0, 40.0, 60.0])
    res = heat_model.run_simulation(
        time_s=t, bottom_temperature_C=Tint,
        materials=DEFAULT_MATERIALS, layers=LEGACY_INSULATED_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    np.testing.assert_array_equal(res["T_top_arr"], res["T_outer_surface_arr"])
    assert res["T_top_surface_arr"].size == 0
    assert res["mesh"].idx_top_surface.size == 0
    assert res["mesh"].top_surface_layer_index is None


def test_legacy_has_no_top_surface_role_in_description():
    desc = heat_model.describe_layer_stack(
        DEFAULT_MATERIALS, LEGACY_INSULATED_LAYERS
    )
    assert desc["has_top_surface_role"] is False
    assert desc["top_surface_position_m"] is None
    assert desc["outer_surface_position_m"] == pytest.approx(4050e-6)
    assert desc["has_air_layer"] is True
    assert desc["has_pdms_layer"] is True


# ===============================================================
# 14. 任意含 Air/PDMS 的层叠仍可构造
# ===============================================================

def test_arbitrary_air_pdms_stack_still_constructible():
    # 材料库仍提供 Air / PDMS
    assert "Air" in DEFAULT_MATERIALS
    assert "PDMS" in DEFAULT_MATERIALS

    mats = {
        "A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0),
        "Air": Material("Air", k_W_mK=0.0257, rho_kg_m3=1.204, cp_J_kgK=1005.0),
        "PDMS": Material("PDMS", k_W_mK=0.15, rho_kg_m3=970.0, cp_J_kgK=1460.0),
    }
    layers = [
        Layer(name="L1", material="A", thickness_m=10e-6, cells=4, role="sample"),
        Layer(name="L2", material="Air", thickness_m=100e-6, dx_target_m=25e-6),
        Layer(name="L3", material="PDMS", thickness_m=50e-6, cells=5),
    ]
    ms = build_layer_stack(mats, layers)
    assert ms.boundaries[-1] == pytest.approx(160e-6)
    present = {layers[i].material for i in np.unique(ms.node_layer_index)}
    assert "Air" in present
    assert "PDMS" in present

    # 求解器对该自定义堆叠仍可运行
    res = heat_model.run_simulation(
        time_s=np.arange(5.0, dtype=float),
        bottom_temperature_C=30.0 + np.arange(5.0, dtype=float),
        materials=mats, layers=layers,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.5,
    )
    assert res["mesh"].Nx == ms.Nx
    # 自定义堆叠无 top_surface 角色 -> 观测数组为空
    assert res["T_top_surface_arr"].size == 0


# ===============================================================
# 15. Optional 控制流运行时路径 (Pylance 收窄对应行为)
# ===============================================================

_MATS1 = {"A": Material("A", k_W_mK=1.0, rho_kg_m3=100.0, cp_J_kgK=1000.0)}


def test_optional_control_flow_cells_specified():
    layers = [Layer(name="L", material="A", thickness_m=10e-6, cells=5)]
    ms = build_layer_stack(_MATS1, layers)
    assert ms.Nx == 6


def test_optional_control_flow_dx_target_specified():
    layers = [Layer(name="L", material="A", thickness_m=10e-6, dx_target_m=5e-6)]
    ms = build_layer_stack(_MATS1, layers)
    assert ms.Nx == 3  # round(10/5) 单元 -> 3 节点


def test_optional_control_flow_neither_specified_raises():
    layers = [Layer(name="L", material="A", thickness_m=10e-6)]
    with pytest.raises(ValueError, match="cells 或 dx_target_m"):
        heat_model.validate_layers(layers, _MATS1)
    with pytest.raises(ValueError, match="cells 或 dx_target_m"):
        build_layer_stack(_MATS1, layers)


def test_optional_control_flow_both_specified_raises():
    layers = [Layer(name="L", material="A", thickness_m=10e-6,
                    cells=5, dx_target_m=5e-6)]
    with pytest.raises(ValueError, match="不能同时"):
        heat_model.validate_layers(layers, _MATS1)
    with pytest.raises(ValueError, match="不能同时"):
        build_layer_stack(_MATS1, layers)


# ===============================================================
# 16. 实验观测坐标 = 850 um 而非 4050 um
# ===============================================================

def test_experimental_top_observation_at_850um_not_4050um():
    ms = build_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert ms.x[-1] == pytest.approx(850e-6, abs=1e-12)
    assert abs(ms.x[-1] - 4050e-6) > 1e-3  # 绝不是旧 PDMS 顶面坐标

    desc = heat_model.describe_layer_stack(DEFAULT_MATERIALS, BARE_TOP_COC_LAYERS)
    assert desc["top_surface_position_m"] == pytest.approx(850e-6, abs=1e-12)
    assert desc["outer_surface_position_m"] == pytest.approx(850e-6, abs=1e-12)
    assert desc["top_surface_material"] == "COC"
    assert desc["outer_surface_material"] == "COC"
    assert desc["sample_position_m"] == (180e-6, 200e-6)
    assert desc["has_air_layer"] is False
    assert desc["has_pdms_layer"] is False


def test_experimental_predicted_observation_is_outer_surface():
    """未来拟合比较: 预测观测 = T[-1] = T_outer_surface = T_top_COC_surface。"""
    t = np.linspace(0.0, 20.0, 21)
    Tb = 40.0 + 30.0 * np.sin(0.3 * t)
    res = heat_model.run_simulation(
        time_s=t, bottom_temperature_C=Tb,
        materials=DEFAULT_MATERIALS, layers=BARE_TOP_COC_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.5,
    )
    np.testing.assert_allclose(
        res["T_top_surface_arr"], res["T_outer_surface_arr"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        res["T_top_arr"], res["T_outer_surface_arr"], rtol=0, atol=0
    )
    ms = res["mesh"]
    assert ms.x[-1] == pytest.approx(850e-6, abs=1e-12)
    assert ms.x[ms.idx_top_surface[0]] == pytest.approx(850e-6, abs=1e-12)
    # 观测点到环境之间没有 Air/PDMS 层 (观测节点即最外节点)
    assert ms.idx_top_surface[0] == ms.Nx - 1


# ===============================================================
# 17-18. 预设注册表 / 多 top_surface 报错
# ===============================================================

def test_layer_stack_preset_registry():
    assert set(heat_model.LAYER_STACK_PRESETS) == {"bare-top", "legacy"}
    assert heat_model.resolve_layer_stack("bare-top") is BARE_TOP_COC_LAYERS
    assert heat_model.resolve_layer_stack("legacy") is LEGACY_INSULATED_LAYERS
    with pytest.raises(ValueError, match="未知层叠预设"):
        heat_model.resolve_layer_stack("unknown")


def test_multiple_top_surface_roles_raise():
    layers = [
        Layer(name="L1", material="A", thickness_m=10e-6, cells=4,
              role="top_surface"),
        Layer(name="L2", material="A", thickness_m=10e-6, cells=4,
              role="top_surface"),
    ]
    with pytest.raises(ValueError, match="多个 role='top_surface'"):
        build_layer_stack(_MATS1, layers)


def test_run_simulation_exposes_new_observation_keys():
    t = np.arange(5.0, dtype=float)
    Tb = 30.0 + t
    res = heat_model.run_simulation(
        time_s=t, bottom_temperature_C=Tb,
        materials=DEFAULT_MATERIALS, layers=BARE_TOP_COC_LAYERS,
        h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
    )
    for key in ("T_outer_surface_arr", "T_top_arr", "T_top_surface_arr"):
        assert key in res, key
    assert len(res["T_outer_surface_arr"]) == len(res["t_array"])
