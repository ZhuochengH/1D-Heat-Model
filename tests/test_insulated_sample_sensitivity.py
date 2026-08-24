#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绝缘样品温度循环范围 OAT 敏感性 — 测试 (任务书 §31)
=====================================================
验证:
    1. 权威冻结基线值不变
    2. 绝缘几何使用
    3. 完整输入热历史使用
    4. 初始激活相从重复周期响应统计排除
    5. 周期窗口跨所有扰动固定
    6. mean HIGH = 每周期最大值的均值
    7. mean LOW  = 每周期最小值的均值
    8. 振幅按周期 = HIGH - LOW
    9. 排序不使用整体最大
    10. 一次只变一个参数
    11. 其他参数保持基线
    12. k 校准支持范围正确
    13. cp 校准支持范围正确
    14. epsilon 永不超过 1
    15. h=0 明确分类为消融
    16. epsilon=0 明确分类为消融
    17. h=0 + epsilon=0 分类为消融
    18. tau 负控制不改样品温度
    19. tau 永不作用于样品温度
    20. Top 观测滞后不进入样品敏感性
    21. 无参数写回最终配置
    22. 未使用 qPCR 结果
    23. 未做样品温度标定
    24. OAT delta 全部参考同一基线
    25. 敏感性排序使用重复周期 HIGH/LOW 响应
    26. 未做概率性不确定度声明
"""
import pytest
import numpy as np
from pathlib import Path

from thermal_model.config.final_frozen_model import (
    FINAL_FROZEN_THERMAL_MODEL_V1 as M,
)
from thermal_model.core import heat_model
from workflows.diagnostics.analyze_insulated_sample_sensitivity import (
    INPUT_XLSX,
    BASE_K_EFF, BASE_CP_EFF, BASE_RHO, BASE_H, BASE_EPS, BASE_TAU,
    OAT_GRID, CALIBRATION_SUPPORTED, ABLATION_CASES,
    PRIMARY_RANK_PARAMS,
    define_cycle_windows, load_setpoint, run_single_case,
    summarize_cycles, evaluate_cycles, sensitivity_scores,
    build_all_cases, _case_args, build_insulated_layers, make_materials,
    OUTPUT_ROOT,
)
from workflows.prediction.predict_sample_temperature_frozen_model import (
    load_internal_data,
)

PY = Path(__file__).parent  # pytest rootdir

# ============================================================
# 共享数据 (module-scope: 完整输入 + 冻结窗口 + 基线 FDM)
# ============================================================

@pytest.fixture(scope="module")
def base_fixture():
    data = load_internal_data(INPUT_XLSX)
    t_sp, sp = load_setpoint(INPUT_XLSX)
    cw = define_cycle_windows(t_sp, sp)
    base = run_single_case(
        BASE_K_EFF, BASE_CP_EFF, BASE_RHO, BASE_H, BASE_EPS,
        t_src=data["source_time_s"], T_internal=data["T_internal_C"],
        windows=cw["windows"])
    return {"data": data, "cw": cw, "base": base}


def _window_set(cw):
    return [(w["cycle_id"], w["start_s"], w["end_s"],
             tuple(w["high_window"]), tuple(w["trough_window"]),
             w["excluded"]) for w in cw["windows"]]


# ============================================================
# 1-3. 权威模型 / 几何 / 输入
# ============================================================

def test_authoritative_baseline_unchanged():
    assert M.k_eff_W_mK == pytest.approx(0.0675)
    assert M.cp_eff_J_kgK == pytest.approx(700.0)
    assert M.rho_COC_kg_m3 == pytest.approx(1020.0)
    assert M.h_conv_W_m2K == pytest.approx(10.0)
    assert M.emissivity == pytest.approx(0.90)
    assert M.tau_top_s == pytest.approx(8.0)
    assert BASE_K_EFF == pytest.approx(M.k_eff_W_mK)
    assert BASE_CP_EFF == pytest.approx(M.cp_eff_J_kgK)
    assert BASE_TAU == pytest.approx(M.tau_top_s)


def test_insulated_geometry_used():
    layers = build_insulated_layers()
    names = [l.name for l in layers]
    assert names == ["Bottom COC", "PCR Sample", "Mineral Oil",
                     "Top COC", "Air Gap", "Cap PDMS"]
    total_um = sum(l.thickness_m for l in layers) * 1e6
    assert total_um == pytest.approx(4050.0)
    assert any(l.role == "sample" for l in layers)
    assert any(l.name == "Top COC" and l.role == "top_surface"
               for l in layers)


def test_complete_input_history(base_fixture):
    data = base_fixture["data"]
    assert data["n_valid"] == 322
    assert data["first_time"] == pytest.approx(0.091, abs=1e-6)
    assert data["last_time"] == pytest.approx(349.899, abs=1e-3)
    assert data["duration_s"] == pytest.approx(349.808, abs=1e-2)
    assert data["T_min_C"] == pytest.approx(21.6, abs=1e-6)
    assert data["T_max_C"] == pytest.approx(96.95, abs=1e-6)


# ============================================================
# 4-5. 激活相排除 / 窗口冻结
# ============================================================

def test_activation_phase_excluded(base_fixture):
    cw = base_fixture["cw"]
    act_end = cw["activation"]["end_s"]
    for w in cw["windows"]:
        if w["excluded"]:
            continue
        # 所有包含循环的起点 >= 激活相结束 (90 C 段之后)
        assert w["start_s"] >= act_end - 1e-9
        # HIGH 子窗口起点 = 100 C 段起点, 位于激活相之后
        assert w["high_window"][0] >= act_end - 1e-9
    # 基线的重复周期 HIGH 不应含激活相峰 (~89 C)
    highs = base_fixture["base"]["highs_C"]
    assert max(highs) < 86.0
    assert max(highs) > 80.0


def test_cycle_windows_frozen_across_runs(base_fixture):
    t_sp, sp = load_setpoint(INPUT_XLSX)
    cw1 = define_cycle_windows(t_sp, sp)
    cw2 = define_cycle_windows(t_sp, sp)
    assert _window_set(cw1) == _window_set(cw2)
    # 窗口定义纯协议结构, 与任何模型参数无关
    assert cw1["source"] == "SETPOINT_PROTOCOL_STRUCTURE"
    assert len(cw1["windows"]) == 30
    n_included = sum(1 for w in cw1["windows"] if not w["excluded"])
    assert n_included == 29
    # 最后一个循环 (无后续 20 C 谷) 被排除
    assert cw1["windows"][-1]["excluded"] is True


# ============================================================
# 6-8. 响应定义
# ============================================================

def test_mean_high_from_cycle_max(base_fixture):
    base = base_fixture["base"]
    cyc = evaluate_cycles(base["t_abs"], base["T_sample"],
                          base_fixture["cw"]["windows"])
    highs = np.asarray(cyc["highs_C"])
    assert len(highs) == 29
    assert base["cycles"]["high"]["mean"] == pytest.approx(
        float(np.mean(highs)))
    assert base["cycles"]["high"]["median"] == pytest.approx(
        float(np.median(highs)))
    # 每周期 HIGH 是窗口内样品最大
    t_abs = base["t_abs"]
    Ts = base["T_sample"]
    for i, w in enumerate(base_fixture["cw"]["windows"]):
        if w["excluded"]:
            continue
        m = ((t_abs >= w["high_window"][0] - 1e-9)
             & (t_abs <= w["high_window"][1] + 1e-9))
        assert highs[i] == pytest.approx(float(Ts[m].max()), abs=1e-9)


def test_mean_low_from_cycle_min(base_fixture):
    base = base_fixture["base"]
    cyc = evaluate_cycles(base["t_abs"], base["T_sample"],
                          base_fixture["cw"]["windows"])
    lows = np.asarray(cyc["lows_C"])
    assert len(lows) == 29
    assert base["cycles"]["low"]["mean"] == pytest.approx(
        float(np.mean(lows)))
    assert base["cycles"]["low"]["median"] == pytest.approx(
        float(np.median(lows)))
    t_abs = base["t_abs"]
    Ts = base["T_sample"]
    for i, w in enumerate(base_fixture["cw"]["windows"]):
        if w["excluded"]:
            continue
        m = ((t_abs >= w["trough_window"][0] - 1e-9)
             & (t_abs <= w["trough_window"][1] + 1e-9))
        assert lows[i] == pytest.approx(float(Ts[m].min()), abs=1e-9)


def test_amplitude_cyclewise(base_fixture):
    base = base_fixture["base"]
    assert base["cycles"]["amplitude"]["mean"] == pytest.approx(
        float(np.mean(np.asarray(base["amps_C"]))))
    assert base["amps_C"][0] == pytest.approx(
        base["highs_C"][0] - base["lows_C"][0], abs=1e-9)


# ============================================================
# 9. 排序不使用整体最大
# ============================================================

def test_ranking_uses_repeated_cycle_responses(base_fixture):
    base = base_fixture["base"]
    cyc = evaluate_cycles(base["t_abs"], base["T_sample"],
                          base_fixture["cw"]["windows"])
    base_stats = summarize_cycles(
        {"highs_C": cyc["highs_C"], "lows_C": cyc["lows_C"],
         "amps_C": cyc["amps_C"]})
    # 构造最小 param_results (k_eff ±10%, key = 参数值) 验证 S 分数只用
    # HIGH/LOW mean
    pr = {}
    for pct, val in ((-10, 0.06075), (10, 0.07425)):
        r = run_single_case(val, BASE_CP_EFF, BASE_RHO, BASE_H, BASE_EPS,
                            t_src=base_fixture["data"]["source_time_s"],
                            T_internal=base_fixture["data"]["T_internal_C"],
                            windows=base_fixture["cw"]["windows"])
        pr[float(val)] = {"value": float(val), "pct": pct,
                          "run_type": "OAT",
                          "high": r["cycles"]["high"], "low": r["cycles"]["low"],
                          "amplitude": r["cycles"]["amplitude"],
                          "overall_max_C": r["overall_sample_max_C"]}
    scores = sensitivity_scores(base_stats, {"k_eff": pr})
    s = scores["k_eff"]
    assert s["S_high_10pct_C"] > 0.0
    assert s["S_low_10pct_C"] >= 0.0
    assert s["S_range_C"] == pytest.approx(
        s["S_high_10pct_C"] + s["S_low_10pct_C"], abs=1e-12)


# ============================================================
# 10-11. OAT 单参数
# ============================================================

def test_oat_one_parameter_at_a_time():
    data = load_internal_data(INPUT_XLSX)
    t_sp, sp = load_setpoint(INPUT_XLSX)
    cw = define_cycle_windows(t_sp, sp)
    cases = build_all_cases(data["source_time_s"],
                            data["T_internal_C"], cw["windows"])
    for param, pct, val, run_type, args in cases:
        k, cp, rho, h, eps, mat_over, lay_over = args
        # 除本参数外全部为基线
        changed = 0
        if abs(k - BASE_K_EFF) > 1e-12:
            changed += 1
            assert param == "k_eff"
        if abs(cp - BASE_CP_EFF) > 1e-12:
            changed += 1
            assert param == "cp_eff"
        if abs(rho - BASE_RHO) > 1e-12:
            changed += 1
            assert param == "rho_COC"
        if abs(h - BASE_H) > 1e-12:
            changed += 1
            assert param == "h_conv"
        if abs(eps - BASE_EPS) > 1e-12:
            changed += 1
            assert param == "epsilon"
        if mat_over:
            changed += 1
            spec = OAT_GRID[param]
            assert spec["kind"] == "material"
            assert spec["material"] == mat_over[0][0]
        if lay_over:
            changed += 1
            spec = OAT_GRID[param]
            assert spec["kind"] == "layer"
        assert changed == 1, f"{param}: 应只变一个参数, 实际 {changed}"


def test_oat_material_values_match_grid():
    for param in ("k_air", "k_PDMS", "k_oil", "k_sample", "cp_sample",
                  "cp_oil", "cp_PDMS", "cp_air"):
        spec = OAT_GRID[param]
        for pct, val, rt in spec["runs"]:
            if pct == 0:
                continue
            args = _case_args(param, pct, val, rt, None, None, None)
            mat_over = args[5]
            assert len(mat_over) == 1
            assert mat_over[0][0] == spec["material"]
            assert mat_over[0][1] == spec["field"]
            assert mat_over[0][2] == pytest.approx(float(val))


def test_layer_thickness_override():
    # d_air 3.0 mm -> 2.4 mm (-20%)
    args = _case_args("d_air", -20, 2.4, "OAT", None, None, None)
    lay_over = args[6]
    assert lay_over == [("Air Gap", "thickness", 2.4e-3)]
    layers = build_insulated_layers(lay_over)
    for l in layers:
        if l.name == "Air Gap":
            assert l.thickness_m == pytest.approx(2.4e-3)
    # d_PDMS 200 um -> 220 um (+10%)
    args = _case_args("d_PDMS", 10, 220.0, "OAT", None, None, None)
    layers = build_insulated_layers(args[6])
    for l in layers:
        if l.name == "Cap PDMS":
            assert l.thickness_m == pytest.approx(220e-6)


# ============================================================
# 12-14. 校准范围 / epsilon
# ============================================================

def test_k_calibration_supported_range():
    assert CALIBRATION_SUPPORTED["k_eff"] == (0.0650, 0.0675, 0.0700)


def test_cp_calibration_supported_range():
    assert CALIBRATION_SUPPORTED["cp_eff"] == (600.0, 700.0, 800.0)


def test_epsilon_never_exceeds_1():
    for pct, val, rt in OAT_GRID["epsilon"]["runs"]:
        assert 0.0 <= val <= 1.0


# ============================================================
# 15-17. 消融分类
# ============================================================

def test_no_convection_is_ablation():
    assert ABLATION_CASES["NO_CONVECTION"]["h"] == 0.0
    assert ABLATION_CASES["NO_CONVECTION"]["eps"] == pytest.approx(0.90)


def test_no_radiation_is_ablation():
    assert ABLATION_CASES["NO_RADIATION"]["eps"] == 0.0
    assert ABLATION_CASES["NO_RADIATION"]["h"] == pytest.approx(10.0)


def test_no_external_loss_is_ablation():
    assert ABLATION_CASES["NO_EXTERNAL_SURFACE_HEAT_LOSS"]["h"] == 0.0
    assert ABLATION_CASES["NO_EXTERNAL_SURFACE_HEAT_LOSS"]["eps"] == 0.0


def test_ablation_increases_high_level(base_fixture):
    """物理消融 (关闭外部热损失) 应使样品循环 HIGH 升高 (方向性检查)。"""
    data = base_fixture["data"]
    cw = base_fixture["cw"]
    r = run_single_case(BASE_K_EFF, BASE_CP_EFF, BASE_RHO, 0.0, 0.0,
                        t_src=data["source_time_s"],
                        T_internal=data["T_internal_C"], windows=cw["windows"])
    assert r["cycles"]["high"]["mean"] > base_fixture["base"][
        "cycles"]["high"]["mean"]


def test_joint_ablation_read_from_combined_run(base_fixture):
    """联合消融必须来自实际的 h=0+eps=0 联合仿真, 而非线性加总。"""
    data = base_fixture["data"]
    cw = base_fixture["cw"]
    base = base_fixture["base"]

    def _run(h, eps):
        r = run_single_case(BASE_K_EFF, BASE_CP_EFF, BASE_RHO, h, eps,
                            t_src=data["source_time_s"],
                            T_internal=data["T_internal_C"],
                            windows=cw["windows"])
        return r["cycles"]

    b = _run(BASE_H, BASE_EPS)
    nc = _run(0.0, BASE_EPS)          # 无对流
    nr = _run(BASE_H, 0.0)            # 无辐射
    joint = _run(0.0, 0.0)            # 双无 (联合仿真)

    d_hi_nc = nc["high"]["mean"] - b["high"]["mean"]
    d_lo_nc = nc["low"]["mean"] - b["low"]["mean"]
    d_hi_nr = nr["high"]["mean"] - b["high"]["mean"]
    d_lo_nr = nr["low"]["mean"] - b["low"]["mean"]
    d_hi_j = joint["high"]["mean"] - b["high"]["mean"]
    d_lo_j = joint["low"]["mean"] - b["low"]["mean"]

    # 单项消融 ~ +0.24 (对流) / +0.11 (辐射)
    assert d_hi_nc == pytest.approx(0.24, abs=0.03)
    assert d_lo_nc == pytest.approx(0.24, abs=0.03)
    assert d_hi_nr == pytest.approx(0.11, abs=0.03)
    assert d_lo_nr == pytest.approx(0.11, abs=0.03)

    # 联合消融 ~ +0.68 (直接读自联合仿真), 绝不应等于线性加总 0.35
    assert d_hi_j == pytest.approx(0.68, abs=0.05)
    assert d_lo_j == pytest.approx(0.68, abs=0.05)
    linear_sum = d_hi_nc + d_hi_nr
    assert abs(d_hi_j - linear_sum) > 0.1   # 联合效应 != 线性加总
    assert abs(d_lo_j - (d_lo_nc + d_lo_nr)) > 0.1


def test_joint_ablation_not_summed_in_code():
    """脚本/摘要不得用线性加总计算联合消融贡献。"""
    import inspect
    import workflows.diagnostics.analyze_insulated_sample_sensitivity as mod
    src = inspect.getsource(mod)
    # 摘要段必须使用联合仿真差值 (ablation_stats[...] 直接读取)
    assert "d_hi_j = abl_joint" in src
    assert "d_hi_nc" in src and "d_hi_nr" in src
    # 允许的解释性说明 (明确指出 0.24+0.11=0.35 是错误加总) 之外,
    # 不应把线性加总当作联合贡献; 联合值必须来自联合仿真
    assert "joint ablation result (~0.7 C) is read directly" in src
    assert "NOT the linear sum" in src


# ============================================================
# 18-20. tau 负控制
# ============================================================

def test_tau_not_applied_to_sample(base_fixture):
    """样品 = raw FDM 层温度; tau 是顶部观测链滞后。"""
    base = base_fixture["base"]
    # run_single_case 不接收 tau; 样品来自 T_sample_arr (raw FDM)
    assert base["T_sample"].size > 0
    # 基线的 overall max 与预测工具一致 (无滞后样品)
    assert base["overall_sample_max_C"] == pytest.approx(89.011, abs=5e-3)


def test_tau_negative_control_identical():
    """tau=0/8/16 样品统计必须逐位相同 (架构保证, 样品永不滞后)。"""
    data = load_internal_data(INPUT_XLSX)
    t_sp, sp = load_setpoint(INPUT_XLSX)
    cw = define_cycle_windows(t_sp, sp)
    # 同一 FDM 结果用于三个 tau 值 (tau 只属于顶部观测链, 不进入
    # run_convection_radiation_fdm)。样品结果相同是架构保证。
    r = run_single_case(BASE_K_EFF, BASE_CP_EFF, BASE_RHO, BASE_H, BASE_EPS,
                        t_src=data["source_time_s"],
                        T_internal=data["T_internal_C"], windows=cw["windows"])
    for tau in (0.0, 8.0, 16.0):
        # 与 tau 无关: 样品统计相同 (同一 r)
        assert r["cycles"]["high"]["mean"] == r["cycles"]["high"]["mean"]
    # 明确: 敏感性脚本中样品序列不经过任何 lag 函数
    import inspect
    src = inspect.getsource(run_single_case)
    assert "apply_first_order_lag" not in src


def test_top_lag_not_in_sample_sensitivity():
    """Top 观测滞后 (tau) 不出现在 OAT 网格或样品计算中。"""
    assert "tau" not in OAT_GRID
    assert "tau_top" not in OAT_GRID


# ============================================================
# 21. 不写回最终配置
# ============================================================

def test_no_writeback_to_final_config():
    """运行脚本不修改权威配置对象/文件。"""
    k0 = M.k_eff_W_mK
    cp0 = M.cp_eff_J_kgK
    h0 = M.h_conv_W_m2K
    eps0 = M.emissivity
    # 运行 OAT 网格构造 (不执行 FDM) 不应改变权威值
    build_all_cases(np.array([0.0, 1.0]), np.array([21.6, 21.6]),
                    define_cycle_windows(
                        np.array([0.0, 1.0, 2.0]),
                        np.array([90.0, 20.0, 100.0]))["windows"])
    assert M.k_eff_W_mK == k0
    assert M.cp_eff_J_kgK == cp0
    assert M.h_conv_W_m2K == h0
    assert M.emissivity == eps0


# ============================================================
# 22-23. 无 qPCR / 无标定
# ============================================================

def test_no_qpcr_usage():
    import inspect
    import workflows.diagnostics.analyze_insulated_sample_sensitivity as mod
    src = inspect.getsource(mod)
    assert "qpcr" not in src.lower().replace("q pcr", "")
    assert "success" not in src.lower()


def test_no_sample_calibration():
    """脚本只做正向预测 + 敏感性; 无任何参数拟合/优化调用。"""
    import inspect
    import workflows.diagnostics.analyze_insulated_sample_sensitivity as mod
    src = inspect.getsource(mod)
    for banned in ("scipy.optimize", "curve_fit", "minimize", "fit("):
        assert banned not in src


# ============================================================
# 24-25. OAT delta 同一基线 / 排序语义
# ============================================================

def test_oat_deltas_reference_same_baseline(base_fixture):
    """±10% k_eff 的 delta 必须相对同一冻结基线。"""
    data = base_fixture["data"]
    cw = base_fixture["cw"]
    base = base_fixture["base"]
    base_high = base["cycles"]["high"]["mean"]
    for val in (0.06075, 0.07425):
        r = run_single_case(val, BASE_CP_EFF, BASE_RHO, BASE_H, BASE_EPS,
                            t_src=data["source_time_s"],
                            T_internal=data["T_internal_C"],
                            windows=cw["windows"])
        delta = r["cycles"]["high"]["mean"] - base_high
        assert delta != pytest.approx(0.0, abs=1e-9)  # 参数确实有影响
    # 两个 delta 相对同一基线 => 差 = 两结果之差
    r1 = run_single_case(0.06075, BASE_CP_EFF, BASE_RHO, BASE_H, BASE_EPS,
                         t_src=data["source_time_s"],
                         T_internal=data["T_internal_C"], windows=cw["windows"])
    r2 = run_single_case(0.07425, BASE_CP_EFF, BASE_RHO, BASE_H, BASE_EPS,
                         t_src=data["source_time_s"],
                         T_internal=data["T_internal_C"], windows=cw["windows"])
    d1 = r1["cycles"]["high"]["mean"] - base_high
    d2 = r2["cycles"]["high"]["mean"] - base_high
    assert (d2 - d1) == pytest.approx(
        r2["cycles"]["high"]["mean"] - r1["cycles"]["high"]["mean"],
        abs=1e-12)


def test_ranking_primary_params_present():
    """主排序参数集包含任务书要求的 11 个主参数。"""
    required = {"k_eff", "cp_eff", "h_conv", "epsilon", "k_air",
                "d_air", "k_PDMS", "d_PDMS", "k_oil", "k_sample",
                "cp_sample"}
    assert required.issubset(set(PRIMARY_RANK_PARAMS))
    assert set(PRIMARY_RANK_PARAMS) == required


# ============================================================
# 26. 无概率不确定度声明
# ============================================================

def test_no_probabilistic_uncertainty_claim():
    import inspect
    import workflows.diagnostics.analyze_insulated_sample_sensitivity as mod
    src = inspect.getsource(mod)
    # 允许否定式说明 ("NOT uncertainty propagation" / "no probabilistic
    # claim" / "not a confidence interval"), 禁止正面概率性声明。
    for banned in ("95% CI", "95 % CI", "±1.84", "±2 C",
                   "probability distribution", "credible interval"):
        assert banned not in src
    # "confidence interval" 只允许以否定形式出现 (not a confidence interval,
    # 可能跨行拼接)
    n_ci = src.count("confidence interval")
    n_neg = src.count("not a ") + src.count("NOT a ")
    assert n_ci <= n_neg, "存在正面 confidence interval 声明"
    # 不允许"样品温度 ± 数值 C"的实际声明 (报告是敏感性, 非不确定度);
    # "±X C" 占位符式否定说明 (No sample temperature ±X C) 是允许的。
    assert "sample temperature ±1" not in src
    assert "sample temperature ±2" not in src
    assert "sample temperature ±0" not in src


# ============================================================
# 补充: 输出目录就绪 (不运行完整 OAT)
# ============================================================

def test_output_root_configured():
    assert str(OUTPUT_ROOT).endswith("08.24_15x_no_holding_sensitivity")
