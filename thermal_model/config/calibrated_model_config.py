#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标定的裸顶模型 —— 系统级有效热参数配置。

本模块定义「已接受的」标定结果 / 历史标定结果, 供:
    - 最终校准模型运行 (run_calibrated_thermal_model.py);
    - 未来热协议重建 / 样品层热重建;
    - 需要「当前有效标定值」的任何调用方。

重要 —— 参数解释
----------------
    k_eff / cp_eff 是 SYSTEM-LEVEL EFFECTIVE (系统级有效) 热参数,
    不是 TOPAS COC 的固有材料常数。它们允许吸收:
        传感器动态 / 传感器不确定度 / 热接触效应 /
        未解析的热阻与热惯性 / 1D 模型简化 / 材料性质不确定度。
    严禁在任何输出中把它们描述为:
        intrinsic COC conductivity / intrinsic COC specific heat /
        measured TOPAS COC constants。

历史说明 (重要)
----------------
    V1/V2 参数扫描的目标函数存在插值查询轴 bug:
        旧实现用实测温度值作为 np.interp 的查询坐标,
        而不是实测时间坐标 (temperature-as-time query)。
    因此 V1/V2 的选参结果 (包括 0.068 / 9200) 是:
        HISTORICAL / PROVISIONAL, 不适用于最终标定。
    0.068 / 9200 保存在 LEGACY_OBJECTIVE_PROVISIONAL_CALIBRATION 中,
    仅作历史参考, 不得作为最终校准模型参数使用。
    修正后的 V3 标定 (corrected_time_objective_v3) 使用
    实测时间坐标插值, 其结果为当前有效标定 (若 V3 完成)。

默认材料库 (DEFAULT_MATERIALS) 保持为材料/参考配置, 本模块不修改它。
"""

from dataclasses import dataclass

from thermal_model.core.heat_model import BARE_TOP_COC_LAYERS, copy_default_materials


@dataclass(frozen=True)
class EffectiveThermalCalibration:
    """一次系统级有效热参数标定 (不可变)。

    字段:
        status                 : "accepted" (当前有效) / "historical_provisional";
        valid_for_final_calibration : 是否可用于最终标定;
        selection_objective    : 用于选参的目标函数;
        note                   : 附注。
    """
    name: str
    k_eff_W_mK: float
    cp_eff_J_kgK: float
    rho_COC_kg_m3: float
    geometry_preset: str
    source_analysis: str
    interpretation: str
    status: str = "accepted"
    valid_for_final_calibration: bool = True
    selection_objective: str = ""
    note: str = ""


# -------------------------------------------------------------
# 历史/暂定标定 —— 旧目标 (temperature-as-time query) 选出
# -------------------------------------------------------------
LEGACY_OBJECTIVE_PROVISIONAL_CALIBRATION = EffectiveThermalCalibration(
    name="legacy_objective_provisional_0p068_9200",
    k_eff_W_mK=0.068,
    cp_eff_J_kgK=9200.0,
    rho_COC_kg_m3=1020.0,
    geometry_preset="BARE_TOP_COC_LAYERS",
    source_analysis="system_effective_extended_v2 (INVALID OBJECTIVE)",
    interpretation=(
        "system-level effective thermal parameters (NOT intrinsic COC "
        "material constants)."
    ),
    status="historical_provisional",
    valid_for_final_calibration=False,
    selection_objective="legacy_temperature_as_time_query",
    note=(
        "Selected under the INVALID legacy objective (temperature values "
        "used as interpolation query coordinates). Corrected-time RMSE at "
        "this pair is ~7.43 C. Do NOT use as final calibrated model."
    ),
)

# -------------------------------------------------------------
# 当前接受的名义标定 —— V3 修正时间目标 (corrected_time_objective_v3)
# -------------------------------------------------------------
NOMINAL_BARE_TOP_CALIBRATION_V1 = EffectiveThermalCalibration(
    name="bare_top_72C_corrected_time_objective_v1",
    k_eff_W_mK=0.0165,
    cp_eff_J_kgK=900.0,
    rho_COC_kg_m3=1020.0,
    geometry_preset="BARE_TOP_COC_LAYERS",
    source_analysis="corrected_time_objective_v3",
    interpretation=(
        "system-level effective thermal parameters (NOT intrinsic COC "
        "material constants); may absorb sensor dynamics/uncertainty, "
        "contact effects, unresolved thermal resistance/inertia, 1D model "
        "simplification, and material-property uncertainty."
    ),
    status="accepted",
    valid_for_final_calibration=True,
    selection_objective="corrected_measurement_time_rmse",
    note=(
        "Selected with the corrected measurement-time interpolation "
        "objective (V3). RMSE = 0.7337 C, MAE = 0.5628 C, "
        "mean residual = -0.2727 C at 299 equal-weight points."
    ),
)


def make_nominal_calibrated_materials(calibration=None):
    """构造标定模型的材料库 (独立副本, 不修改 DEFAULT_MATERIALS)。

    - 从材料库副本开始;
    - 仅替换共享 COC 材料为 k_eff / cp_eff / rho (Bottom COC 与 Top COC
      都引用 "COC", 自动使用相同标定值);
    - Water / Oil 等其余材料逐位不变。

    默认 (calibration=None) 使用 LEGACY_OBJECTIVE_PROVISIONAL_CALIBRATION
    (历史/暂定)。V3 完成后将切换为 V3 参数。
    """
    cal = calibration if calibration is not None else NOMINAL_BARE_TOP_CALIBRATION_V1
    mats = copy_default_materials()
    coc = mats["COC"]
    mats["COC"] = type(coc)(
        name=coc.name,
        k_W_mK=cal.k_eff_W_mK,
        rho_kg_m3=cal.rho_COC_kg_m3,
        cp_J_kgK=cal.cp_eff_J_kgK,
    )
    return mats


def nominal_layer_stack(calibration=None):
    """返回名义标定模型的层叠结构 (当前为裸顶 850 um 实验几何)。"""
    return BARE_TOP_COC_LAYERS
