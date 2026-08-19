#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标定的裸顶模型 —— 名义系统级有效热参数配置 (唯一权威定义)。

本模块定义「已接受的」标定结果, 供:
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

来源
----
    参数扫描 V2: parameter_scan_output/72C/system_effective_extended_v2/
    (k_eff=0.068, cp_eff=9200, RMSE=4.7449 C, MAE=2.3973 C,
     mean residual=-0.2241 C; 内点盆地, 见 extended_combined_scan.csv)

默认材料库 (DEFAULT_MATERIALS) 保持为材料/参考配置, 本模块不修改它。
"""

from dataclasses import dataclass

from heat_model import BARE_TOP_COC_LAYERS, copy_default_materials


@dataclass(frozen=True)
class EffectiveThermalCalibration:
    """一次已接受的系统级有效热参数标定 (不可变)。"""
    name: str
    k_eff_W_mK: float
    cp_eff_J_kgK: float
    rho_COC_kg_m3: float
    geometry_preset: str
    source_analysis: str
    interpretation: str


# 名义标定 V1 —— 当前项目接受的校准配置
NOMINAL_BARE_TOP_CALIBRATION_V1 = EffectiveThermalCalibration(
    name="bare_top_72C_system_effective_v1",
    k_eff_W_mK=0.068,
    cp_eff_J_kgK=9200.0,
    rho_COC_kg_m3=1020.0,
    geometry_preset="BARE_TOP_COC_LAYERS",
    source_analysis="system_effective_extended_v2",
    interpretation=(
        "system-level effective thermal parameters (NOT intrinsic COC "
        "material constants); may absorb sensor dynamics/uncertainty, "
        "contact effects, unresolved thermal resistance/inertia, 1D model "
        "simplification, and material-property uncertainty."
    ),
)


def make_nominal_calibrated_materials(calibration=None):
    """构造标定模型的材料库 (独立副本, 不修改 DEFAULT_MATERIALS)。

    - 从材料库副本开始;
    - 仅替换共享 COC 材料为 k_eff / cp_eff / rho (Bottom COC 与 Top COC
      都引用 "COC", 自动使用相同标定值);
    - Water / Oil 等其余材料逐位不变。
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
