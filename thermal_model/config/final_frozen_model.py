#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL FROZEN THERMAL MODEL V1 — 唯一权威最终模型配置
====================================================

模型 ID:
    FINAL_FROZEN_THERMAL_MODEL_V1

状态:
    FROZEN (2026-08-22)。
    任何未来对 k / cp / tau / h / epsilon / 几何 / 滞后位置的修改
    都必须作为一个新模型版本, 而不是静默修改 V1。

来源:
    - 66C_RECALIBRATED_CANDIDATE_V1 (修正后 66C redo 数据集的重新标定);
    - 经实验已知计时偏移 (SETPOINT_90C_EVENT_PLUS_1S) 的多数据集
      零重拟合验证后晋升。

最终有效参数 (系统级降阶有效参数, 不是 TOPAS 固有材料常数):
    k_eff   = 0.0675 W/(m K)
    cp_eff  = 700 J/(kg K)
    rho     = 1020 kg/m3
    tau_top = 8.0 s   (输出侧有效滞后, 仅作用于 Top 观测模型;
                        绝不作用于样品温度)

固定边界 (不拟合):
    h_conv  = 10.0 W/(m2 K)
    epsilon = 0.90
    sigma_SB= 5.670374419e-8 W/(m2 K4)
    F_view  = 1.0
    非线性 Stefan-Boltzmann 辐射
    几何     = BARE_TOP_COC_LAYERS (标定/验证); 绝缘为前向扩展
              (LEGACY_INSULATED_LAYERS, 3 mm 密封空气 + 200 um PDMS,
              未独立验证)

权威证据:
    66C 标定 RMSE = 0.6368 C
    60C 外部验证   = 1.3749 C (已知偏移, 权威)
    72C 外部验证   = 3.0817 C (已知偏移, 权威; 冷却相 RMSE ~4.09 C 局限)
    3s  外部验证   = 1.0643 C (权威)
    外部验证均值   = 1.8403 C

架构:
    实测内部温度 -> 1D 多层 FDM + 对流 + 非线性辐射 -> raw Top COC
    -> 输出侧一阶滞后 tau_top -> 预测实测 Top COC。
    样品温度 = raw FDM 样品层 (控制体积加权), 绝无滞后。
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from thermal_model.core import heat_model
from thermal_model.core.convection_radiation_thermal_model import (
    H_CONV_STRATEGY_E_W_M2K,
    EMISSIVITY_STRATEGY_E,
    SIGMA_SB_W_M2_K4,
    VIEW_FACTOR_STRATEGY_E,
)

MODEL_ID = "FINAL_FROZEN_THERMAL_MODEL_V1"
DATE_PROMOTED = "2026-08-22"

K_EFF_W_MK = 0.0675
CP_EFF_J_KGK = 700.0
RHO_COC_KG_M3 = 1020.0
TAU_TOP_S = 8.0
H_CONV_W_M2K = H_CONV_STRATEGY_E_W_M2K          # 10.0
EMISSIVITY = EMISSIVITY_STRATEGY_E               # 0.90
SIGMA_SB_W_M2K4 = SIGMA_SB_W_M2_K4              # 5.670374419e-8
VIEW_FACTOR = VIEW_FACTOR_STRATEGY_E             # 1.0

# 权威验证证据 (只读)
CALIBRATION_66C_RMSE_C = 0.6368
EXTERNAL_VALIDATION_RMSE_C = {
    "60C": 1.3749,
    "72C": 3.0817,
    "3s_extension": 1.0643,
}
EXTERNAL_VALIDATION_MEAN_RMSE_C = 1.8403
EXTERNAL_VALIDATION_MEDIAN_RMSE_C = 1.3749
EXTERNAL_VALIDATION_WORST_RMSE_C = 3.0817
KNOWN_LIMITATION = (
    "72C protocol shows a larger cooling-phase mismatch "
    "(cooling RMSE ~4.09 C); interpreted as a documented reduced-order "
    "transient limitation. Parameters were NOT altered to improve it."
)

# 科学状态
SAMPLE_DIRECTLY_MEASURED = False
INSULATED_GEOMETRY_INDEPENDENTLY_VALIDATED = False


@dataclass(frozen=True)
class FinalFrozenThermalModelV1:
    """最终冻结模型 (不可变; 防止意外修改)。"""
    model_id: str = MODEL_ID
    date_promoted: str = DATE_PROMOTED
    k_eff_W_mK: float = K_EFF_W_MK
    cp_eff_J_kgK: float = CP_EFF_J_KGK
    rho_COC_kg_m3: float = RHO_COC_KG_M3
    tau_top_s: float = TAU_TOP_S
    h_conv_W_m2K: float = H_CONV_W_M2K
    emissivity: float = EMISSIVITY
    sigma_SB_W_m2K4: float = SIGMA_SB_W_M2K4
    view_factor: float = VIEW_FACTOR
    calibration_66C_RMSE_C: float = CALIBRATION_66C_RMSE_C
    external_validation_RMSE_C: Dict[str, float] = field(
        default_factory=lambda: dict(EXTERNAL_VALIDATION_RMSE_C))
    external_validation_mean_RMSE_C: float = EXTERNAL_VALIDATION_MEAN_RMSE_C

    def __post_init__(self) -> None:
        for name, val in (("k_eff_W_mK", self.k_eff_W_mK),
                          ("cp_eff_J_kgK", self.cp_eff_J_kgK),
                          ("rho_COC_kg_m3", self.rho_COC_kg_m3),
                          ("h_conv_W_m2K", self.h_conv_W_m2K)):
            if not (val > 0 and np.isfinite(val)):
                raise ValueError(f"{name} 必须 > 0 且有限: {val!r}")
        if not (self.tau_top_s >= 0 and np.isfinite(self.tau_top_s)):
            raise ValueError(f"tau_top_s 必须 >= 0 且有限: {self.tau_top_s!r}")
        if not (0.0 <= self.emissivity <= 1.0):
            raise ValueError(f"emissivity 必须在 [0,1]: {self.emissivity!r}")

    # ---- 派生 (公式计算, 不硬编码) ----
    @property
    def alpha_eff_m2_s(self) -> float:
        return self.k_eff_W_mK / (self.rho_COC_kg_m3 * self.cp_eff_J_kgK)

    @property
    def effusivity(self) -> float:
        return float(np.sqrt(
            self.k_eff_W_mK * self.rho_COC_kg_m3 * self.cp_eff_J_kgK))

    @property
    def bare_layers(self):
        """权威裸顶几何 (标定/外部验证配置)。"""
        return heat_model.BARE_TOP_COC_LAYERS

    @property
    def insulated_layers(self):
        """绝缘前向扩展几何 (3 mm 密封空气 + 200 um PDMS; 未独立验证)。"""
        return heat_model.LEGACY_INSULATED_LAYERS


FINAL_FROZEN_THERMAL_MODEL_V1 = FinalFrozenThermalModelV1()


def final_model_dict() -> dict:
    """最终模型纯 dict 描述 (元数据用)。"""
    m = FINAL_FROZEN_THERMAL_MODEL_V1
    return {
        "model_id": m.model_id,
        "date_promoted": m.date_promoted,
        "status": "FROZEN",
        "k_eff_W_mK": m.k_eff_W_mK,
        "cp_eff_J_kgK": m.cp_eff_J_kgK,
        "rho_COC_kg_m3": m.rho_COC_kg_m3,
        "tau_top_s": m.tau_top_s,
        "alpha_eff_m2_s": m.alpha_eff_m2_s,
        "effusivity": m.effusivity,
        "h_conv_W_m2K": m.h_conv_W_m2K,
        "emissivity": m.emissivity,
        "sigma_SB_W_m2K4": m.sigma_SB_W_m2K4,
        "view_factor": m.view_factor,
        "radiation": "nonlinear Stefan-Boltzmann",
        "lag_placement": "output-side (Top observation only; never sample)",
        "calibration_66C_RMSE_C": m.calibration_66C_RMSE_C,
        "external_validation_RMSE_C": m.external_validation_RMSE_C,
        "external_validation_mean_RMSE_C": m.external_validation_mean_RMSE_C,
        "known_limitation": KNOWN_LIMITATION,
        "sample_directly_measured": SAMPLE_DIRECTLY_MEASURED,
        "insulated_geometry_independently_validated":
            INSULATED_GEOMETRY_INDEPENDENTLY_VALIDATED,
    }
