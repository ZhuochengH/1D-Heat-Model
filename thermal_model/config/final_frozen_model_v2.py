#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL FROZEN THERMAL MODEL V2 — 唯一权威最终模型配置 (FC-70 + no-PDMS)
======================================================================

模型 ID:
    FINAL_FROZEN_THERMAL_MODEL_V2

状态:
    FROZEN (2026-08-30)。
    从 THERMAL_MODEL_V2_CANDIDATE 晋升 (经 recalibration / 独立外部验证 /
    sample prediction / sensitivity validation 全部完成后)。
    任何未来对 k / cp / tau / h / epsilon / 几何 / 滞后位置 / 材料的修改
    都必须作为一个新模型版本, 而不是静默修改 V2。

与 V1 的两项科学变更:
  CHANGE A — 材料: 通用矿物油 -> 3M Fluorinert FC-70 (Sigma-Aldrich F9880)。
             k_FC70 = 0.070 W/(m K), rho_FC70 = 1940 kg/m3,
             cp_FC70 = 1050 J/(kg K) (室温代表性常数, 无温度依赖)。
  CHANGE B — 绝缘几何: 移除 PDMS 层 (薄聚合物密封层在简化模型中有意忽略)。
             新绝缘堆栈: Bottom COC -> sample -> FC-70 -> Top COC
                         -> sealed air (3 mm) -> external boundary。
             裸顶标定/验证几何仅 oil -> FC-70 替换, 结构不变。

最终有效参数 (系统级降阶有效参数, 不是 TOPAS 固有材料常数):
    k_eff   = 0.0700 W/(m K)
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

几何:
    bare      : Bottom COC 180 / Sample 20 / FC-70 50 / Top COC 600 um
                (总 850 um, 无 Air / PDMS)
    insulated : Bottom COC 180 / Sample 20 / FC-70 50 / Top COC 600 /
                Sealed air 3000 um (总 3850 um, 无 PDMS)

权威证据:
    66C 标定 RMSE = 0.6333 C
    60C 外部验证   = 1.3139 C (已知偏移, 权威)
    72C 外部验证   = 3.0132 C (已知偏移, 权威; 冷却相局限同 V1)
    3s  外部验证   = 1.0386 C (权威)
    外部验证均值   = 1.7886 C
    样品预测 (08.24 no-holding, 29 周期): mean HIGH 84.703 C /
    mean LOW 58.686 C / range 26.017 C
    Sensitivity: 与 V1 总体一致 (k_eff 最敏感, cp_eff/rho_COC 主导)

V1 保留:
    FINAL_FROZEN_THERMAL_MODEL_V1 继续存在, 作为 legacy/reproducibility
    模型保留历史 Oil + PDMS 定义 (绝不删除)。

数值表述:
    本模型使用一维节点中心有限体积 (node-centered finite-volume) 离散
    (physical finite-volume solution)。历史代码/函数名中的 "fdm" 仅为
    历史命名, 不表示数值方法本身是 FDM。
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from thermal_model.core import heat_model
from thermal_model.core.convection_radiation_thermal_model import (
    H_CONV_STRATEGY_E_W_M2K,
    EMISSIVITY_STRATEGY_E,
    SIGMA_SB_W_M2_K4,
    VIEW_FACTOR_STRATEGY_E,
)
from thermal_model.config.thermal_model_v2_candidate import (
    FC70_K_W_MK,
    FC70_RHO_KG_M3,
    FC70_CP_J_KGK,
    V2_BARE_TOP_COC_LAYERS,
    V2_INSULATED_NO_PDMS_LAYERS,
)

MODEL_ID = "FINAL_FROZEN_THERMAL_MODEL_V2"
DATE_PROMOTED = "2026-08-30"
PREDECESSOR_ID = "THERMAL_MODEL_V2_CANDIDATE"

# ---- 校准有效参数 (系统级降阶有效参数, 不是固有材料常数) ----
K_EFF_W_MK = 0.0700
CP_EFF_J_KGK = 700.0
RHO_COC_KG_M3 = 1020.0
TAU_TOP_S = 8.0

# ---- 固定边界 (不拟合) ----
H_CONV_W_M2K = H_CONV_STRATEGY_E_W_M2K          # 10.0
EMISSIVITY = EMISSIVITY_STRATEGY_E               # 0.90
SIGMA_SB_W_M2K4 = SIGMA_SB_W_M2_K4              # 5.670374419e-8
VIEW_FACTOR = VIEW_FACTOR_STRATEGY_E             # 1.0

# ---- FC-70 (唯一权威材料定义引用, 源自 candidate 模块) ----
FC70_K_W_MK_FINAL = FC70_K_W_MK                 # 0.070
FC70_RHO_KG_M3_FINAL = FC70_RHO_KG_M3           # 1940
FC70_CP_J_KGK_FINAL = FC70_CP_J_KGK             # 1050

# ---- 权威验证证据 (只读) ----
CALIBRATION_66C_RMSE_C = 0.6333
EXTERNAL_VALIDATION_RMSE_C = {
    "60C": 1.3139,
    "72C": 3.0132,
    "3s_extension": 1.0386,
}
EXTERNAL_VALIDATION_MEAN_RMSE_C = 1.7886
EXTERNAL_VALIDATION_MEDIAN_RMSE_C = 1.3139
EXTERNAL_VALIDATION_WORST_RMSE_C = 3.0132

# 样品预测参考 (08.24 no-holding, Setpoint 周期, 29 周期)
SAMPLE_PREDICTION = {
    "dataset": "08.24 am_15x primers, no holding",
    "cycle_source": "SETPOINT_PROTOCOL_STRUCTURE",
    "n_complete_cycles": 29,
    "mean_high_C": 84.703,
    "mean_low_C": 58.686,
    "mean_range_C": 26.017,
    "median_high_C": 84.696,
    "median_low_C": 58.972,
    "overall_sample_max_C": 89.097,
}

KNOWN_LIMITATION = (
    "72C protocol shows a larger cooling-phase mismatch "
    "(cooling RMSE ~4 C); interpreted as a documented reduced-order "
    "transient limitation. Parameters were NOT altered to improve it. "
    "This limitation is inherited from V1."
)

# 科学状态
SAMPLE_DIRECTLY_MEASURED = False
INSULATED_GEOMETRY_INDEPENDENTLY_VALIDATED = False


@dataclass(frozen=True)
class FinalFrozenThermalModelV2:
    """最终冻结模型 V2 (不可变; 防止意外修改)。"""
    model_id: str = MODEL_ID
    date_promoted: str = DATE_PROMOTED
    predecessor_id: str = PREDECESSOR_ID
    k_eff_W_mK: float = K_EFF_W_MK
    cp_eff_J_kgK: float = CP_EFF_J_KGK
    rho_COC_kg_m3: float = RHO_COC_KG_M3
    tau_top_s: float = TAU_TOP_S
    h_conv_W_m2K: float = H_CONV_W_M2K
    emissivity: float = EMISSIVITY
    sigma_SB_W_m2K4: float = SIGMA_SB_W_M2K4
    view_factor: float = VIEW_FACTOR
    fc70_k_W_mK: float = FC70_K_W_MK_FINAL
    fc70_rho_kg_m3: float = FC70_RHO_KG_M3_FINAL
    fc70_cp_J_kgK: float = FC70_CP_J_KGK_FINAL
    calibration_66C_RMSE_C: float = CALIBRATION_66C_RMSE_C
    external_validation_RMSE_C: Dict[str, float] = field(
        default_factory=lambda: dict(EXTERNAL_VALIDATION_RMSE_C))
    external_validation_mean_RMSE_C: float = EXTERNAL_VALIDATION_MEAN_RMSE_C

    def __post_init__(self) -> None:
        for name, val in (("k_eff_W_mK", self.k_eff_W_mK),
                          ("cp_eff_J_kgK", self.cp_eff_J_kgK),
                          ("rho_COC_kg_m3", self.rho_COC_kg_m3),
                          ("h_conv_W_m2K", self.h_conv_W_m2K),
                          ("fc70_k_W_mK", self.fc70_k_W_mK),
                          ("fc70_rho_kg_m3", self.fc70_rho_kg_m3),
                          ("fc70_cp_J_kgK", self.fc70_cp_J_kgK)):
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
    def bare_layers(self) -> List:
        """权威裸顶几何 (Bottom COC 180 / Sample 20 / FC-70 50 / Top COC 600 um)。"""
        return V2_BARE_TOP_COC_LAYERS

    @property
    def insulated_layers(self) -> List:
        """权威简化绝缘几何 (无 PDMS; 密封空气 3 mm, 总 3850 um)。"""
        return V2_INSULATED_NO_PDMS_LAYERS


FINAL_FROZEN_THERMAL_MODEL_V2 = FinalFrozenThermalModelV2()


def final_model_dict() -> dict:
    """最终模型 V2 纯 dict 描述 (元数据用)。"""
    m = FINAL_FROZEN_THERMAL_MODEL_V2
    return {
        "model_id": m.model_id,
        "date_promoted": m.date_promoted,
        "predecessor_id": m.predecessor_id,
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
        "fc70_k_W_mK": m.fc70_k_W_mK,
        "fc70_rho_kg_m3": m.fc70_rho_kg_m3,
        "fc70_cp_J_kgK": m.fc70_cp_J_kgK,
        "material": "3M Fluorinert FC-70 (Sigma-Aldrich F9880)",
        "radiation": "nonlinear Stefan-Boltzmann",
        "lag_placement": "output-side (Top observation only; never sample)",
        "geometry_bare": "Bottom COC 180 / Sample 20 / FC-70 50 / Top COC 600 um",
        "geometry_insulated": ("Bottom COC 180 / Sample 20 / FC-70 50 / "
                               "Top COC 600 / Sealed air 3000 um; NO PDMS"),
        "calibration_66C_RMSE_C": m.calibration_66C_RMSE_C,
        "external_validation_RMSE_C": m.external_validation_RMSE_C,
        "external_validation_mean_RMSE_C": m.external_validation_mean_RMSE_C,
        "sample_prediction": dict(SAMPLE_PREDICTION),
        "known_limitation": KNOWN_LIMITATION,
        "sample_directly_measured": SAMPLE_DIRECTLY_MEASURED,
        "insulated_geometry_independently_validated":
            INSULATED_GEOMETRY_INDEPENDENTLY_VALIDATED,
        "numerical_method": (
            "one-dimensional node-centered finite-volume model "
            "(physical finite-volume solution)"),
    }
