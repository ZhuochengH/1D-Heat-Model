#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冻结的 Strategy G 保守跨协议预测候选
====================================

ID:
    strategy_G_conservative_cross_protocol_v1

状态:
    FROZEN FOR CROSS-PROTOCOL PREDICTION
    / NOT FINAL VALIDATED NOMINAL MODEL

参数:
    k_eff_W_mK  = 0.055
    cp_eff_J_kgK = 1200
    tau_lag_s   = 8.5
    rho_COC     = 1020 kg/m3 (固定)

派生 (不硬编码, 每次由公式计算):
    alpha_eff = k / (rho * cp)
    effusivity = sqrt(k * rho * cp)

固定边界 (Strategy E, 本模块不重复实现):
    h_conv = 10.0 W/(m2 K)
    epsilon_surface = 0.90
    sigma_SB = 5.670374419e-8 W/(m2 K4)
    F_view = 1.0
    非线性 Stefan-Boltzmann 辐射
    输出侧一阶滞后 (只作用 T_top_observed_predicted)

选择理由 (透明模型选择决定, 非统计证明):
    - 绝对 Strategy G RMSE 最优为 k=0.0525/cp=1000/tau=9.5 (RMSE 0.7869 C),
      但 cp=1000 是局部细化网格下界;
    - 冻结候选 k=0.055/cp=1200/tau=8.5 (RMSE ~0.8892 C):
        * k 保持于稳定最优盆地 (0.0525-0.0550);
        * cp 不在局部下界;
        * tau 接近已分辨滞后范围;
        * 拟合质量仍 <1 C;
        * 相对 RMSE 最优仅牺牲 ~0.10 C。
"""
from dataclasses import dataclass

import numpy as np

from thermal_model.core import convection_radiation_thermal_model as cr

FROZEN_K_W_MK = 0.055
FROZEN_CP_J_KGK = 1200.0
FROZEN_TAU_S = 8.5
FROZEN_RHO_COC = 1020.0


@dataclass(frozen=True)
class FrozenStrategyGCandidate:
    """冻结跨协议预测候选 (不可变; 防止意外修改)。"""
    k_eff_W_mK: float = FROZEN_K_W_MK
    cp_eff_J_kgK: float = FROZEN_CP_J_KGK
    tau_lag_s: float = FROZEN_TAU_S
    rho_COC_kg_m3: float = FROZEN_RHO_COC
    h_conv_W_m2K: float = cr.H_CONV_STRATEGY_E_W_M2K
    emissivity: float = cr.EMISSIVITY_STRATEGY_E
    sigma_SB_W_m2K4: float = cr.SIGMA_SB_W_M2_K4
    view_factor: float = cr.VIEW_FACTOR_STRATEGY_E

    def __post_init__(self) -> None:
        if not (self.k_eff_W_mK > 0 and np.isfinite(self.k_eff_W_mK)):
            raise ValueError(f"k 必须 >0 且有限: {self.k_eff_W_mK!r}")
        if not (self.cp_eff_J_kgK > 0 and np.isfinite(self.cp_eff_J_kgK)):
            raise ValueError(f"cp 必须 >0 且有限: {self.cp_eff_J_kgK!r}")
        if not (self.tau_lag_s >= 0 and np.isfinite(self.tau_lag_s)):
            raise ValueError(f"tau 必须 >=0 且有限: {self.tau_lag_s!r}")
        if not (self.rho_COC_kg_m3 > 0 and np.isfinite(self.rho_COC_kg_m3)):
            raise ValueError(f"rho 必须 >0 且有限: {self.rho_COC_kg_m3!r}")

    # ---- 派生 (不硬编码) ----
    @property
    def alpha_eff_m2_s(self) -> float:
        return self.k_eff_W_mK / (self.rho_COC_kg_m3 * self.cp_eff_J_kgK)

    @property
    def effusivity(self) -> float:
        return float(np.sqrt(
            self.k_eff_W_mK * self.rho_COC_kg_m3 * self.cp_eff_J_kgK))

    @property
    def rth_bottom_area_m2K_W(self) -> float:
        return 180e-6 / self.k_eff_W_mK


# 模块级单例 (只读使用)
FROZEN_STRATEGY_G_CANDIDATE = FrozenStrategyGCandidate()

# 权威 72C 校准回归值 (从存储的 Strategy G 扫描行解析)
STRATEGY_G_STORED_RMSE_72C = 0.8891597125869538  # k=0.055, cp=1200, tau=8.5


def candidate_dict() -> dict:
    """返回冻结候选的纯 dict 描述 (元数据用)。"""
    c = FROZEN_STRATEGY_G_CANDIDATE
    return {
        "id": "strategy_G_conservative_cross_protocol_v1",
        "status": ("FROZEN FOR CROSS-PROTOCOL PREDICTION / "
                   "NOT FINAL VALIDATED NOMINAL MODEL"),
        "k_eff_W_mK": c.k_eff_W_mK,
        "cp_eff_J_kgK": c.cp_eff_J_kgK,
        "tau_lag_s": c.tau_lag_s,
        "rho_COC_kg_m3": c.rho_COC_kg_m3,
        "alpha_eff_m2_s": c.alpha_eff_m2_s,
        "effusivity": c.effusivity,
        "rth_bottom_area_m2K_W": c.rth_bottom_area_m2K_W,
        "h_conv_W_m2K": c.h_conv_W_m2K,
        "emissivity": c.emissivity,
        "sigma_SB_W_m2K4": c.sigma_SB_W_m2K4,
        "view_factor": c.view_factor,
        "stored_72C_RMSE_C": STRATEGY_G_STORED_RMSE_72C,
    }
