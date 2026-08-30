#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THERMAL MODEL V2 CANDIDATE — FC-70 + no-PDMS insulated geometry
================================================================

候选模型配置 (NOT final / NOT frozen)。该模块是 V2 候选的唯一权威定义来源,
包含两项科学变更:

  CHANGE A — 材料修正:
      通用矿物油 -> 3M Fluorinert FC-70 (Sigma-Aldrich F9880)。
      室温代表性常数 (基于 3M 官方技术数据, 常数物性, 无温度依赖):
          k_FC70   = 0.070 W/(m K)
          rho_FC70 = 1940 kg/m^3
          cp_FC70  = 1050 J/(kg K)

  CHANGE B — 几何修正 (仅绝缘几何):
      移除 PDMS 层。绝缘堆栈变为:
          Bottom COC -> aqueous sample -> FC-70 -> Top COC
          -> sealed air -> external convection/radiation boundary
      (薄聚合物密封层在简化模型中有意忽略。)

绝不修改:
    - FINAL_FROZEN_THERMAL_MODEL_V1 (V1 冻结模型);
    - heat_model.DEFAULT_MATERIALS / LEGACY_INSULATED_LAYERS /
      BARE_TOP_COC_LAYERS (V1 权威层叠/材料);
    - 样品厚度 / FC-70 层厚度 / COC 几何 / 水物性 / 顶部边界参数 /
      样品加权。

本模块不修改任何 V1 源文件; FC-70 常量在此唯一定义, 供所有 V2 工作流
统一 import, 避免跨工作流重复定义。

科学身份命名: 材料标识为 FC-70 (内部键名 "FC70", 展示名 "FC-70")。
通用内部变量名 (如 oil) 不在此模块保留, V2 层叠直接引用 "FC70" 材料。
"""
from dataclasses import dataclass
from typing import Dict, List

from thermal_model.core import heat_model
from thermal_model.core.heat_model import Layer, Material

MODEL_ID = "THERMAL_MODEL_V2_CANDIDATE"
STATUS = "CANDIDATE"          # 未冻结 / 未晋升为 final
DATE_CREATED = "2026-08-29"

# ------------------------------------------------------------
# FC-70 材料 (唯一权威定义; 常数物性, 无温度依赖)
# ------------------------------------------------------------
FC70_MATERIAL_NAME = "FC70"     # 材料库键/名 (validate_materials 要求键==名)
FC70_K_W_MK = 0.070          # W/(m K)
FC70_RHO_KG_M3 = 1940.0      # kg/m^3
FC70_CP_J_KGK = 1050.0       # J/(kg K)

# 官方制造商相关性 (仅供文档记录; 本候选不使用温度依赖物性):
#   cp(T)  = 1014 + 1.554 T        [J/(kg K)]
#   k(T)   = 0.07 - 0.00001 T      [W/(m K)]
#   rho(T) = 1984 - 1.93 T         [kg/m^3]   (T in Celsius)
# 25 °C 近似: k=0.06975, rho=1935.75, cp=1052.85 -> 取整 0.070/1940/1050。
FC70_MANUFACTURER_CORRELATIONS = {
    "cp_J_kgK(T_C)": "1014 + 1.554*T",
    "k_W_mK(T_C)": "0.07 - 0.00001*T",
    "rho_kg_m3(T_C)": "1984 - 1.93*T",
    "note": ("documentation only; the V2 candidate uses constant "
             "room-temperature representative properties 0.070/1940/1050"),
}

# COC 密度 (与既有模型一致; 校准中唯一拟合 k_eff/cp_eff/tau_top, rho 固定)
RHO_COC_KG_M3 = 1020.0


def fc70_material() -> Material:
    """返回 FC-70 Material (不可变语义; 每次返回新实例)。"""
    return Material(name=FC70_MATERIAL_NAME,
                    k_W_mK=FC70_K_W_MK,
                    rho_kg_m3=FC70_RHO_KG_M3,
                    cp_J_kgK=FC70_CP_J_KGK)


def make_v2_materials(k_eff_W_mK, cp_eff_J_kgK,
                      rho_COC_kg_m3=RHO_COC_KG_M3) -> Dict[str, Material]:
    """构造 V2 候选材料库 (独立副本, 不修改 heat_model.DEFAULT_MATERIALS)。

    - 从默认材料库副本开始;
    - 用校准有效参数替换 COC (k_eff / cp_eff / rho; Bottom 与 Top COC 同引用);
    - 用 FC-70 替换通用矿物油 (删除 "Oil" 键, 新增 "FC70" 键);
    - Water / Air / PDMS 等其余材料逐位不变。
    """
    mats = heat_model.copy_default_materials()
    coc = mats["COC"]
    mats["COC"] = Material(name=coc.name,
                           k_W_mK=float(k_eff_W_mK),
                           rho_kg_m3=float(rho_COC_kg_m3),
                           cp_J_kgK=float(cp_eff_J_kgK))
    mats.pop("Oil", None)
    mats["FC70"] = fc70_material()
    return mats


# ------------------------------------------------------------
# V2 裸顶层叠 (标定 / 外部验证) —— 与 V1 裸顶结构一致, 仅 oil -> FC-70
# ------------------------------------------------------------
# 结构 (自下而上): Bottom COC (180 um) -> Sample (20 um) -> FC-70 (50 um)
#                  -> Top COC (600 um, 外表面暴露于环境)
# 总厚度 850 um; 无 Air / 无 PDMS; role="top_surface" 标记 Top COC 外表面。
V2_BARE_TOP_COC_LAYERS: List[Layer] = [
    Layer(name="Bottom COC", material="COC", thickness_m=180e-6,
          dx_target_m=5e-6),
    Layer(name="PCR Sample", material="Water", thickness_m=20e-6,
          dx_target_m=5e-6, role="sample"),
    Layer(name="FC-70", material="FC70", thickness_m=50e-6, dx_target_m=5e-6),
    Layer(name="Top COC", material="COC", thickness_m=600e-6,
          dx_target_m=5e-6, role="top_surface"),
]

# ------------------------------------------------------------
# V2 绝缘层叠 (无 PDMS) —— 前向扩展
# ------------------------------------------------------------
# 结构 (自下而上): Bottom COC (180 um) -> Sample (20 um) -> FC-70 (50 um)
#                  -> Top COC (600 um) -> Air Gap (3000 um, 密封空气)
# 总厚度 3850 um (V1 绝缘 4050 um 减去 200 um PDMS)。
# 无 PDMS 层; 外部对流+辐射边界作用于最终外表面 (= 密封空气外表面)。
# role="top_surface" 不设在此基础定义 (由预测工具按需在本地副本添加)。
V2_INSULATED_NO_PDMS_LAYERS: List[Layer] = [
    Layer(name="Bottom COC", material="COC", thickness_m=180e-6,
          dx_target_m=5e-6),
    Layer(name="PCR Sample", material="Water", thickness_m=20e-6,
          dx_target_m=5e-6, role="sample"),
    Layer(name="FC-70", material="FC70", thickness_m=50e-6, dx_target_m=5e-6),
    Layer(name="Top COC", material="COC", thickness_m=600e-6,
          dx_target_m=5e-6),
    Layer(name="Air Gap", material="Air", thickness_m=3000e-6,
          dx_target_m=200e-6),
]

# 层叠预设注册 (供 CLI / 诊断使用)
V2_LAYER_STACK_PRESETS: Dict[str, List[Layer]] = {
    "v2-bare-top": V2_BARE_TOP_COC_LAYERS,
    "v2-insulated-no-pdms": V2_INSULATED_NO_PDMS_LAYERS,
}


# ------------------------------------------------------------
# 临时冻结的 V2 候选参数 (标定后回填; 未标定前为 None)
# ------------------------------------------------------------

@dataclass(frozen=True)
class ThermalModelV2Candidate:
    """V2 候选模型 (不可变)。仅在校准完成后实例化并临时冻结以供验证。"""
    model_id: str = MODEL_ID
    status: str = STATUS
    k_eff_W_mK: float = None          # 校准后回填
    cp_eff_J_kgK: float = None
    tau_top_s: float = None
    rho_COC_kg_m3: float = RHO_COC_KG_M3

    @property
    def fc70_k_W_mK(self) -> float:
        return FC70_K_W_MK

    @property
    def fc70_rho_kg_m3(self) -> float:
        return FC70_RHO_KG_M3

    @property
    def fc70_cp_J_kgK(self) -> float:
        return FC70_CP_J_KGK

    @property
    def alpha_eff_m2_s(self):
        if self.k_eff_W_mK is None or self.cp_eff_J_kgK is None:
            return None
        return self.k_eff_W_mK / (self.rho_COC_kg_m3 * self.cp_eff_J_kgK)


def v2_candidate_dict(k_eff, cp_eff, tau_top, rho=RHO_COC_KG_M3) -> dict:
    """V2 候选模型纯 dict 描述 (元数据 / 输出用)。"""
    return {
        "model_id": MODEL_ID,
        "status": STATUS,
        "k_eff_W_mK": k_eff,
        "cp_eff_J_kgK": cp_eff,
        "tau_top_s": tau_top,
        "rho_COC_kg_m3": rho,
        "fc70_k_W_mK": FC70_K_W_MK,
        "fc70_rho_kg_m3": FC70_RHO_KG_M3,
        "fc70_cp_J_kgK": FC70_CP_J_KGK,
        "material_change": "generic mineral oil -> Fluorinert FC-70",
        "insulated_geometry_change": "PDMS layer removed (no-PDMS insulated)",
        "bare_geometry": "BARE_TOP_COC_LAYERS structure unchanged "
                         "except oil -> FC-70",
    }
