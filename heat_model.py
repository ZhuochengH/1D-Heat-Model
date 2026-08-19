#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D 多层瞬态热传导模型的「材料库 / 层叠结构」配置层。

本模块把「材料是什么」与「材料在哪里使用」两个概念分离:

    User-defined materials (Material)
            ↓
    User-defined layer stack (Layer)
            ↓
    build_layer_stack(...)  ->  自动网格 / 属性数组
            ↓
    (外部) 现有 FDM 求解器
            ↓
    温度预测

设计原则
--------
- Material 描述一种材料的本征 / 有效热物性 (k, rho, cp), 与几何无关;
- Layer 描述一层几何 (厚度) + 材料引用 + 网格分辨率 + 可选语义角色 (role);
- 同一 Material 可被多个 Layer 引用 —— 为未来 k_eff / cp_eff 网格搜索铺路:
  修改 materials["COC"].k_W_mK 后, 所有引用 "COC" 的层自动使用新值;
- build_layer_stack 是「配置 -> 数值数组」的唯一转换入口 (单一事实来源);
- 本版本只支持: 1D、多层、瞬态导热、各层内均匀、各向同性、常数 k/rho/cp。

明确不包含 (本版本):
    - k(T) / cp(T) / rho(T) 温度依赖
    - 各向异性
    - 相变
    - 接触热阻 / 界面额外热阻
    - 内部热源 / 辐射
    - 2D/3D 几何 / 自适应网格
    - 外部 YAML/JSON 材料库
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

import numpy as np


# =============================================================
# 数据模型
# =============================================================

@dataclass
class Material:
    """一种材料的本征/有效热物性定义 (常数, 各向同性)。"""
    name: str
    k_W_mK: float      # 导热系数  W/(m·K)
    rho_kg_m3: float   # 密度      kg/m^3
    cp_J_kgK: float    # 比热容    J/(kg·K)


@dataclass
class Layer:
    """一层几何定义 + 材料引用 + 网格分辨率 + 可选语义角色。

    cells 与 dx_target_m 二者只能提供一个:
      - cells        : 显式单元数 (优先级高于 dx_target_m, 二者都填则报错);
      - dx_target_m  : 目标网格步长, 单元数 = max(1, round(span / dx_target_m))
                       (复现原始模型的 make_layer 行为)。
    role            : 语义观察用途, 例如 "sample" (样品观测) 或
                      "top_surface" (顶表面观测, 对应实测外表面位置)。
    """
    name: str
    material: str
    thickness_m: float
    cells: Optional[int] = None
    dx_target_m: Optional[float] = None
    role: Optional[str] = None


@dataclass
class LayerStack:
    """由材料库 + 层叠结构构造的数值网格/属性数组 (单一事实来源)。"""
    x: np.ndarray                 # 节点坐标 (m), 长度 Nx
    h: np.ndarray                 # 节点间距 (m), 长度 Nx-1
    k: np.ndarray                 # 每节点导热系数 (W/m·K) [诊断用, 界面节点=左侧材料]
    rho: np.ndarray               # 每节点密度 (kg/m^3) [诊断用]
    cp: np.ndarray                # 每节点比热容 (J/kg·K) [诊断用]
    k_face: np.ndarray            # 区间 [x_i, x_{i+1}] 物理导热系数 (W/m·K), 长度 Nx-1
                                  # [瞬态能量平衡的权威导热量]
    rho_cp: np.ndarray            # 节点控制体积等效体积热容 (J/(m^3·K)), 长度 Nx
                                  # [瞬态能量平衡的权威热容]
    Nx: int
    idx_sample: np.ndarray        # 样品层观测节点索引 (可能为空)
    sample_weights: np.ndarray    # 样品层空间平均权重 (长度 Nx, sum=1; 空层为全零)
    idx_top_surface: np.ndarray   # role="top_surface" 层外表面节点索引 (可能为空)
    boundaries: np.ndarray        # 层界面位置, 长度 N_layers+1
    node_layer_index: np.ndarray  # 每节点所属层索引, 长度 Nx
    layer_names: List[str]
    layer_thicknesses: List[float]
    sample_layer_index: Optional[int]
    top_surface_layer_index: Optional[int]


# =============================================================
# 默认配置 —— 两套层叠预设 + 历史默认别名 (历史数值必须与原脚本逐位一致)
# =============================================================
#
# 材料库 (材料"可用") 与层叠结构 (材料"出现在当前实验里") 分离:
#   - DEFAULT_MATERIALS       : 通用材料库, 含 Air / PDMS (仍可用, 不删除);
#   - LEGACY_INSULATED_LAYERS : 历史 6 层绝缘配置 (含 Air + PDMS 盖帽),
#                               总厚度 4050 um, 对应旧论文几何;
#   - BARE_TOP_COC_LAYERS     : 当前实测实验的裸顶配置 (无 Air / PDMS),
#                               总厚度 850 um, Top COC 外表面直接暴露于环境;
#   - DEFAULT_LAYERS          : LEGACY_INSULATED_LAYERS 的向后兼容别名
#                               (历史脚本/测试仍引用它; 含义 = 旧绝缘几何,
#                               不是裸顶实验几何)。

DEFAULT_MATERIALS: Dict[str, Material] = {
    "COC": Material(name="COC", k_W_mK=0.13, rho_kg_m3=1020.0, cp_J_kgK=1800.0),
    "Water": Material(name="Water", k_W_mK=0.60, rho_kg_m3=1000.0, cp_J_kgK=4180.0),
    "Oil": Material(name="Oil", k_W_mK=0.142, rho_kg_m3=876.0, cp_J_kgK=1962.0),
    "Air": Material(name="Air", k_W_mK=0.0257, rho_kg_m3=1.204, cp_J_kgK=1005.0),
    "PDMS": Material(name="PDMS", k_W_mK=0.15, rho_kg_m3=970.0, cp_J_kgK=1460.0),
}

# -------------------------------------------------------------
# 历史 (绝缘) 层叠预设 —— 与原始脚本完全一致, 属性/网格勿改
# -------------------------------------------------------------
# 层叠顺序 (自下而上) 与原始脚本完全一致:
#   Bottom COC (180 um) -> Water sample (20 um) -> Oil (50 um)
#   -> Top COC (600 um) -> Air gap (3000 um) -> Cap PDMS (200 um)
# 总厚度: 180 + 20 + 50 + 600 + 3000 + 200 = 4050 um。
# 网格分辨率与原始脚本一致:
#   关注区 (COC/sample/oil) 5 um; Air gap 200 um; PDMS 50 um。
LEGACY_INSULATED_LAYERS: List[Layer] = [
    Layer(name="Bottom COC", material="COC", thickness_m=180e-6, dx_target_m=5e-6),
    Layer(name="PCR Sample", material="Water", thickness_m=20e-6, dx_target_m=5e-6,
          role="sample"),
    Layer(name="Mineral Oil", material="Oil", thickness_m=50e-6, dx_target_m=5e-6),
    Layer(name="Top COC", material="COC", thickness_m=600e-6, dx_target_m=5e-6),
    Layer(name="Air Gap", material="Air", thickness_m=3000e-6, dx_target_m=200e-6),
    Layer(name="Cap PDMS", material="PDMS", thickness_m=200e-6, dx_target_m=50e-6),
]

# 向后兼容别名: 历史脚本/测试引用的 "默认" 即旧绝缘配置 (同一对象)。
DEFAULT_LAYERS: List[Layer] = LEGACY_INSULATED_LAYERS

# -------------------------------------------------------------
# 实验 (裸顶) 层叠预设 —— 对应 extension 72°C.xls / T Avg 实测几何
# -------------------------------------------------------------
# 实际实验物理结构 (自下而上, 无任何绝缘层):
#   Bottom COC (180 um) -> PCR sample / water (20 um) -> Oil (50 um)
#   -> Top COC (600 um, 外表面直接暴露于环境空气)
# 总厚度: 180 + 20 + 50 + 600 = 850 um。
# 无 Air 层、无 PDMS 层: 环境空气只通过顶部 Robin 对流边界进入模型:
#   -k dT/dx = h_conv * (T_surface - T_air_ambient)   (x = 850 um)
# role="top_surface" 标记 Top COC 外表面观测位置
# (本预设下 = 模型最外节点 x[-1] = 850 um)。
BARE_TOP_COC_LAYERS: List[Layer] = [
    Layer(name="Bottom COC", material="COC", thickness_m=180e-6, dx_target_m=5e-6),
    Layer(name="PCR Sample", material="Water", thickness_m=20e-6, dx_target_m=5e-6,
          role="sample"),
    Layer(name="Mineral Oil", material="Oil", thickness_m=50e-6, dx_target_m=5e-6),
    Layer(name="Top COC", material="COC", thickness_m=600e-6, dx_target_m=5e-6,
          role="top_surface"),
]

# 层叠预设注册表 (供 CLI 名称解析, 例如 --layer-stack bare-top|legacy)。
LAYER_STACK_PRESETS: Dict[str, List[Layer]] = {
    "bare-top": BARE_TOP_COC_LAYERS,
    "legacy": LEGACY_INSULATED_LAYERS,
}


def copy_default_materials() -> Dict[str, Material]:
    """返回默认材料库的独立副本。

    未来 k_eff/cp_eff 网格搜索可这样在不改动全局常量的前提下调整:
        materials = copy_default_materials()
        materials["COC"].k_W_mK = candidate_k
    """
    return {
        name: Material(
            name=m.name, k_W_mK=m.k_W_mK,
            rho_kg_m3=m.rho_kg_m3, cp_J_kgK=m.cp_J_kgK,
        )
        for name, m in DEFAULT_MATERIALS.items()
    }


def copy_layers(layers: List[Layer]) -> List[Layer]:
    """返回任意层叠结构的独立副本 (可单独调整某层厚度/网格/角色)。"""
    return [
        Layer(
            name=layer.name, material=layer.material,
            thickness_m=layer.thickness_m,
            cells=layer.cells, dx_target_m=layer.dx_target_m, role=layer.role,
        )
        for layer in layers
    ]


def copy_default_layers() -> List[Layer]:
    """返回历史默认 (绝缘) 层叠结构 LEGACY_INSULATED_LAYERS 的独立副本。"""
    return copy_layers(DEFAULT_LAYERS)


def copy_bare_top_layers() -> List[Layer]:
    """返回实验用裸顶层叠结构 BARE_TOP_COC_LAYERS 的独立副本。"""
    return copy_layers(BARE_TOP_COC_LAYERS)


def resolve_layer_stack(name: str) -> List[Layer]:
    """按名称解析层叠预设 ('bare-top' | 'legacy'); 未知名称报错。

    返回的是预设常量本身 (只读); 需要可修改副本时用 copy_layers。
    """
    if name not in LAYER_STACK_PRESETS:
        raise ValueError(
            f"未知层叠预设 {name!r}; 可用: {sorted(LAYER_STACK_PRESETS)}"
        )
    return LAYER_STACK_PRESETS[name]


class LayerStackDescription(TypedDict):
    """describe_layer_stack 的返回类型 (供脚本诊断/元数据使用)。"""
    layer_count: int
    total_thickness_m: float
    boundaries_m: List[float]
    layer_names: List[str]
    layer_materials: List[str]
    layer_roles: List[Optional[str]]
    outer_surface_position_m: float
    outer_surface_material: str
    has_air_layer: bool
    has_pdms_layer: bool
    has_top_surface_role: bool
    top_surface_position_m: Optional[float]
    top_surface_material: Optional[str]
    sample_position_m: Optional[tuple[float, float]]
    sample_weights: List[float]
    sample_integral_width_m: float


def describe_layer_stack(materials: Dict[str, Material],
                         layers: List[Layer]) -> LayerStackDescription:
    """返回层叠结构的几何/观测描述 (仅诊断与元数据用, 不改变任何数值)。"""
    validate_materials(materials)
    validate_layers(layers, materials)
    mesh = build_layer_stack(materials, layers)
    si = mesh.sample_layer_index
    return {
        "layer_count": len(layers),
        "total_thickness_m": float(mesh.boundaries[-1]),
        "boundaries_m": mesh.boundaries.tolist(),
        "layer_names": list(mesh.layer_names),
        "layer_materials": [layers[i].material for i in range(len(layers))],
        "layer_roles": [layers[i].role for i in range(len(layers))],
        "outer_surface_position_m": float(mesh.x[-1]),
        "outer_surface_material": layers[int(mesh.node_layer_index[-1])].material,
        "has_air_layer": any(l.material == "Air" for l in layers),
        "has_pdms_layer": any(l.material == "PDMS" for l in layers),
        "has_top_surface_role": mesh.idx_top_surface.size > 0,
        "top_surface_position_m": (
            float(mesh.x[mesh.idx_top_surface[0]])
            if mesh.idx_top_surface.size else None
        ),
        "top_surface_material": (
            layers[int(mesh.top_surface_layer_index)].material
            if mesh.top_surface_layer_index is not None else None
        ),
        "sample_position_m": (
            (float(mesh.boundaries[si]), float(mesh.boundaries[si + 1]))
            if si is not None else None
        ),
        "sample_weights": mesh.sample_weights.tolist(),
        "sample_integral_width_m": (
            float(mesh.boundaries[si + 1] - mesh.boundaries[si])
            if si is not None else 0.0
        ),
    }


# =============================================================
# 校验
# =============================================================

def _require_positive_scalar(field_name: str, value) -> None:
    """要求 value 是有穷正实数; 非数值报 TypeError, 非正/非有穷报 ValueError。"""
    if isinstance(value, (bool, complex)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field_name} 必须为实数数值, 收到: {value!r}")
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{field_name} 必须是有穷正数, 收到: {value!r}")


def validate_material(mat: Material) -> None:
    """校验单个 Material: 名称非空, k/rho/cp 均为有穷正数。"""
    if not isinstance(mat, Material):
        raise TypeError(f"期望 Material 实例, 收到: {type(mat).__name__}")
    if not isinstance(mat.name, str) or not mat.name.strip():
        raise ValueError("Material.name 必须是非空字符串。")
    _require_positive_scalar("k_W_mK", mat.k_W_mK)
    _require_positive_scalar("rho_kg_m3", mat.rho_kg_m3)
    _require_positive_scalar("cp_J_kgK", mat.cp_J_kgK)


def validate_materials(materials: Dict[str, Material]) -> None:
    """校验材料库: 非空, 每个条目合法, 键与 Material.name 一致。"""
    if not materials:
        raise ValueError("材料库不能为空。")
    for key, mat in materials.items():
        validate_material(mat)
        if key != mat.name:
            raise ValueError(
                f"材料库键 {key!r} 与 Material.name {mat.name!r} 不一致。"
            )


def _validate_cells(layer: Layer) -> None:
    c = layer.cells
    if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
        raise TypeError(f"层 {layer.name!r} 的 cells 必须是整数, 收到: {c!r}")
    if int(c) < 1:
        raise ValueError(f"层 {layer.name!r} 的 cells 必须 >= 1, 收到: {c}")


def validate_layers(layers: List[Layer], materials: Dict[str, Material]) -> None:
    """校验层叠结构: 非空, 材料引用存在, 厚度>0, 网格分辨率合法。"""
    if not layers:
        raise ValueError("层叠结构不能为空。")
    for layer in layers:
        if not isinstance(layer, Layer):
            raise TypeError(f"期望 Layer 实例, 收到: {type(layer).__name__}")
        if not isinstance(layer.name, str) or not layer.name.strip():
            raise ValueError("Layer.name 必须是非空字符串。")
        if layer.material not in materials:
            raise ValueError(
                f"层 {layer.name!r} 引用了未知材料 {layer.material!r}; "
                f"可用材料: {sorted(materials)}"
            )
        _require_positive_scalar("thickness_m", layer.thickness_m)
        if layer.cells is not None and layer.dx_target_m is not None:
            raise ValueError(
                f"层 {layer.name!r} 不能同时指定 cells 与 dx_target_m (只能其一)。"
            )
        if layer.cells is not None:
            _validate_cells(layer)
        elif layer.dx_target_m is not None:
            _require_positive_scalar("dx_target_m", layer.dx_target_m)
        else:
            raise ValueError(
                f"层 {layer.name!r} 必须指定 cells 或 dx_target_m 之一。"
            )


# =============================================================
# 自动网格 / 属性数组构造
# =============================================================

# 界面节点归属容差 (与原始脚本的 +1e-9 逐位一致)
_NODE_TOL = 1e-9


def _resolve_cells(layer: Layer, span: float) -> int:
    """确定一层的单元数; span = boundaries[i+1] - boundaries[i]。

    用 span (而非 layer.thickness_m) 计算, 以与原始 make_layer 的
    int(round((x1 - x0) / dx)) 在浮点上逐位一致。
    """
    if layer.cells is not None:
        return int(layer.cells)
    if layer.dx_target_m is None:
        # build_layer_stack 已先经 validate_layers 校验 (二者必居其一);
        # 此处显式收窄, 让 Pylance 得到 dx_target_m: float, 行为保持不变。
        raise ValueError(
            f"层 {layer.name!r} 必须指定 cells 或 dx_target_m 之一。"
        )
    dx = layer.dx_target_m
    return max(1, int(round(span / dx)))


def compute_sample_weights(x: np.ndarray,
                           x_left: float,
                           x_right: float) -> np.ndarray:
    """节点控制体积与样品层区间重叠宽度 (长度 Nx, 未归一化)。

    每个节点的控制体积:
      - 内部节点: [ (x[i-1]+x[i])/2, (x[i]+x[i+1])/2 ];
      - 域边界节点: 使用实际域边界 (节点 0 左面 = x[0], 末节点右面 = x[-1])。
    重叠宽度 = length(控制体积 ∩ [x_left, x_right])。

    性质: 控制体积恰为全域 [x[0], x[-1]] 的连续剖分, 因此重叠宽度之和
    = x_right - x_left (样品层厚度, 浮点容差内)。
    """
    x = np.asarray(x, dtype=float)
    Nx = len(x)
    cv_left = np.empty(Nx)
    cv_right = np.empty(Nx)
    cv_left[0] = x[0]
    cv_right[-1] = x[-1]
    if Nx > 1:
        mid = (x[:-1] + x[1:]) / 2.0
        cv_left[1:] = mid
        cv_right[:-1] = mid
    overlap = np.maximum(
        0.0,
        np.minimum(cv_right, float(x_right))
        - np.maximum(cv_left, float(x_left)),
    )
    return overlap


def _layer_property_tables(materials: Dict[str, Material],
                           layers: List[Layer]):
    """每层一个导热系数与体积热容的查找表 (长度 n_layers)。"""
    layer_k = np.empty(len(layers), dtype=float)
    layer_rho_cp = np.empty(len(layers), dtype=float)
    for i, layer in enumerate(layers):
        mat = materials[layer.material]
        layer_k[i] = mat.k_W_mK
        layer_rho_cp[i] = mat.rho_kg_m3 * mat.cp_J_kgK
    return layer_k, layer_rho_cp


def build_face_conductivity(materials: Dict[str, Material],
                            layers: List[Layer],
                            node_layer: np.ndarray) -> np.ndarray:
    """区间 [x_i, x_{i+1}] 的物理导热系数 k_face (长度 Nx-1)。

    修正版有限体积: 当前网格把材料界面精确放在节点上, 因此每个节点区间整体
    位于单一材料内, 其所属层 = 右端节点所属层 (界面节点归属左侧层)。区间导热
    系数就是该材料的 k, 不应把两个端点节点的材料标签做调和平均。
    """
    layer_k, _ = _layer_property_tables(materials, layers)
    return layer_k[np.asarray(node_layer)[1:]]


def build_volumetric_capacity(materials: Dict[str, Material],
                              layers: List[Layer],
                              node_layer: np.ndarray,
                              h: np.ndarray) -> np.ndarray:
    """节点控制体积的等效体积热容 rho_cp (J/(m^3·K), 长度 Nx)。

    节点 j 的控制体积为 [x_j - h_L/2, x_j + h_R/2]:
      - 普通节点: 左右半区间同材料 -> rho_cp = 材料 rho*cp;
      - 材料界面节点: 左右半区间分属两种材料 -> 按半宽体积加权
            rho_cp = (rhoCp_L*h_L + rhoCp_R*h_R) / (h_L + h_R)。
    """
    _, layer_rho_cp = _layer_property_tables(materials, layers)
    nl = np.asarray(node_layer)
    h = np.asarray(h)
    rc = np.empty(len(nl), dtype=float)
    rc[0] = layer_rho_cp[nl[0]]
    rc[-1] = layer_rho_cp[nl[-1]]
    rc[1:-1] = (
        layer_rho_cp[nl[1:-1]] * h[:-1]
        + layer_rho_cp[nl[2:]] * h[1:]
    ) / (h[:-1] + h[1:])
    return rc


def build_layer_stack(materials: Dict[str, Material],
                      layers: List[Layer]) -> LayerStack:
    """把「材料库 + 层叠结构」转换成数值网格与属性数组 (单一事实来源)。

    关键行为:
      - 界面位置 boundaries 用顺序累加得到 (与原始 x_*_end 变量一致);
      - 每层网格 linspace(x0, x1, n_cells+1), 层间界面节点经 np.unique 共享;
      - 节点 -> 材料 用 "xi <= layer_end + 1e-9" 规则, 界面节点归属左侧层;
      - k/rho/cp 节点数组仅作诊断/解释用 (界面节点=左侧材料);
      - k_face (区间导热, 长度 Nx-1) 与 rho_cp (节点体积热容, 长度 Nx) 为
        修正版有限体积的权威数值量, 见 build_face_conductivity /
        build_volumetric_capacity;
      - 样品层由 role="sample" 识别, 观测节点 = (x > left) & (x <= right+1e-9)。
    """
    validate_materials(materials)
    validate_layers(layers, materials)

    n_layers = len(layers)
    boundaries = np.empty(n_layers + 1, dtype=float)
    boundaries[0] = 0.0
    for i, layer in enumerate(layers):
        boundaries[i + 1] = boundaries[i] + layer.thickness_m

    # 各层网格节点 (非均匀网格)
    pieces = []
    for i, layer in enumerate(layers):
        span = boundaries[i + 1] - boundaries[i]
        n_cells = _resolve_cells(layer, span)
        pieces.append(np.linspace(boundaries[i], boundaries[i + 1], n_cells + 1))
    x = np.unique(np.concatenate(pieces))
    h = np.diff(x)
    Nx = len(x)

    # 节点 -> 材料属性 (界面节点归属左侧层; k/rho/cp 仅作诊断/解释用)
    rho = np.zeros(Nx)
    k = np.zeros(Nx)
    cp = np.zeros(Nx)
    node_layer = np.zeros(Nx, dtype=int)
    for i, xi in enumerate(x):
        for li in range(n_layers):
            if xi <= boundaries[li + 1] + _NODE_TOL:
                mat = materials[layers[li].material]
                rho[i] = mat.rho_kg_m3
                k[i] = mat.k_W_mK
                cp[i] = mat.cp_J_kgK
                node_layer[i] = li
                break

    # 修正版有限体积属性 (单一事实来源):
    #   k_face[i] : 区间 [x_i, x_{i+1}] 的物理导热系数 (长度 Nx-1);
    #               界面恰在节点上 -> 区间整体位于单一材料 (右端节点所属层);
    #   rho_cp[i] : 节点控制体积等效体积热容 (长度 Nx);
    #               界面节点按左右半宽体积加权, 普通节点 = 材料 rho*cp。
    k_face = build_face_conductivity(materials, layers, node_layer)
    rho_cp = build_volumetric_capacity(materials, layers, node_layer, h)

    # 样品层识别 (role="sample")
    sample_indices = [i for i, layer in enumerate(layers) if layer.role == "sample"]
    if len(sample_indices) > 1:
        raise ValueError(
            f"存在多个 role='sample' 的层 (索引 {sample_indices}); "
            "当前模型仅支持一个样品层。"
        )
    if sample_indices:
        si = sample_indices[0]
        x_left = float(boundaries[si])
        x_right = float(boundaries[si + 1])
        idx_sample = np.where(
            (x > x_left) & (x <= x_right + _NODE_TOL)
        )[0]
        # 样品层空间平均: 节点控制体积重叠宽度归一化 (sum = 1)
        raw_weights = compute_sample_weights(x, x_left, x_right)
        w_sum = float(np.sum(raw_weights))
        if w_sum <= 0.0:
            raise RuntimeError(
                f"role='sample' 层 {layers[si].name!r} 的样品层区间 "
                f"[{x_left:.6e}, {x_right:.6e}] m 与网格控制体积无重叠。"
            )
        sample_weights = raw_weights / w_sum
    else:
        si = None
        idx_sample = np.array([], dtype=int)
        sample_weights = np.zeros(Nx, dtype=float)

    # 顶表面观测层识别 (role="top_surface")
    top_indices = [i for i, layer in enumerate(layers)
                   if layer.role == "top_surface"]
    if len(top_indices) > 1:
        raise ValueError(
            f"存在多个 role='top_surface' 的层 (索引 {top_indices}); "
            "当前模型仅支持一个顶表面观测层。"
        )
    if top_indices:
        ti = top_indices[0]
        # 该层外边界 (顶面) 处的节点; 界面节点经 np.unique 共享, 必存在唯一节点。
        idx_top_surface = np.where(
            np.abs(x - boundaries[ti + 1]) <= _NODE_TOL
        )[0]
        if idx_top_surface.size != 1:
            raise RuntimeError(
                f"role='top_surface' 层 {layers[ti].name!r} 的外边界 "
                f"x={boundaries[ti + 1]:.6e} m 未找到唯一节点。"
            )
    else:
        ti = None
        idx_top_surface = np.array([], dtype=int)

    return LayerStack(
        x=x,
        h=h,
        k=k,
        rho=rho,
        cp=cp,
        k_face=k_face,
        rho_cp=rho_cp,
        Nx=Nx,
        idx_sample=idx_sample,
        sample_weights=sample_weights,
        idx_top_surface=idx_top_surface,
        boundaries=boundaries,
        node_layer_index=node_layer,
        layer_names=[layer.name for layer in layers],
        layer_thicknesses=[layer.thickness_m for layer in layers],
        sample_layer_index=si,
        top_surface_layer_index=ti,
    )


# =============================================================
# FDM 求解器 (唯一权威实现) —— 修正版有限体积
# =============================================================
# 底部边界为显式的 Dirichlet 温度迹线, 与来源无关:
#     T[0] = bottom_temperature_C(t)
# 求解器不包含任何 a/b 校准、tau 动态滤波、或 Excel 协议解析。
#
# 界面有限体积修正 (对比旧版):
#   旧: 界面恰在节点上, 但区间导热系数取两端节点材料标签的调和平均
#       (k_{i+1/2} = 2 k_i k_{i+1}/(k_i+k_{i+1})),
#       界面节点热容整段取左侧材料 rho*cp;
#   新: 每个节点区间 [x_i, x_{i+1}] 物理上位于单一材料内, 区间导热系数
#       = 该材料 k (k_face, 见 build_face_conductivity);
#       界面节点控制体积横跨两材料, 热容按左右半宽体积加权
#       (rho_cp, 见 build_volumetric_capacity)。
# 其余 (底部 Dirichlet / 顶部 Robin / 显式 Euler / 样品观测) 不变。

def _compute_dt(mesh: LayerStack) -> float:
    """从网格计算显式 FDM 的稳定时间步长 dt。

    与 run_simulation 使用相同的修正版有限体积系数 (k_face / rho_cp):
    逐节点稳定性上限 dt_i ≤ rhoCp_i·(h_m+h_p) / [2·(k_p/h_p + k_m/h_m)]。
    """
    k_face = mesh.k_face
    h = mesh.h
    rho_cp = mesh.rho_cp
    h_m = h[:-1]       # x[i] - x[i-1]
    h_p = h[1:]        # x[i+1] - x[i]
    k_m = k_face[:-1]  # k_{i-1/2}
    k_p = k_face[1:]   # k_{i+1/2}
    rc_int = rho_cp[1:-1]
    # 逐节点稳定性上限: dt_i ≤ rhoCp_i·(h_m+h_p) / [2·(k_p/h_p + k_m/h_m)]
    dt_stable = rc_int * (h_m + h_p) / (2 * (k_p / h_p + k_m / h_m))
    return float(np.min(dt_stable) * 0.9)


def compute_stable_dt(materials: Dict[str, Material],
                      layers: List[Layer]):
    """返回 (mesh, dt): 显式 FDM 的稳定时间步长。

    供需要在 FDM 时间轴上做边界预处理 (例如校准版动态表面模型) 的调用方,
    在调用 run_simulation 前先取得 dt 以构建时间网格。此 dt 与
    run_simulation 内部使用的一致 (逐位相同)。
    """
    mesh = build_layer_stack(materials, layers)
    return mesh, _compute_dt(mesh)


def _validate_boundary_trace(time_s, bottom_temperature_C):
    """校验底部边界时间序列, 返回 (time_s, bottom) 的 float64 一维数组。"""
    time_s = np.asarray(time_s, dtype=float)
    bottom_temperature_C = np.asarray(bottom_temperature_C, dtype=float)
    if time_s.ndim != 1 or bottom_temperature_C.ndim != 1:
        raise ValueError("time_s 与 bottom_temperature_C 必须是一维数组。")
    if len(time_s) != len(bottom_temperature_C):
        raise ValueError(
            f"time_s (长度 {len(time_s)}) 与 bottom_temperature_C "
            f"(长度 {len(bottom_temperature_C)}) 长度必须一致。"
        )
    if len(time_s) < 2:
        raise ValueError("边界时间序列至少需要 2 个点。")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("time_s 必须为有限数值。")
    if not np.all(np.isfinite(bottom_temperature_C)):
        raise ValueError("bottom_temperature_C 必须为有限数值。")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s 必须严格单调递增。")
    return time_s, bottom_temperature_C


def run_simulation(time_s, bottom_temperature_C, materials, layers,
                   h_conv=5.0, T_air_ambient=25.0, save_dt=0.1,
                   T_initial_C=25.0):
    """运行 1D 多层瞬态 FDM 求解 (唯一权威数值实现)。

    底部边界为显式的 Dirichlet 温度迹线, 与来源无关:

        T[0](t) = bottom_temperature_C(t)   (插值到 FDM 时间网格)

    调用方负责准备 bottom_temperature_C (可以是实测 T_internal、校准后的
    T_surface_dynamic、或任意合成协议); 求解器不关心其来源, 也不会在内部
    施加任何 a/b 稳态校准或 tau 动态滤波。

    参数:
        time_s               : 边界时间轴 (秒), 严格单调递增;
        bottom_temperature_C : 边界温度迹线 (°C), 与 time_s 等长;
        materials, layers    : 材料库 / 层叠结构 (见 build_layer_stack);
        h_conv               : 顶部自然对流换热系数 W/(m²·K);
        T_air_ambient        : 顶部环境温度 (°C);
        save_dt              : 输出下采样间隔 (秒);
        T_initial_C          : 初始均匀温度场 (°C)。

    返回 dict:
        time_fdm                : FDM 时间轴 (全分辨率);
        bottom_temperature_fdm  : 插值到 FDM 时间网格的底部边界;
        t_array                 : 下采样时间轴;
        T_bottom_arr            : 下采样底部边界温度;
        T_sample_arr            : 下采样样品层空间平均温度
                                  (role='sample' 层, 节点控制体积重叠加权,
                                  sum(sample_weights)=1);
        T_outer_surface_arr     : 下采样最外表面温度 = T[-1]
                                  (模型域最外节点, 不绑定具体材料/坐标);
        T_top_arr               : T_outer_surface_arr 的向后兼容别名
                                  (含义一致 = 最外表面温度, 不是固定坐标);
        T_top_surface_arr       : 下采样 role='top_surface' 层外表面温度
                                  (实验观测位置; 无该角色层时为空数组);
        T_final                 : 最终温度场 (长度 Nx, 稳态/诊断用);
        dt / Nt / Nx / save_interval / mesh。
    """
    time_s, bottom_temperature_C = _validate_boundary_trace(
        time_s, bottom_temperature_C
    )

    mesh = build_layer_stack(materials, layers)
    if mesh.idx_sample.size == 0:
        raise ValueError(
            "层叠结构中没有 role='sample' 的层, 无法提取样品温度。"
        )

    x = mesh.x
    h = mesh.h
    Nx = mesh.Nx
    idx_sample = mesh.idx_sample
    idx_top_surface = mesh.idx_top_surface
    has_top_surface_obs = idx_top_surface.size > 0

    dt = _compute_dt(mesh)

    t_total = float(time_s[-1] - time_s[0])
    Nt = int(t_total / dt) + 1
    time_fdm = np.linspace(0.0, t_total, Nt)   # 协议起点映射为 t=0

    # 底部边界评估到 FDM 时间网格 (np.interp; 若输入已在 FDM 网格上则恒等)
    bottom_fdm = np.interp(time_fdm, time_s, bottom_temperature_C)

    # 预计算三对角更新系数（完全向量化，时间循环内仅三次向量乘加）
    # T_new[i] = c_c[i]*T[i] + c_m[i]*T[i-1] + c_p[i]*T[i+1]
    # 修正版有限体积:
    #   k_face[i] = 区间 [x_i, x_{i+1}] 物理导热系数 (界面在节点上,
    #               区间整体位于单一材料 -> k_face = 该材料 k,
    #               不用端点调和平均);
    #   rho_cp[i] = 节点控制体积等效体积热容 (界面节点左右半宽体积加权)。
    k_face = mesh.k_face
    rho_cp = mesh.rho_cp
    h_m = h[:-1]
    h_p = h[1:]
    k_m = k_face[:-1]
    k_p = k_face[1:]
    rc_int = rho_cp[1:-1]

    fac = 2 * dt / ((h_m + h_p) * rc_int)
    c_p = fac * k_p / h_p   # 上邻居权重
    c_m = fac * k_m / h_m   # 下邻居权重
    c_c = 1.0 - c_p - c_m   # 对角权重（稳定时 ≥ 0）

    # 顶部 Robin BC 预计算（顶面 → 自然对流; 用顶部区间导热系数 k_face[-1]）
    # -k*(T[-1]-T[-2])/h[-1] = h_conv*(T[-1]-T_amb)
    # → T[-1] = bc_A * T[-2] + bc_B
    bc_A = (k_face[-1] / h[-1]) / (k_face[-1] / h[-1] + h_conv)
    bc_B = h_conv * T_air_ambient / (k_face[-1] / h[-1] + h_conv)

    # ==========================================
    # FDM 热传导主循环 (显式, 底部 Dirichlet + 顶部 Robin)
    # ==========================================
    T = np.ones(Nx) * float(T_initial_C)

    save_interval = max(1, int(save_dt / dt))
    plot_times       = []
    plot_T_bottom    = []
    plot_T_sample    = []
    plot_T_outer     = []
    plot_T_top_surf  = []

    for n in range(Nt):
        if n % save_interval == 0:
            plot_times.append(time_fdm[n])
            plot_T_bottom.append(bottom_fdm[n])
            plot_T_sample.append(np.dot(mesh.sample_weights, T))
            plot_T_outer.append(T[-1])
            if has_top_surface_obs:
                plot_T_top_surf.append(float(T[idx_top_surface[0]]))

        T[0]    = bottom_fdm[n]                             # 底部 Dirichlet BC
        T[1:-1] = c_c * T[1:-1] + c_m * T[:-2] + c_p * T[2:]  # 内部节点更新
        T[-1]   = bc_A * T[-2] + bc_B                       # 顶部 Robin BC

    return {
        "time_fdm": time_fdm,
        "bottom_temperature_fdm": bottom_fdm,
        "t_array": np.array(plot_times),
        "T_bottom_arr": np.array(plot_T_bottom),
        "T_sample_arr": np.array(plot_T_sample),
        "T_outer_surface_arr": np.array(plot_T_outer),
        "T_top_arr": np.array(plot_T_outer),  # 向后兼容别名 (= 最外表面温度)
        "T_top_surface_arr": np.array(plot_T_top_surf),
        "T_final": T,
        "dt": dt,
        "Nt": Nt,
        "Nx": Nx,
        "save_interval": save_interval,
        "mesh": mesh,
    }
