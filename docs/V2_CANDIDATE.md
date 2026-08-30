# THERMAL MODEL V2 CANDIDATE (历史记录 — 已晋升为 FINAL_FROZEN_THERMAL_MODEL_V2)

状态: **PROMOTED (2026-08-30)**。本候选已通过 recalibration / 独立外部验证 /
sample prediction / sensitivity validation, 正式晋升为:

    FINAL_FROZEN_THERMAL_MODEL_V2
    (见 thermal_model/config/final_frozen_model_v2.py 与
     docs/FINAL_THERMAL_MODEL_V2.md)

本文件保留作为候选阶段的科学记录 (数值与 final 完全一致)。

## 两个科学变更

1. **材料修正**: 通用矿物油 -> 3M Fluorinert FC-70 (Sigma-Aldrich F9880)。
   室温代表性常数物性 (无温度依赖):
   - k = 0.070 W/(m·K)
   - rho = 1940 kg/m³
   - cp = 1050 J/(kg·K)

2. **绝缘几何修正** (仅绝缘几何): 移除 PDMS 层。
   - 旧: Bottom COC -> sample -> oil -> Top COC -> sealed air -> PDMS -> boundary
   - 新: Bottom COC -> sample -> FC-70 -> Top COC -> sealed air -> boundary

裸顶标定/验证几何仅做 oil -> FC-70 替换, 结构不变。

## 实现位置 (单一权威定义)

- `thermal_model/config/thermal_model_v2_candidate.py`
  - `FC70_K_W_MK / FC70_RHO_KG_M3 / FC70_CP_J_KGK`
  - `make_v2_materials(k_eff, cp_eff, rho)` (删除 Oil, 新增 FC70)
  - `V2_BARE_TOP_COC_LAYERS` (4 层, 850 um)
  - `V2_INSULATED_NO_PDMS_LAYERS` (5 层, 3850 um, 无 PDMS)
- `workflows/v2_fc70_no_pdms/v2_candidate_workflow.py` (完整工作流)
- `tests/test_thermal_model_v2_candidate.py` (17 项)

## 结果摘要 (2026-08-29)

标定 (同一 66C redo 数据 / 同一流程):
- V1: k=0.0675, cp=700, tau=8.0, RMSE 0.6368
- V2: k=0.070,  cp=700, tau=8.0, RMSE 0.6333

三个外部验证 (零重拟合) V2 均小幅改善:
- 60C: 1.3749 -> 1.3139
- 72C: 3.0817 -> 3.0132
- 3s:  1.0643 -> 1.0386

样品预测 (08.24 无保持, Setpoint 周期, 29 周期):
- V1 mean HIGH 84.574 / V2 84.703 (Δ +0.129 °C)
- 完整对比见 `outputs/v2_fc70_no_pdms_comparison/`。

敏感性 (08.24 no-holding, OAT ±10%, S_range C per ±10%):
- 排名与 V1 总体一致: k_eff 0.880 (最敏感) > cp_eff 0.531 /
  rho_COC 0.531 (主导) > cp_FC70 0.182 > cp_sample 0.136 > d_air 0.084 /
  k_air 0.084 > k_FC70 0.046 > h_conv 0.028 > epsilon 0.018 >
  k_sample 0.006 > cp_air 0.0005
- FC-70 热容局部敏感性略大于历史 oil (cp_oil 0.156 -> cp_FC70 0.182),
  但无任何参数接近 k_eff
- tau_top 负控制 PASS (0/8/16 s 样品完全一致)
- 边界消融 (h=0/eps=0 联合) 与 V1 量级相当 (V2 ~+0.64/+0.73 C vs
  V1 ~+0.68/+0.68 C), 相对整体循环范围仍较小
- 完整输出见 `outputs/v2_fc70_no_pdms_sensitivity/`。

## 保护

- `FINAL_FROZEN_THERMAL_MODEL_V1` 未修改。
- `heat_model.DEFAULT_MATERIALS` / `LEGACY_INSULATED_LAYERS` /
  `BARE_TOP_COC_LAYERS` 未修改 (V1 权威对象保留)。
- 候选阶段未提交 / 未推送 / 未修改 Git tag; 晋升时以正式 commit 提交。
