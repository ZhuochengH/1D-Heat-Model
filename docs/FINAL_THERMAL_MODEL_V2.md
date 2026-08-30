# FINAL FROZEN THERMAL MODEL V2 (FC-70 + no-PDMS simplified insulated geometry)

状态: **FROZEN** (2026-08-30)。本模型为项目当前权威最终热模型, 用于后续
thesis analysis 与 sample-temperature prediction。

## Why V2 replaced V1

1. 实验密封液体被正确识别为 3M Fluorinert FC-70 (Sigma-Aldrich F9880),
   替代了 V1 中的通用矿物油。
2. 简化绝缘几何有意忽略薄 PDMS 密封层 (仅绝缘几何; 裸顶标定/验证几何仅
   oil -> FC-70 替换, 结构不变)。
3. 重校准 (同一 66C redo 数据集, 同一流程):
   - k_eff: 0.0675 -> 0.0700 W/(m K)
   - cp_eff: 700 -> 700 J/(kg K)
   - tau_top: 8 -> 8 s
4. 标定 RMSE: 0.6368 -> 0.6333 C (基本不变, 略改善)
5. 外部验证 RMSE (零重拟合):
   - 60C: 1.3749 -> 1.3139 C
   - 72C: 3.0817 -> 3.0132 C
   - 3s : 1.0643 -> 1.0386 C
6. V2 样品预测 (08.24 no-holding, 29 周期):
   - mean HIGH = 84.703 C
   - mean LOW  = 58.686 C
   - HIGH-LOW  = 26.017 C
7. V1 -> V2 样品预测变化: HIGH +0.129 C, LOW -0.071 C, range +0.200 C
8. Sensitivity 与 V1 总体一致 (k_eff 最敏感, cp_eff/rho_COC 主导;
   FC-70 热容局部敏感性略大于历史 oil, 但无任何参数接近 k_eff)。
9. tau_top 不影响样品预测 (负控制 PASS)。
10. 完整测试套件通过。

## 科学表述 (必须遵守)

- **validation RMSE 不是 sample-temperature uncertainty。**
- **sensitivity 不是 uncertainty analysis。**
- **tau_top 是输出侧 (output-side) Top 观测滞后**, 只作用于预测的
  Top 观测模型, 绝不作用于样品温度。
- **样品温度取自原始物理有限体积温度场** (physical finite-volume field)
  经样品层控制体积加权 (sample-layer weighting), 从不经过滞后。
  注: 历史代码/函数名中的 "fdm" 仅为历史命名, 数值方法是一维节点中心
  有限体积 (node-centered finite-volume) 模型。
- V2 现为最终模型; V1 保留用于历史复现 (legacy/reproducibility)。

## 最终参数

| 参数 | 值 |
|---|---|
| k_eff | 0.0700 W/(m K) |
| cp_eff | 700 J/(kg K) |
| rho_COC | 1020 kg/m3 |
| tau_top | 8.0 s (输出侧, 仅 Top 观测) |
| h_conv | 10 W/(m2 K) |
| epsilon | 0.90 |
| sigma_SB | 5.670374419e-8 W/(m2 K4) |
| F_view | 1.0 |
| FC-70 k | 0.070 W/(m K) |
| FC-70 rho | 1940 kg/m3 |
| FC-70 cp | 1050 J/(kg K) |

## 最终几何

裸顶 (bare, 标定/验证):
```
Bottom COC 180 um
Water sample 20 um
FC-70 50 um
Top COC 600 um          (总 850 um)
```

简化绝缘 (insulated, 前向扩展, 无 PDMS):
```
Bottom COC 180 um
Water sample 20 um
FC-70 50 um
Top COC 600 um
Sealed air 3000 um      (总 3850 um; 外部对流+辐射边界作用于空气外表面)
```

## 验证参考 (只读)

- 66C 标定 RMSE = 0.6333 C; MAE = 0.4802 C; mean residual = +0.0534 C;
  R² = 0.9904
- 60C 外部验证 = 1.3139 C (SETPOINT_90C_EVENT_PLUS_1S, 权威)
- 72C 外部验证 = 3.0132 C (已知偏移, 权威; 冷却相局限同 V1)
- 3s 外部验证 = 1.0386 C (权威)
- 外部验证均值 = 1.7886 C; 中位 = 1.3139 C; 最差 = 3.0132 C
- 样品预测 (08.24 no-holding, 29 周期): mean HIGH 84.703 / mean LOW
  58.686 / range 26.017 / median HIGH 84.696 / median LOW 58.972 /
  overall max 89.097 C

## 权威配置模块

- `thermal_model/config/final_frozen_model_v2.py` (FINAL_FROZEN_THERMAL_MODEL_V2)
- FC-70 常量与 V2 层叠定义来源:
  `thermal_model/config/thermal_model_v2_candidate.py` (候选历史定义,
  数值与 final 完全一致; 保留为候选记录)
- V1 历史冻结: `thermal_model/config/final_frozen_model.py`
  (FINAL_FROZEN_THERMAL_MODEL_V1, 保留 Oil + PDMS 定义, 未修改)

## V1 保留确认

- FINAL_FROZEN_THERMAL_MODEL_V1 未修改, 可复现历史结果。
- heat_model.DEFAULT_MATERIALS 中的 Oil (0.142/876/1962) 与
  PDMS (0.15/970/1460) 定义保留。
- LEGACY_INSULATED_LAYERS (6 层, 含 PDMS) 与 BARE_TOP_COC_LAYERS (Oil)
  保留为历史几何。
- 历史 V1 tags (thermal-model-final-v1 / thermal-model-final-v1.0.1) 未动。
