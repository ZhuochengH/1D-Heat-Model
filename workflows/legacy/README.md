# workflows/legacy — 历史/已取代工作流

本目录保存已明确取代或不再属于当前推荐执行路径的脚本, 仅用于标定开发溯源
与历史图表复现。**不是当前推荐工作流**。

- `main.py` — 初始 uv 脚手架生成的占位入口 (仅打印 "Hello")。
- 其它被取代的旧策略脚本若仍有科学溯源价值, 应保留在对应功能目录
  (calibration/validation/prediction/diagnostics) 或在此归档, 并记录取代关系。

当前推荐入口:

- 样品预测: `workflows/prediction/predict_sample_temperature_frozen_model.py`
- 已知偏移权威验证: `workflows/validation/validate_66C_candidate_known_offset.py`
- 最终模型配置: `thermal_model/config/final_frozen_model.py`
