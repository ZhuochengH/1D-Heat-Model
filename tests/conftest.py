"""pytest 共享配置：让 tests 能导入 Surface_calibration 包内的模块。"""

import sys
from pathlib import Path

# 集成测试会运行完整 main() 流程（含绘图），使用无头后端避免 GUI 依赖
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURFACE_CALIBRATION_DIR = PROJECT_ROOT / "Surface_calibration"

# 让 tests 也能导入仓库根目录下的 heat_model 等顶层模块
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SURFACE_CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_CALIBRATION_DIR))
