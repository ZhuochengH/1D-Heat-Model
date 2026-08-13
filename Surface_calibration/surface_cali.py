import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# Block 1: 输入实验数据
# ============================================================

# Peltier 内置温度 / Set value
T_set = np.array([
    23.6, 30, 40, 50, 60, 70, 80, 90, 100
])

# 第一次表面温度测量
T1 = np.array([
    23.8, 30.2, 39.6, 49.2, 58.7, 68.5, 77.9, 87.4, 96.9
])

# 第二次表面温度测量
T2 = np.array([
    23.9, 30.1, 39.2, 48.4, 57.7, 66.9, 76.6, 86.2, 95.8
])


# ============================================================
# Block 2: 合并重复测量
# ============================================================

# 每个设定温度都有两次测量
# 为了保留重复实验的信息，把两组数据都用于回归
X = np.concatenate([T_set, T_set]).reshape(-1, 1)
Y = np.concatenate([T1, T2])

# 同时计算每一个设定温度的平均表面温度
T_mean = (T1 + T2) / 2

# 计算两次测量的标准差，可用于 error bar
T_std = np.std(
    np.vstack([T1, T2]),
    axis=0,
    ddof=1
)


# ============================================================
# Block 3: 进行线性回归
# ============================================================

model = LinearRegression()

model.fit(X, Y)

# 获取拟合参数
a = model.coef_[0]
b = model.intercept_

print("Linear regression result:")
print(f"T_surface = {a:.5f} * T_set + {b:.5f}")


# ============================================================
# Block 4: 评价拟合效果
# ============================================================

Y_pred = model.predict(X)

R2 = r2_score(Y, Y_pred)
RMSE = np.sqrt(mean_squared_error(Y, Y_pred))

print(f"R²   = {R2:.6f}")
print(f"RMSE = {RMSE:.3f} °C")


# ============================================================
# Block 5: 绘图
# ============================================================

# 生成连续的横坐标，用于画拟合直线
T_fit = np.linspace(
    T_set.min(),
    T_set.max(),
    300
).reshape(-1, 1)

surface_fit = model.predict(T_fit)


plt.figure(figsize=(7, 5))

# 两次原始实验
plt.scatter(
    T_set,
    T1,
    marker='o',
    label='Measurement 1'
)

plt.scatter(
    T_set,
    T2,
    marker='s',
    label='Measurement 2'
)

# 平均值 + error bar
plt.errorbar(
    T_set,
    T_mean,
    yerr=T_std,
    fmt='o',
    capsize=4,
    label='Mean ± SD'
)

# 回归直线
plt.plot(
    T_fit,
    surface_fit,
    label=(
        f'Linear fit\n'
        f'T_surface = {a:.4f} T_set + {b:.4f}\n'
        f'R² = {R2:.5f}'
    )
)

# 理想情况 y=x
plt.plot(
    T_fit,
    T_fit,
    linestyle='--',
    label='Ideal: T_surface = T_set'
)

plt.xlabel('Peltier set / internal temperature (°C)')
plt.ylabel('Measured Peltier surface temperature (°C)')

plt.legend()
plt.tight_layout()
plt.show()