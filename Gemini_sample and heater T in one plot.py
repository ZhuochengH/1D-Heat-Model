import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义几何厚度与空间网格
# ==========================================
L_coc_bot = 180e-6      # 底层COC (去除20um Microwell后)
L_sample = 20e-6        # 水性样品层
L_oil = 50e-6           # 矿物油层 (按50um计算)
L_coc_top = 600e-6      # 顶层COC

L_total = L_coc_bot + L_sample + L_oil + L_coc_top  # 总厚度 850 um
dx = 5e-6               # 空间步长 5 um (划分170个网格以保证精度)
Nx = int(round(L_total / dx)) + 1
x = np.linspace(0, L_total, Nx)

# ==========================================
# 2. 定义材料属性并分配到对应网格
# ==========================================
rho_coc, k_coc, cp_coc = 1020.0, 0.13, 1800.0
rho_w, k_w, cp_w = 1000.0, 0.60, 4180.0
rho_oil, k_oil, cp_oil = 876.0, 0.142, 1962.0

rho = np.zeros(Nx)
k = np.zeros(Nx)
cp = np.zeros(Nx)

for i, xi in enumerate(x):
    if xi <= L_coc_bot + 1e-9:
        rho[i], k[i], cp[i] = rho_coc, k_coc, cp_coc
    elif xi <= L_coc_bot + L_sample + 1e-9:
        rho[i], k[i], cp[i] = rho_w, k_w, cp_w
    elif xi <= L_coc_bot + L_sample + L_oil + 1e-9:
        rho[i], k[i], cp[i] = rho_oil, k_oil, cp_oil
    else:
        rho[i], k[i], cp[i] = rho_coc, k_coc, cp_coc

# 记录Sample层对应的节点索引，用于后续求平均温度
idx_sample = np.where((x > L_coc_bot) & (x <= L_coc_bot + L_sample + 1e-9))[0]

# ==========================================
# 3. 设定Heater的热循环曲线 (Thermocycling Profile)
# ==========================================
ramp_rate = 10.0  # 升降温速率 
times = [0.0]
temps = [25.0]

def add_step(target_T, hold_time):
    # 计算达到目标温度所需的Ramp时间
    ramp_time = abs(target_T - temps[-1]) / ramp_rate
    times.append(times[-1] + ramp_time)
    temps.append(target_T)
    # 添加保持时间 (Hold)
    times.append(times[-1] + hold_time)
    temps.append(target_T)

# 按照您的PCR Protocol逐步添加
add_step(95.0, 135.0)  # 升至86度，保持 2分15秒 (135秒)
add_step(60.0, 60.0)   # 降至60度，保持 1分钟 (60秒)
add_step(95.0, 15.0)   # 升至86度，保持 15秒
add_step(60.0, 60.0)   # 降至60度，保持 1分钟 (60秒)

t_total = times[-1]

# 计算满足数值稳定性的最大时间步长 dt
alpha_max = np.max(k / (rho * cp))
dt_max = (dx**2) / (2 * alpha_max)
dt = dt_max * 0.9  # 取90%作为安全步长
Nt = int(t_total / dt) + 1
time_array = np.linspace(0, t_total, Nt)

# 生成Heater在每个时间步的温度数组 (线性插值)
T_heater_array = np.interp(time_array, times, temps)

# ==========================================
# 4. FDM 热传导计算 (使用NumPy向量化加速)
# ==========================================
T = np.ones(Nx) * 25.0  # 初始温度 25度

# 预先计算节点界面的调和平均导热系数 (Harmonic mean for interfaces)
k_plus = np.zeros(Nx-2)
k_minus = np.zeros(Nx-2)
for i in range(1, Nx-1):
    k_plus[i-1] = 2 * k[i] * k[i+1] / (k[i] + k[i+1])
    k_minus[i-1] = 2 * k[i] * k[i-1] / (k[i] + k[i-1])

# FDM 系数
coeff = dt / (rho[1:-1] * cp[1:-1] * dx**2)

# 用于绘图的数据记录 (不需要记录每个dt，每隔一段记录一次以节省内存)
save_interval = max(1, int(0.1 / dt)) # 每0.1秒记录一次
plot_times = []
plot_T_heater = []
plot_T_sample = []

for n in range(Nt):
    # 记录数据
    if n % save_interval == 0:
        plot_times.append(time_array[n])
        plot_T_heater.append(T_heater_array[n])
        plot_T_sample.append(np.mean(T[idx_sample]))
    
    # 边界条件：底部接触Heater
    T[0] = T_heater_array[n]
    
    # 核心计算：内部节点温度更新 (向量化操作，比for循环快上百倍)
    T_new_internal = T[1:-1] + coeff * (k_plus * (T[2:] - T[1:-1]) - k_minus * (T[1:-1] - T[:-2]))
    T[1:-1] = T_new_internal
    
    # 边界条件：顶部绝热
    T[-1] = T[-2]

# ==========================================
# 5. 绘图 (修改部分：添加两条可拖动竖线与交点坐标)
# ==========================================
# 注意：要实现拖动功能，请确保你在支持交互的窗口中运行此代码。
# 如果使用 Jupyter Notebook，请在代码最上方添加 `%matplotlib qt` 或 `%matplotlib widget`
# 如果使用 PyCharm / VS Code 等普通 Python 脚本运行，则直接运行即可。

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

t_array = np.array(plot_times)
T_s_array = np.array(plot_T_sample)

# 绘制基础曲线
ax.plot(plot_times, plot_T_heater, label='Heater Temperature', color='#d62728', linestyle='--', linewidth=2)
ax.plot(plot_times, plot_T_sample, label='Sample Layer Temperature', color='#1f77b4', linewidth=2)

ax.set_title('Thermal Cycling Profile: Heater vs Sample Layer', fontsize=15, fontweight='bold')
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Temperature ($^\circ$C)', fontsize=12)

# 添加辅助线标注关键温度
ax.axhline(y=86, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=60, color='gray', linestyle=':', alpha=0.5)

ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(0, max(plot_times))
ax.set_ylim(20, 110)
ax.legend(fontsize=12, loc='lower right')

# --- 核心：定义可拖动竖线的交互类 ---
class DraggableVLine:
    def __init__(self, ax, x_init, x_data, y_data, color):
        self.ax = ax
        self.x_data = x_data
        self.y_data = y_data
        
        # 1. 创建竖线，picker=5 表示鼠标在距离线条 5 个像素内即可选中
        self.line = ax.axvline(x_init, color=color, linestyle='-', linewidth=2.5, picker=5, alpha=0.8)
        
        # 2. 计算初始的交点 y 值
        y_init = np.interp(x_init, x_data, y_data)
        
        # 3. 创建交点处的圆点标记和文本标签
        self.marker, = ax.plot([x_init], [y_init], marker='o', color=color, markersize=8)
        self.text = ax.text(x_init, y_init + 2, f'{y_init:.2f} °C\n({x_init:.1f}s)', 
                            color=color, ha='center', va='bottom', fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)) # 加个半透明白色背景防遮挡
        
        self.press = None
        self.connect()

    def connect(self):
        # 绑定鼠标事件
        self.cidpress = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        # 确保点击在图表内且点中了当前竖线
        if event.inaxes != self.ax: return
        contains, _ = self.line.contains(event)
        if not contains: return
        # 记录按下时的初始位置
        self.press = self.line.get_xdata()[0], event.xdata

    def on_motion(self, event):
        # 如果没有按下鼠标，或者拖出图表外，则不作处理
        if self.press is None: return
        if event.inaxes != self.ax: return
        
        x0, xpress = self.press
        dx = event.xdata - xpress
        new_x = x0 + dx
        
        # 限制竖线不能拖出数据的时间范围
        new_x = max(min(self.x_data), min(new_x, max(self.x_data)))
        
        # 更新竖线位置
        self.line.set_xdata([new_x, new_x])
        
        # 重新插值计算当前 x 对应的 Sample 曲线上的 y 值
        new_y = np.interp(new_x, self.x_data, self.y_data)
        
        # 同步更新圆点标记和文本标签
        self.marker.set_data([new_x], [new_y])
        self.text.set_position((new_x, new_y + 2))
        self.text.set_text(f'{new_y:.2f} °C\n({new_x:.1f}s)')
        
        # 重绘图表
        self.line.figure.canvas.draw_idle()

    def on_release(self, event):
        # 松开鼠标，清除状态
        self.press = None
        self.line.figure.canvas.draw_idle()

# --- 实例化两条可拖动的竖线 ---
# 必须赋值给变量（如 vline1, vline2），否则 Python 的垃圾回收机制会清理掉它们，导致交互失效
# 我将它们分别初始化在 50 秒和 200 秒的位置，你可以随意拖动
vline1 = DraggableVLine(ax, x_init=50.0, x_data=t_array, y_data=T_s_array, color='darkorange')
vline2 = DraggableVLine(ax, x_init=200.0, x_data=t_array, y_data=T_s_array, color='purple')

plt.tight_layout()
plt.show()