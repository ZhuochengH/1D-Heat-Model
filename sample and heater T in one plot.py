import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义几何厚度与空间网格
# ==========================================
L_coc_bot = 180e-6      # 底层COC (去除20um Microwell后)
L_sample = 20e-6        # 水性样品层
L_oil = 50e-6           # 矿物油层 (按50um计算)
L_coc_top = 600e-6      # 顶层COC
L_air = 3000e-6         # Air gap层 (3000um，位于COC_top之上)
L_pdms = 200e-6         # Cap PDMS层 (200um，位于Air之上，最顶层)

L_total = L_coc_bot + L_sample + L_oil + L_coc_top + L_air + L_pdms  # 总厚度 4050 um
dx = 5e-6               # 空间步长 5 um (划分170个网格以保证精度)
Nx = int(round(L_total / dx)) + 1
x = np.linspace(0, L_total, Nx)

# ==========================================
# 2. 定义材料属性并分配到对应网格
# ==========================================
rho_coc, k_coc, cp_coc = 1020.0, 0.13, 1800.0
rho_w, k_w, cp_w = 1000.0, 0.60, 4180.0
rho_oil, k_oil, cp_oil = 876.0, 0.142, 1962.0
rho_air, k_air, cp_air = 1.204, 0.0257, 1005.0   # Air gap层物性参数
rho_pdms, k_pdms, cp_pdms = 970.0, 0.15, 1460.0  # Cap PDMS层物性参数

# 顶部对流换热边界条件参数 (PDMS顶面向自然空气散热)
h_conv = 5.0        # 自然对流换热系数 W/(m²·K)
T_air_ambient = 25.0  # 环境温度 °C

rho = np.zeros(Nx)
k = np.zeros(Nx)
cp = np.zeros(Nx)

# 各层界面位置 (从下到上：COC_bot → Sample → Oil → COC_top → Air → PDMS)
x_coc_bot_end  = L_coc_bot
x_sample_end   = L_coc_bot + L_sample
x_oil_end      = L_coc_bot + L_sample + L_oil
x_coc_top_end  = L_coc_bot + L_sample + L_oil + L_coc_top
x_air_end      = L_coc_bot + L_sample + L_oil + L_coc_top + L_air

for i, xi in enumerate(x):
    if xi <= x_coc_bot_end + 1e-9:
        rho[i], k[i], cp[i] = rho_coc, k_coc, cp_coc
    elif xi <= x_sample_end + 1e-9:
        rho[i], k[i], cp[i] = rho_w, k_w, cp_w
    elif xi <= x_oil_end + 1e-9:
        rho[i], k[i], cp[i] = rho_oil, k_oil, cp_oil
    elif xi <= x_coc_top_end + 1e-9:
        rho[i], k[i], cp[i] = rho_coc, k_coc, cp_coc
    elif xi <= x_air_end + 1e-9:
        rho[i], k[i], cp[i] = rho_air, k_air, cp_air  # Air gap层
    else:
        rho[i], k[i], cp[i] = rho_pdms, k_pdms, cp_pdms  # Cap PDMS层

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
add_step(105.0, 135.0)  # 升至105度，保持 2分15秒 (135秒)
add_step(60.0, 60.0)   # 降至60度，保持 1分钟 (60秒)
add_step(105.0, 15.0)   # 升至105度，保持 15秒
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
    
    # 边界条件：顶部对流换热 (PDMS顶面牛顿冷却定律：-k dT/dn = h*(T_surf - T_amb))
    T[-1] = (k[-1] * T[-2] / dx + h_conv * T_air_ambient) / (k[-1] / dx + h_conv)

# ==========================================
# 5. 绘图 (修改部分：添加两条可拖动竖线与交点坐标)
# ==========================================
# 注意：要实现拖动功能，请确保你在支持交互的窗口中运行此代码。
# 如果使用 Jupyter Notebook，请在代码最上方添加 `%matplotlib qt` 或 `%matplotlib widget`
# 如果使用 PyCharm / VS Code 等普通 Python 脚本运行，则直接运行即可。

import matplotlib.pyplot as plt
import numpy as np
 
fig, ax = plt.subplots(figsize=(14, 7))
 
t_array = np.array(plot_times)
T_s_array = np.array(plot_T_sample)
 
# 绘制基础曲线
ax.plot(plot_times, plot_T_heater, label='Heater Temperature', color='#d62728', linestyle='--', linewidth=2)
ax.plot(plot_times, plot_T_sample, label='Sample Layer Temperature', color='#1f77b4', linewidth=2)
 
ax.set_title('Thermal Cycling Profile: Heater vs Sample Layer', fontsize=15, fontweight='bold')
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel(r'Temperature ($^\circ$C)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(0, max(plot_times))
ax.set_ylim(20, 110)
ax.legend(fontsize=12, loc='lower right')
 
 
def find_intersections(x_data, y_data, y_level):
    """
    找出曲线 (x_data, y_data) 与水平线 y=y_level 的所有交点的 x 坐标。
    原理：检测相邻两个数据点之间 y 值是否跨越了 y_level，
    如果跨越了，就用线性插值精确计算交点的 x 位置。
    """
    intersections = []
    for i in range(len(y_data) - 1):
        y0, y1 = y_data[i], y_data[i + 1]
        # 检测是否跨越 y_level（一个在上面，一个在下面，或者恰好等于）
        if (y0 - y_level) * (y1 - y_level) < 0:
            # 线性插值求精确交点
            t_cross = x_data[i] + (y_level - y0) / (y1 - y0) * (x_data[i + 1] - x_data[i])
            intersections.append(t_cross)
        elif abs(y0 - y_level) < 1e-10:
            intersections.append(x_data[i])
    return intersections
 
 
class DraggableHLine:
    """
    可拖动的水平线类。
    拖动时实时计算与 Sample 温度曲线的所有交点，
    并在每个交点处标记圆点和时间坐标。
    """
    def __init__(self, ax, y_init, x_data, y_data, color, label_prefix):
        self.ax = ax
        self.x_data = x_data
        self.y_data = y_data
        self.color = color
        self.label_prefix = label_prefix
 
        # 1. 创建水平线, picker=5 表示鼠标距离线条 5 像素内即可选中
        self.line = ax.axhline(y_init, color=color, linestyle='-', linewidth=2, picker=5, alpha=0.8)
 
        # 2. 在横线右侧显示当前温度值
        self.temp_label = ax.text(
            max(x_data) * 1.01, y_init,
            f'{y_init:.1f} °C',
            color=color, fontweight='bold', va='center', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1)
        )
 
        # 3. 初始化交点标记（圆点）和文本标签列表
        self.markers = []
        self.texts = []
 
        # 4. 绘制初始交点
        self._update_intersections(y_init)
 
        self.press = None
        self.connect()
 
    def connect(self):
        self.cidpress = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
 
    def _update_intersections(self, y_level):
        """清除旧的交点标记，重新计算并绘制新的交点"""
        # 清除旧标记
        for m in self.markers:
            m.remove()
        for t in self.texts:
            t.remove()
        self.markers.clear()
        self.texts.clear()
 
        # 计算新交点
        cross_times = find_intersections(self.x_data, self.y_data, y_level)
 
        # 为每个交点绘制标记和标签
        for i, t_cross in enumerate(cross_times):
            # 圆点标记
            marker, = self.ax.plot([t_cross], [y_level], marker='o', color=self.color,
                                   markersize=7, zorder=5)
            self.markers.append(marker)
 
            # 时间标签 (交替放在上方和下方，避免重叠)
            offset = 4 if i % 2 == 0 else -4
            va = 'bottom' if i % 2 == 0 else 'top'
            text = self.ax.text(
                t_cross, y_level + offset,
                f'{t_cross:.1f}s',
                color=self.color, ha='center', va=va, fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
            )
            self.texts.append(text)
 
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains, _ = self.line.contains(event)
        if not contains:
            return
        self.press = self.line.get_ydata()[0], event.ydata
 
    def on_motion(self, event):
        if self.press is None:
            return
        if event.inaxes != self.ax:
            return
 
        y0, ypress = self.press
        dy = event.ydata - ypress
        new_y = y0 + dy
 
        # 限制横线不能拖出图表的温度范围
        new_y = max(min(self.y_data), min(new_y, max(self.y_data)))
 
        # 更新横线位置
        self.line.set_ydata([new_y, new_y])
 
        # 更新右侧温度标签
        self.temp_label.set_position((max(self.x_data) * 1.01, new_y))
        self.temp_label.set_text(f'{new_y:.1f} °C')
 
        # 重新计算并绘制交点
        self._update_intersections(new_y)
 
        self.line.figure.canvas.draw_idle()
 
    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw_idle()
 
 
# --- 实例化两条可拖动的横线 ---
# 必须赋值给变量，否则 Python 的垃圾回收机制会清理掉它们，导致交互失效
hline1 = DraggableHLine(ax, y_init=95.0, x_data=t_array, y_data=T_s_array,
                         color='darkorange', label_prefix='Denaturation')
hline2 = DraggableHLine(ax, y_init=60.0, x_data=t_array, y_data=T_s_array,
                         color='purple', label_prefix='Annealing')
 
plt.tight_layout()
plt.show()




# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(12, 6))

# t_array = np.array(plot_times)
# T_s_array = np.array(plot_T_sample)

# # 绘制基础曲线
# ax.plot(plot_times, plot_T_heater, label='Heater Temperature', color='#d62728', linestyle='--', linewidth=2)
# ax.plot(plot_times, plot_T_sample, label='Sample Layer Temperature', color='#1f77b4', linewidth=2)

# ax.set_title('Thermal Cycling Profile: Heater vs Sample Layer', fontsize=15, fontweight='bold')
# ax.set_xlabel('Time (seconds)', fontsize=12)
# ax.set_ylabel(r'Temperature ($^\circ$C)', fontsize=12)

# # 添加辅助线标注关键温度
# ax.axhline(y=86, color='gray', linestyle=':', alpha=0.5)
# ax.axhline(y=60, color='gray', linestyle=':', alpha=0.5)

# ax.grid(True, linestyle='--', alpha=0.6)
# ax.set_xlim(0, max(plot_times))
# ax.set_ylim(20, 110)
# ax.legend(fontsize=12, loc='lower right')

# # --- 核心：定义可拖动竖线的交互类 ---
# class DraggableVLine:
#     def __init__(self, ax, x_init, x_data, y_data, color):
#         self.ax = ax
#         self.x_data = x_data
#         self.y_data = y_data
        
#         # 1. 创建竖线，picker=5 表示鼠标在距离线条 5 个像素内即可选中
#         self.line = ax.axvline(x_init, color=color, linestyle='-', linewidth=2.5, picker=5, alpha=0.8)
        
#         # 2. 计算初始的交点 y 值
#         y_init = np.interp(x_init, x_data, y_data)
        
#         # 3. 创建交点处的圆点标记和文本标签
#         self.marker, = ax.plot([x_init], [y_init], marker='o', color=color, markersize=8)
#         self.text = ax.text(x_init, y_init + 2, f'{y_init:.2f} °C\n({x_init:.1f}s)', 
#                             color=color, ha='center', va='bottom', fontweight='bold',
#                             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)) # 加个半透明白色背景防遮挡
        
#         self.press = None
#         self.connect()

#     def connect(self):
#         # 绑定鼠标事件
#         self.cidpress = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
#         self.cidrelease = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
#         self.cidmotion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

#     def on_press(self, event):
#         # 确保点击在图表内且点中了当前竖线
#         if event.inaxes != self.ax: return
#         contains, _ = self.line.contains(event)
#         if not contains: return
#         # 记录按下时的初始位置
#         self.press = self.line.get_xdata()[0], event.xdata

#     def on_motion(self, event):
#         # 如果没有按下鼠标，或者拖出图表外，则不作处理
#         if self.press is None: return
#         if event.inaxes != self.ax: return
        
#         x0, xpress = self.press
#         dx = event.xdata - xpress
#         new_x = x0 + dx
        
#         # 限制竖线不能拖出数据的时间范围
#         new_x = max(min(self.x_data), min(new_x, max(self.x_data)))
        
#         # 更新竖线位置
#         self.line.set_xdata([new_x, new_x])
        
#         # 重新插值计算当前 x 对应的 Sample 曲线上的 y 值
#         new_y = np.interp(new_x, self.x_data, self.y_data)
        
#         # 同步更新圆点标记和文本标签
#         self.marker.set_data([new_x], [new_y])
#         self.text.set_position((new_x, new_y + 2))
#         self.text.set_text(f'{new_y:.2f} °C\n({new_x:.1f}s)')
        
#         # 重绘图表
#         self.line.figure.canvas.draw_idle()

#     def on_release(self, event):
#         # 松开鼠标，清除状态
#         self.press = None
#         self.line.figure.canvas.draw_idle()

# # --- 实例化两条可拖动的竖线 ---
# # 必须赋值给变量（如 vline1, vline2），否则 Python 的垃圾回收机制会清理掉它们，导致交互失效
# # 我将它们分别初始化在 50 秒和 200 秒的位置，你可以随意拖动
# vline1 = DraggableVLine(ax, x_init=50.0, x_data=t_array, y_data=T_s_array, color='darkorange')
# vline2 = DraggableVLine(ax, x_init=200.0, x_data=t_array, y_data=T_s_array, color='purple')

# plt.tight_layout()
# plt.show()