import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义几何厚度与空间网格（非均匀网格）
# ==========================================
L_coc_bot = 180e-6      # 底层COC
L_sample   = 20e-6      # 水性样品层
L_oil      = 50e-6      # 矿物油层
L_coc_top  = 600e-6     # 顶层COC
L_air      = 3000e-6    # Air gap层（隔热层）
L_pdms     = 200e-6     # Cap PDMS层

L_total = L_coc_bot + L_sample + L_oil + L_coc_top + L_air + L_pdms

# 各层界面位置
x_coc_bot_end = L_coc_bot
x_sample_end  = L_coc_bot + L_sample
x_oil_end     = L_coc_bot + L_sample + L_oil
x_coc_top_end = L_coc_bot + L_sample + L_oil + L_coc_top
x_air_end     = x_coc_top_end + L_air

# 非均匀网格步长：
#   关注区（COC_bot/Sample/Oil/COC_top）: dx_fine=5 μm，保留原精度
#   隔热层（Air gap）: dx_air=200 μm，粗化约40倍
#   顶盖（PDMS）:      dx_pdms=50 μm，适度粗化
# 核心加速原理：Air的热扩散率α≈2.1e-5 m²/s，远大于其他层（Water: 1.4e-7）。
# 均匀5μm网格时 dt≈5.9e-7 s（被Air约束）；Air粗化后 dt≈8.7e-5 s（被Water约束），加速约147×。
# 节点数同时从811减至~190，每步计算量再降约4×，综合加速≈700×。
dx_fine = 5e-6    # 精细区域（关注区）
dx_air  = 200e-6  # Air gap（隔热层，不关注温度分布）
dx_pdms = 50e-6   # PDMS顶盖

def make_layer(x0, x1, dx):
    n = max(1, int(round((x1 - x0) / dx)))
    return np.linspace(x0, x1, n + 1)

x = np.unique(np.concatenate([
    make_layer(0,             x_coc_bot_end, dx_fine),
    make_layer(x_coc_bot_end, x_sample_end,  dx_fine),
    make_layer(x_sample_end,  x_oil_end,     dx_fine),
    make_layer(x_oil_end,     x_coc_top_end, dx_fine),
    make_layer(x_coc_top_end, x_air_end,     dx_air ),
    make_layer(x_air_end,     L_total,       dx_pdms),
]))
Nx = len(x)
h  = np.diff(x)  # 节点间距数组，长度 Nx-1

# ==========================================
# 2. 定义材料属性并分配到对应网格
# ==========================================
rho_coc,  k_coc,  cp_coc  = 1020.0, 0.13,   1800.0
rho_w,    k_w,    cp_w    = 1000.0, 0.60,   4180.0
rho_oil,  k_oil,  cp_oil  = 876.0,  0.142,  1962.0
rho_air,  k_air,  cp_air  = 1.204,  0.0257, 1005.0
rho_pdms, k_pdms, cp_pdms = 970.0,  0.15,   1460.0

h_conv        = 5.0    # 顶部自然对流换热系数 W/(m²·K)
T_air_ambient = 25.0   # 环境温度 °C

rho = np.zeros(Nx)
k   = np.zeros(Nx)
cp  = np.zeros(Nx)

for i, xi in enumerate(x):
    if xi <= x_coc_bot_end + 1e-9:
        rho[i], k[i], cp[i] = rho_coc,  k_coc,  cp_coc
    elif xi <= x_sample_end + 1e-9:
        rho[i], k[i], cp[i] = rho_w,    k_w,    cp_w
    elif xi <= x_oil_end + 1e-9:
        rho[i], k[i], cp[i] = rho_oil,  k_oil,  cp_oil
    elif xi <= x_coc_top_end + 1e-9:
        rho[i], k[i], cp[i] = rho_coc,  k_coc,  cp_coc
    elif xi <= x_air_end + 1e-9:
        rho[i], k[i], cp[i] = rho_air,  k_air,  cp_air
    else:
        rho[i], k[i], cp[i] = rho_pdms, k_pdms, cp_pdms

idx_sample = np.where((x > L_coc_bot) & (x <= L_coc_bot + L_sample + 1e-9))[0]

# ==========================================
# 3. 设定Heater的热循环曲线 (Thermocycling Profile)
# ==========================================
ramp_rate = 10.0
times = [0.0]
temps = [25.0]

def add_step(target_T, hold_time):
    ramp_time = abs(target_T - temps[-1]) / ramp_rate
    times.append(times[-1] + ramp_time)
    temps.append(target_T)
    times.append(times[-1] + hold_time)
    temps.append(target_T)

t_set_top = 103.0
t_set_bottom = 50.0
add_step(95.0, 120.0)
add_step(t_set_bottom, 2)
add_step(t_set_top, 0.0)
add_step(t_set_bottom, 2)
add_step(t_set_top, 0.0)
add_step(t_set_bottom, 2)
add_step(t_set_top, 0.0)

t_total = times[-1]

# ==========================================
# 4. 预计算非均匀网格 FDM 系数
# ==========================================

# 各界面处调和平均导热系数 k_{i+1/2}（长度 Nx-1）
k_half = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])

# 内部节点（i=1..Nx-2）的前后间距与界面导热系数
h_m = h[:-1]       # x[i] - x[i-1]，长度 Nx-2
h_p = h[1:]        # x[i+1] - x[i]，长度 Nx-2
k_m = k_half[:-1]  # k_{i-1/2}，长度 Nx-2
k_p = k_half[1:]   # k_{i+1/2}，长度 Nx-2

rho_int = rho[1:-1]
cp_int  = cp[1:-1]

# 非均匀显式 FDM 稳定性条件（逐节点计算 dt 上限）：
# dt_i ≤ ρ_i·c_i·(h_m+h_p) / [2·(k_p/h_p + k_m/h_m)]
dt_stable = rho_int * cp_int * (h_m + h_p) / (2 * (k_p / h_p + k_m / h_m))
dt = np.min(dt_stable) * 0.9

Nt = int(t_total / dt) + 1
time_array     = np.linspace(0, t_total, Nt)
T_heater_array = np.interp(time_array, times, temps)

print(f"网格节点数: {Nx}（原均匀5μm网格: {int(round(L_total/5e-6))+1}）")
print(f"时间步长 dt = {dt*1e6:.1f} μs，总步数 Nt = {Nt:,}")

# 预计算三对角更新系数（完全向量化，时间循环内仅三次向量乘加）
# T_new[i] = c_c[i]*T[i] + c_m[i]*T[i-1] + c_p[i]*T[i+1]
fac = 2 * dt / ((h_m + h_p) * rho_int * cp_int)
c_p = fac * k_p / h_p   # 上邻居权重
c_m = fac * k_m / h_m   # 下邻居权重
c_c = 1.0 - c_p - c_m   # 对角权重（稳定时 ≥ 0）

# 顶部 Robin BC 预计算（PDMS顶面 → 自然对流）
# -k*(T[-1]-T[-2])/h[-1] = h_conv*(T[-1]-T_amb)
# → T[-1] = bc_A * T[-2] + bc_B
bc_A = (k[-1] / h[-1]) / (k[-1] / h[-1] + h_conv)
bc_B = h_conv * T_air_ambient / (k[-1] / h[-1] + h_conv)

# ==========================================
# 5. FDM 热传导主循环
# ==========================================
T = np.ones(Nx) * 25.0

save_interval = max(1, int(0.1 / dt))
plot_times    = []
plot_T_heater = []
plot_T_sample = []

for n in range(Nt):
    if n % save_interval == 0:
        plot_times.append(time_array[n])
        plot_T_heater.append(T_heater_array[n])
        plot_T_sample.append(np.mean(T[idx_sample]))

    T[0]    = T_heater_array[n]                            # 底部 Heater BC
    T[1:-1] = c_c * T[1:-1] + c_m * T[:-2] + c_p * T[2:] # 内部节点更新
    T[-1]   = bc_A * T[-2] + bc_B                          # 顶部 Robin BC

# ==========================================
# 6. 绘图（添加两条可拖动水平线与交点坐标）
# ==========================================
fig, ax = plt.subplots(figsize=(14, 7))

t_array   = np.array(plot_times)
T_s_array = np.array(plot_T_sample)

ax.plot(plot_times, plot_T_heater, label='Heater Temperature',       color='#d62728', linestyle='--', linewidth=2)
ax.plot(plot_times, plot_T_sample, label='Sample Layer Temperature', color='#1f77b4', linewidth=2)

ax.set_title('Thermal Cycling Profile: Heater vs Sample Layer', fontsize=15, fontweight='bold')
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel(r'Temperature ($^\circ$C)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(0, max(plot_times))
ax.set_ylim(20, 110)
ax.legend(fontsize=12, loc='lower right')


def find_intersections(x_data, y_data, y_level):
    intersections = []
    for i in range(len(y_data) - 1):
        y0, y1 = y_data[i], y_data[i + 1]
        if (y0 - y_level) * (y1 - y_level) < 0:
            t_cross = x_data[i] + (y_level - y0) / (y1 - y0) * (x_data[i + 1] - x_data[i])
            intersections.append(t_cross)
        elif abs(y0 - y_level) < 1e-10:
            intersections.append(x_data[i])
    return intersections


class DraggableHLine:
    def __init__(self, ax, y_init, x_data, y_data, color, label_prefix):
        self.ax = ax
        self.x_data = x_data
        self.y_data = y_data
        self.color = color
        self.label_prefix = label_prefix

        self.line = ax.axhline(y_init, color=color, linestyle='-', linewidth=2, picker=5, alpha=0.8)

        self.temp_label = ax.text(
            max(x_data) * 1.01, y_init,
            f'{y_init:.1f} °C',
            color=color, fontweight='bold', va='center', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1)
        )

        self.markers = []
        self.texts = []
        self._update_intersections(y_init)
        self.press = None
        self.connect()

    def connect(self):
        self.cidpress   = self.line.figure.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion  = self.line.figure.canvas.mpl_connect('motion_notify_event',  self.on_motion)

    def _update_intersections(self, y_level):
        for m in self.markers: m.remove()
        for t in self.texts:   t.remove()
        self.markers.clear()
        self.texts.clear()

        cross_times = find_intersections(self.x_data, self.y_data, y_level)
        for i, t_cross in enumerate(cross_times):
            marker, = self.ax.plot([t_cross], [y_level], marker='o', color=self.color, markersize=7, zorder=5)
            self.markers.append(marker)

            offset = 4 if i % 2 == 0 else -4
            va     = 'bottom' if i % 2 == 0 else 'top'
            text   = self.ax.text(
                t_cross, y_level + offset, f'{t_cross:.1f}s',
                color=self.color, ha='center', va=va, fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
            )
            self.texts.append(text)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        contains, _ = self.line.contains(event)
        if not contains: return
        self.press = self.line.get_ydata()[0], event.ydata

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.ax: return

        y0, ypress = self.press
        new_y = y0 + (event.ydata - ypress)
        new_y = max(min(self.y_data), min(new_y, max(self.y_data)))

        self.line.set_ydata([new_y, new_y])
        self.temp_label.set_position((max(self.x_data) * 1.01, new_y))
        self.temp_label.set_text(f'{new_y:.1f} °C')
        self._update_intersections(new_y)
        self.line.figure.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw_idle()


hline1 = DraggableHLine(ax, y_init=90.0, x_data=t_array, y_data=T_s_array,
                         color='darkorange', label_prefix='Denaturation')
hline2 = DraggableHLine(ax, y_init=60.0, x_data=t_array, y_data=T_s_array,
                         color='purple', label_prefix='Annealing')

plt.tight_layout()
plt.show()
