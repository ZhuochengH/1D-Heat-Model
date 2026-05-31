# ==========================================
# 5. 绘图 (修改部分：添加两条可拖动横线，标记与Sample曲线的所有交点)
# ==========================================
# 注意：要实现拖动功能，请确保你在支持交互的窗口中运行此代码。
# 如果使用 Jupyter Notebook，请在代码最上方添加 `%matplotlib qt` 或 `%matplotlib widget`

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
ax.set_ylabel('Temperature ($^\circ$C)', fontsize=12)
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
