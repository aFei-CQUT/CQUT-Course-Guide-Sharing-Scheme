import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 定义数据
v = np.arange(50, 300, 50)
v_one_over_two = v**(1/2)
ip = np.array([-1.361e-5, -2.841e-5, -4.252e-5, -5.687e-5, -7.175e-5]) * -1

# 创建一个指定大小的新图
plt.figure(figsize=(12, 6))

# 绘制 v 对 ip 的关系图
plt.subplot(1, 2, 1)
plt.plot(v, ip, marker='o', linestyle='-', color='b', label='$i_{p}$')

# 进行线性拟合
coefficients_v, residuals_v = np.polyfit(v, ip, 1, cov=True)
poly_v = np.poly1d(coefficients_v)
plt.plot(v, poly_v(v), linestyle='--', color='gray', label='Linear Fit')

# 设置 v 对 ip 的标签和标题
plt.xlabel('$v / (mV/s)$', fontsize=12, weight='bold')
plt.ylabel('$i_{p} / A$', fontsize=12, weight='bold')
plt.title('$i_{p} - v$ 关系图', fontsize=14, weight='bold')
plt.grid(True, which='both')
plt.legend()
plt.minorticks_on()

# 绘制 v^(1/2) 对 ip 的关系图
plt.subplot(1, 2, 2)
plt.plot(v_one_over_two, ip, marker='o', linestyle='-', color='r', label='$i_{p}$ ($v^{1/2}$)')

# 进行最优拟合（例如二次多项式）
coefficients_sqrt_v, residuals_sqrt_v = np.polyfit(v_one_over_two, ip, 2, cov=True)
poly_sqrt_v = np.poly1d(coefficients_sqrt_v)
plt.plot(v_one_over_two, poly_sqrt_v(v_one_over_two), linestyle='--', color='gray', label='Quadratic Fit')

# 设置 v^(1/2) 对 ip 的标签和标题
plt.xlabel('$v^{1/2} / (mV/s^{1/2})$', fontsize=12, weight='bold')
plt.ylabel('$i_{p} / A$', fontsize=12, weight='bold')
plt.title('$i_{p} - v^{1/2}$ 关系图', fontsize=14, weight='bold')
plt.grid(True, which='both')
plt.legend()
plt.minorticks_on()

# 调整布局并显示图形
plt.tight_layout()
plt.savefig('./循环伏安拟合图结果/扫描速度vs峰值电流.png', dpi=300)

plt.show()
