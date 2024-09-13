import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, diff, solve
from matplotlib import rcParams
import zipfile
import os

class CombustionHeatAnalysis:
    """
    燃烧热分析类
    
    用于分析和可视化燃烧热实验数据。
    """

    def __init__(self, temperature_data, substance_name, arc_name):
        """
        初始化燃烧热分析对象
        
        :param temperature_data: 温度数据列表
        :param substance_name: 物质名称
        :param arc_name: 存档名称
        """
        self.temperature_data = temperature_data
        self.substance_name = substance_name
        self.arc_name = arc_name
        self.setup_plot_config()
        self.process_data()

    def setup_plot_config(self):
        """设置绘图配置"""
        config = {
            "font.family": 'serif',
            "font.size": 12,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
        }
        rcParams.update(config)

    def segment_data(self):
        """将数据分段为点火前、反应中和熄灭后"""
        self.before_ignition = self.temperature_data[:7]
        self.during_reaction = self.temperature_data[7:-6]
        self.after_extinguish = self.temperature_data[-7:]

    def create_point_data(self):
        """创建点数据数组"""
        self.point_before = np.array([range(1, 8), self.before_ignition]).T
        self.point_during = np.array([range(8, len(self.temperature_data)-5), self.during_reaction]).T
        self.point_after = np.array([range(len(self.temperature_data)-6, len(self.temperature_data)+1), self.after_extinguish]).T

    def mark_special_points(self):
        """标记特殊点"""
        self.point_positive7 = [7, self.temperature_data[6]]
        self.point_negative7 = [len(self.temperature_data)-6, self.temperature_data[-7]]
        self.point_mean = np.mean([self.point_positive7, self.point_negative7], axis=0)

    def fit_data(self):
        """拟合数据"""
        self.coeffs_before, _ = curve_fit(lambda x, *coeffs: self.polynomial_fit(x, *coeffs),
                                          self.point_before[:, 0], self.point_before[:, 1], p0=[1]*4)
        self.coeffs_during, _ = curve_fit(lambda x, *coeffs: self.polynomial_fit(x, *coeffs),
                                          self.point_during[:, 0], self.point_during[:, 1], p0=[1]*8)
        self.coeffs_after, _ = curve_fit(lambda x, *coeffs: self.polynomial_fit(x, *coeffs),
                                         self.point_after[:, 0], self.point_after[:, 1], p0=[1]*4)

    def polynomial_fit(self, x, *coeffs):
        """多项式拟合函数"""
        return sum(c * x**i for i, c in enumerate(coeffs))

    def piecewise_fit(self, x):
        """分段拟合函数"""
        if x < 8:
            return self.polynomial_fit(x, *self.coeffs_before)
        elif 8 <= x <= len(self.temperature_data)-6:
            return self.polynomial_fit(x, *self.coeffs_during)
        elif len(self.temperature_data)-6 <= x <= len(self.temperature_data):
            return self.polynomial_fit(x, *self.coeffs_after)

    def calculate_slopes(self):
        """计算斜率"""
        x = symbols('x')
        fit_before = sum(c * x**i for i, c in enumerate(self.coeffs_before))
        fit_after = sum(c * x**i for i, c in enumerate(self.coeffs_after))
        derivative_before = diff(fit_before, x)
        derivative_after = diff(fit_after, x)
        self.slope_positive7 = derivative_before.subs(x, 7)
        self.slope_negative7 = derivative_after.subs(x, len(self.temperature_data)-6)

    def calculate_tangent_lines(self):
        """计算切线"""
        self.tangent_positive7 = lambda x: self.slope_positive7 * (x - 7) + self.point_positive7[1]
        self.tangent_negative7 = lambda x: self.slope_negative7 * (x - (len(self.temperature_data)-6)) + self.point_negative7[1]

    def calculate_delta_T(self):
        """计算温度差"""
        x = symbols('x')
        sol = solve(self.polynomial_fit(x, *self.coeffs_during) - self.point_mean[1], x)
        self.point_mean = [sol[0].evalf(), self.point_mean[1]]
        self.T1 = self.tangent_positive7(self.point_mean[0])
        self.T2 = self.tangent_negative7(self.point_mean[0])
        self.delta_T = self.T2 - self.T1

    def plot_results(self):
        """绘制结果图"""
        plt.figure(figsize=(16, 10))
        plt.title(rf'Combustion Heat Analysis - The plot of $\theta - t$ ({self.substance_name})')
        plt.xlabel(r'$t \ / \ \rm{s}$')
        plt.ylabel(r'$\theta \ / \ ^{\circ}C$')

        plt.grid(zorder=0)

        # 使用统一的颜色绘制各阶段的数据点
        plt.scatter(range(1, 8), self.before_ignition, c='blue', marker='o', label='Before ignition', zorder=3)
        plt.scatter(range(8, len(self.temperature_data)-5), self.during_reaction, c='red', marker='o', label='During reaction', zorder=3)
        plt.scatter(range(len(self.temperature_data)-6, len(self.temperature_data)+1), self.after_extinguish, c='green', marker='o', label='After extinguish', zorder=3)

        x_vals = np.linspace(1, len(self.temperature_data)+1, 1000)
        plt.plot(x_vals, [self.piecewise_fit(x) for x in x_vals], 'k-', label='Piecewise fit', zorder=2)

        # 绘制切线和中间点线
        plt.plot([7, 10], [self.tangent_positive7(7), self.tangent_positive7(10)],\
                 'blue', label='Tangent line at Point +7')
        plt.plot([8, 23], [self.tangent_negative7(len(self.temperature_data)-6), \
                           self.tangent_negative7(len(self.temperature_data))], 'green', label='Tangent line at Point -7')
        plt.axvline(self.point_mean[0], color='purple', linestyle='dashed', label='Mean point line')
    
        # 计算并标注点A和点B
        x_A = self.point_mean[0]
        y_A = self.tangent_positive7(x_A)
        plt.scatter(x_A, y_A, color='purple', zorder=4)
        plt.text(x_A, y_A + 0.1, 'A', color='purple', fontsize=12)
    
        x_B = self.point_mean[0]
        y_B = self.tangent_negative7(x_B)
        plt.scatter(x_B, y_B, color='purple', zorder=4)
        plt.text(x_B, y_B - 0.1, 'B', color='purple', fontsize=12)
    
        # 标注其他特殊点
        plt.scatter(*self.point_positive7, color='blue', zorder=3)
        plt.text(self.point_positive7[0], self.point_positive7[1] - 0.1, 'Point +7', color='blue')
        plt.scatter(*self.point_negative7, color='green', zorder=3)
        plt.text(self.point_negative7[0], self.point_negative7[1] + 0.1, 'Point -7', color='green')
        plt.scatter(*self.point_mean, color='purple', zorder=3)
        plt.text(self.point_mean[0] + 0.5, self.point_mean[1], 'Mean point', color='purple', 
                 verticalalignment='center', horizontalalignment='left',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
        plt.legend(loc='upper left')
        plt.savefig(fr'./拟合图结果/{self.arc_name}', dpi=300)
        plt.close()

    def print_results(self):
        """打印结果"""
        print(f"Mean point: {tuple(self.point_mean)}")
        print(f"T_1: {self.T1} °C")
        print(f"T_2: {self.T2} °C")
        print(f"ΔT: {self.delta_T} °C")

    def process_data(self):
        """处理数据的主要流程"""
        self.segment_data()
        self.create_point_data()
        self.mark_special_points()
        self.fit_data()
        self.calculate_slopes()
        self.calculate_tangent_lines()
        self.calculate_delta_T()
        self.plot_results()
        self.print_results()

    def compress_results(self):
        """压缩结果文件"""
        dir_to_zip = r'./拟合图结果'
        dir_to_save = r'./拟合图结果.zip'

        with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dir_to_zip):
                for file in files:
                    file_dir = os.path.join(root, file)
                    arc_name = os.path.relpath(file_dir, dir_to_zip)
                    zipf.write(file_dir, arc_name)

        print(f'压缩完成，文件保存为: {dir_to_save}')

# 主程序
file_dir = r'./燃烧热的测定原始记录表(非).xlsx'
sheet_names = pd.ExcelFile(file_dir).sheet_names

ans_benzoic_acid_origin_data_df = pd.read_excel(file_dir, sheet_name=sheet_names[0], header=None)
ans_naphthalene_origin_data_df = pd.read_excel(file_dir, sheet_name=sheet_names[1], header=None)

data_benzoic_acid_temperature = np.array(ans_benzoic_acid_origin_data_df.iloc[3:, 2].values, dtype=float)
data_naphthalene_temperature = np.array(ans_naphthalene_origin_data_df.iloc[3:, 2].values, dtype=float)

benzoic_acid_analysis = CombustionHeatAnalysis(data_benzoic_acid_temperature, 'Benzoic acid', '苯甲酸燃烧热的测定')
naphthalene_analysis = CombustionHeatAnalysis(data_naphthalene_temperature, 'Naphthalene', '萘燃烧热的测定')

# 压缩结果
benzoic_acid_analysis.compress_results()
