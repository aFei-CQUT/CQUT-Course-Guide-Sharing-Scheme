import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import zipfile
import os

class SaturatedVaporPressureAnalysis:
    """
    静态法测量液体饱和蒸气压的分析类
    
    该类用于处理和分析液体饱和蒸气压的实验数据，包括数据加载、处理、
    统计分析、线性回归、热力学性质计算、误差分析、结果可视化等功能。
    """

    def __init__(self, file_path):
        """
        初始化分析对象
        
        :param file_path: Excel文件路径，包含原始实验数据
        """
        self.file_path = file_path
        self.R = 8.314  # 气体常数
        self.standard_boiling_point = 76.8  # 标准沸点
        self.standard_vap_enthalpy = 30.81 * 1000  # 标准摩尔蒸发焓
        self.load_data()
        self.process_data()
        self.perform_analysis()

    def load_data(self):
        """从Excel文件加载原始数据"""
        excel_data = pd.ExcelFile(self.file_path)
        self.df = pd.read_excel(excel_data, sheet_name=0)

    def process_data(self):
        """处理原始数据，计算所需的物理量"""
        self.temperature_celsius = self.df['T/℃'].values
        self.temperature_kelvin = self.temperature_celsius + 273.15
        self.p0 = self.df['p0/kPa'].values
        self.p_watch = self.df['p表/kPa'].values
        self.p = self.p0 - self.p_watch
        self.lgp = np.log10(self.p)
        self.inv_temperature = 1 / self.temperature_kelvin

    def perform_analysis(self):
        """执行数据分析的主要流程"""
        self.calculate_statistics()
        self.perform_linear_regression()
        self.calculate_thermodynamic_properties()
        self.calculate_errors()

    def calculate_statistics(self):
        """计算统计量，如方差和相关系数"""
        self.variance_lgp = np.var(self.lgp)
        self.variance_temp_inv = np.var(self.inv_temperature)
        self.correlation_coefficient = np.corrcoef(self.lgp, self.inv_temperature)[0, 1]

    def perform_linear_regression(self):
        """执行线性回归分析"""
        self.slope, self.intercept, self.r_value, self.p_value, self.std_err = linregress(self.inv_temperature, self.lgp)
        self.fit_polynomial = self.slope * self.inv_temperature + self.intercept
        self.residuals = self.lgp - self.fit_polynomial
        self.residuals_variance = np.var(self.residuals)

    def calculate_thermodynamic_properties(self):
        """计算热力学性质，如摩尔蒸发焓和沸点"""
        self.delta_vap_Hm = -self.slope * self.R * 2.303
        self.boiling_point = 1 / (1 / (self.standard_boiling_point + 273.15) - (self.R * np.log(101.325 / 101.10)) / self.delta_vap_Hm) - 273.15

    def calculate_errors(self):
        """计算相对误差"""
        self.relative_error_Hm = ((self.standard_vap_enthalpy - self.delta_vap_Hm) * 100) / self.standard_vap_enthalpy
        self.relative_error_boiling_point = ((self.boiling_point - self.standard_boiling_point) * 100) / self.standard_boiling_point

    def create_results_dataframe(self):
        """创建包含所有结果的DataFrame"""
        return pd.DataFrame({
            "T/℃": self.temperature_celsius,
            "T/K": self.temperature_kelvin,
            "p0/kPa": self.p0,
            "p表/kPa": self.p_watch,
            "p/kPa": self.p,
            "lgp/kPa": self.lgp,
            "1/T/K^-1": self.inv_temperature
        })

    def print_results(self):
        """打印分析结果"""
        print('原始数据:')
        print(self.df)
        print("\n所有结果:")
        print(self.create_results_dataframe())
        print(f"\nlgp的方差: {self.variance_lgp}")
        print(f"1/T的方差: {self.variance_temp_inv}")
        print(f"相关系数: {self.correlation_coefficient}")
        print(f"残差方差: {self.residuals_variance}")
        print(f"拟合多项式: y = {self.slope}x + {self.intercept}")
        print(f"斜率A: {self.slope}")
        print(f"摩尔蒸发焓: {self.delta_vap_Hm} J/mol")
        print(f"计算得到的沸点: {self.boiling_point} ℃")
        print(f"摩尔蒸发焓相对误差: {self.relative_error_Hm} %")
        print(f"正常沸点相对误差: {self.relative_error_boiling_point} %")

    def plot_results(self):
        """绘制结果图并保存"""
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({'font.size': 12})

        plt.gca().spines['top'].set_linewidth(1.25)
        plt.gca().spines['bottom'].set_linewidth(1.25)
        plt.gca().spines['left'].set_linewidth(1.25)
        plt.gca().spines['right'].set_linewidth(1.25)

        plt.scatter(self.inv_temperature, self.lgp, color='red', label='数据点')
        plt.plot(self.inv_temperature, self.fit_polynomial, color='black', label='拟合直线')
        plt.title('$lg(p) - \\frac{1}{T}$ 图')
        plt.xlabel('$\\frac{1}{T}$    单位：$K^{-1}$')
        plt.ylabel('$lg(p)$    单位：$Kpa$')
        plt.legend()

        plt.grid(True, which='both', linestyle='-', linewidth=1)
        plt.minorticks_on()

        plt.savefig(r'./拟合图结果/1.png', dpi=300)
        plt.show()

    def compress_results(self):
        """压缩结果图文件"""
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
file_dir = './静态法测液体的饱和蒸气压原始记录表(非).xlsx'
analysis = SaturatedVaporPressureAnalysis(file_dir)
analysis.print_results()
analysis.plot_results()
analysis.compress_results()
