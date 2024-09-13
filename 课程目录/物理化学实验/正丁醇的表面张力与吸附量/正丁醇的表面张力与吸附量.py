import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
import os

class SurfaceTensionAnalyzer:
    """
    表面张力和吸附量分析类

    用于分析正丁醇水溶液的表面张力和吸附量数据。
    包括数据导入、计算、拟合和可视化功能。
    支持多阶泰勒拟合。
    """

    def __init__(self, file_path, degree_fit1=1, degree_fit2=1):
        """
        初始化分析器

        参数:
        file_path (str): Excel文件的路径
        degree_fit1 (int): 表面张力拟合的阶数，默认拟合阶数为一阶
        degree_fit2 (int): 吸附量拟合的阶数，默认拟合阶数为一阶
        """
        self.file_path = file_path
        self.degree_fit1 = degree_fit1
        self.degree_fit2 = degree_fit2
        self.df1 = None
        self.df2 = None
        self.K = None
        self.concentrations = None
        self.σ_Butanol_array = None
        self.gamma_array = None
        self.taylor_fit1 = None
        self.taylor_fit2 = None

    def load_data(self):
        """加载Excel文件中的数据"""
        xls = pd.ExcelFile(self.file_path)
        sheet_names = xls.sheet_names
        self.df1 = pd.read_excel(self.file_path, sheet_name=sheet_names[0], header=None)
        self.df2 = pd.read_excel(self.file_path, sheet_name=sheet_names[1], header=None)

    def calculate_K(self):
        """计算标定系数K"""
        ΔpH2O_array = np.reshape(np.array(self.df1.iloc[2,2:].values, dtype=float), (1, 3))
        ΔpH2O_mean = np.mean(ΔpH2O_array)
        σ = 0.07118
        self.K = σ / ΔpH2O_mean

    def process_data(self):
        """处理数据,计算表面张力和吸附量"""
        self.concentrations = np.array(self.df2.iloc[2:, 1].values, dtype=float)
        Δp_Butanol_array = np.array(self.df2.iloc[2:, 2:].values, dtype=float)
        mean_values = np.mean(Δp_Butanol_array, axis=1)
        self.σ_Butanol_array = self.K * mean_values

        # 多阶泰勒拟合
        coefficients1 = np.polyfit(self.concentrations, self.σ_Butanol_array, self.degree_fit1)
        self.taylor_fit1 = np.poly1d(coefficients1)

        # 计算吸附量
        derivative = np.polyder(self.taylor_fit1)
        slope_array = np.polyval(derivative, self.concentrations)
        self.gamma_array = -(self.concentrations * slope_array) / (8.314 * (30 + 273.15))

        # 吸附量的多阶泰勒拟合
        coefficients2 = np.polyfit(self.concentrations, self.gamma_array, self.degree_fit2)
        self.taylor_fit2 = np.poly1d(coefficients2)

    def create_data_table(self):
        """创建数据表格"""
        Δp_Butanol_array = np.array(self.df2.iloc[2:, 2:].values, dtype=float)
        mean_values = np.mean(Δp_Butanol_array, axis=1)
        derivative = np.polyder(self.taylor_fit1)
        slope_array = np.polyval(derivative, self.concentrations)

        ans_data_dict = {
            "Concentration (mol/L)": self.concentrations,
            "ΔP_1 (mmH2O)": Δp_Butanol_array[:, 0],
            "ΔP_2 (mmH2O)": Δp_Butanol_array[:, 1],
            "ΔP_3 (mmH2O)": Δp_Butanol_array[:, 2],
            "ΔP_mean": mean_values,
            "σ_Butanol array": self.σ_Butanol_array,
            "Slope array": slope_array,
            "Γ array": self.gamma_array
        }
        return pd.DataFrame(ans_data_dict)

    def plot_results(self):
        """绘制结果图表"""
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(14, 6))

        concentrations_dense = np.linspace(self.concentrations.min(), self.concentrations.max(), 100)

        plt.subplot(1, 2, 1)
        plt.scatter(self.concentrations, self.σ_Butanol_array, color='red', label='数据')
        plt.plot(concentrations_dense, self.taylor_fit1(concentrations_dense), '-', color='blue', label=f'{self.degree_fit1}阶Taylor拟合')
        plt.xlabel('$c$    单位:mol/L')
        plt.ylabel('$\\sigma$    单位:N/m')
        plt.title('表面张力数据及其Taylor拟合')
        plt.grid(True, which='both')
        plt.minorticks_on()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(concentrations_dense, self.taylor_fit2(concentrations_dense), '-', color='blue', label=f'{self.degree_fit2}阶Taylor拟合')
        plt.scatter(self.concentrations, self.gamma_array, color='red', label='数据')
        plt.xlabel('$c$    单位:mol/L')
        plt.ylabel('$\\Gamma$    单位:mol/m^2')
        plt.title('吸附量数据及其Taylor拟合')
        plt.grid(True, which='both')
        plt.minorticks_on()
        plt.legend()

        plt.tight_layout()
        plt.savefig(r'./拟合图结果/正丁醇的表面张力与吸附量.png', dpi=300)
        plt.show()

    def compress_results(self):
        """压缩结果图片"""
        dir_to_zip = r'./拟合图结果'
        dir_to_save = r'./拟合图结果.zip'

        with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dir_to_zip):
                for file in files:
                    file_dir = os.path.join(root, file)
                    arc_name = os.path.relpath(file_dir, dir_to_zip)
                    zipf.write(file_dir, arc_name)

        print(f'压缩完成，文件保存为: {dir_to_save}')

    def run_analysis(self):
        """运行完整的分析流程"""
        self.load_data()
        self.calculate_K()
        self.process_data()
        data_table = self.create_data_table()
        print("\n数据表:")
        print(data_table.to_string(index=False))
        self.plot_results()
        self.compress_results()
        print(f"\n\n表面张力 - Fitted Polynomial 1 (阶数: {self.degree_fit1}):\n {self.taylor_fit1}")
        print(f"\n\n表面吸附 - Fitted Polynomial 2 (阶数: {self.degree_fit2}):\n {self.taylor_fit2}")

# 使用示例
if __name__ == "__main__":
    analyzer = SurfaceTensionAnalyzer(r'./正丁醇的表面张力与吸附量原始记录表(非).xlsx', degree_fit1=1, degree_fit2=1)
    analyzer.run_analysis()
