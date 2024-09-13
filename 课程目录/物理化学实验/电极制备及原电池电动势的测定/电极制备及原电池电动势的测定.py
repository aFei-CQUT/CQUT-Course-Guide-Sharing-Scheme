import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os

class ElectrodeCalculator:
    """
    电极电势计算器类

    用于计算电极电势并生成相关图表和数据。
    """

    def __init__(self, file_path):
        """
        初始化电极计算器

        参数:
        file_path (str): Excel文件的路径
        """
        self.file_path = file_path
        self.ans_data_preprocess_df = None
        self.ans_parameter_values_df = None

    def calculate_electrode_potential(self):
        """
        计算电极电势

        读取Excel文件,处理数据,计算各种电势值。
        """
        # 读取Excel文件
        xls = pd.ExcelFile(self.file_path)
        sheet_names = xls.sheet_names
        df = pd.read_excel(self.file_path, sheet_name=sheet_names[0], header=None)

        # 电势原始数据
        e = df.iloc[1:, 1:].values.astype(float)
        e = np.round(e, 6)  # 保留6位小数

        # 数据预处理:计算平均值
        e_mean = np.round(np.mean(e, axis=0), 6)  # 保留6位小数

        # 创建数据表格
        title = ["序号/项目", "e_Zn_Hg", "e_Cu_Zn", "e_Cu_Hg"]
        data_preprocess_array = np.column_stack((np.arange(1, e.shape[0] + 1), e))
        data_preprocess_array = np.vstack((data_preprocess_array, np.append(["平均值"], e_mean)))

        # 创建DataFrame
        self.ans_data_preprocess_df = pd.DataFrame(data_preprocess_array, columns=title)

        # 常量
        T = round(16.6 + 273.15, 6)
        R = 8.314000
        F = 96485.000000
        a_Cu = round(0.1000 * 0.150, 6)
        a_Zn = round(0.1000 * 0.150, 6)
        α_Cu = -0.000016
        β_Cu = 0.000000
        α_Zn = 0.000100
        β_Zn = round(0.62 * 10**-6, 6)
        φ_sec = round(0.2415 - 7.61 * 10**-4 * (T - 298), 6)

        # 计算过程
        φ_Zn = round(φ_sec - e_mean[0], 6)
        φ_Cu = round(φ_sec + e_mean[2], 6)

        φ_theta_Cu = round(φ_Cu + (R * T) / (2 * F) * np.log(1 / a_Cu), 6)
        φ_theta_Zn = round(φ_Zn + (R * T) / (2 * F) * np.log(1 / a_Zn), 6)

        φ_theta_Cu_298K = round(φ_theta_Cu - α_Cu * (T - 298) - 1/2 * β_Cu * (T - 298)**2, 6)
        φ_theta_Zn_298K = round(φ_theta_Zn - α_Zn * (T - 298) - 1/2 * β_Zn * (T - 298)**2, 6)

        e_measure = round(e_mean[1], 6)
        e_standard = round(φ_theta_Cu_298K - φ_theta_Zn_298K, 6)
        Er = round(np.abs(e_measure - e_standard) / e_standard * 100, 6)

        # 创建 ans_parameter_values_df 结构
        self.ans_parameter_values_df = pd.DataFrame([
            {"Description": "甘汞电极电势", "Variable": "φ_sec", "Value": φ_sec},
            {"Description": "锌极电势", "Variable": "φ_Zn", "Value": φ_Zn},
            {"Description": "铜极电势", "Variable": "φ_Cu", "Value": φ_Cu},
            {"Description": "铜的标准电极电势", "Variable": "φ_theta_Cu", "Value": φ_theta_Cu},
            {"Description": "转化为298K时的铜的标准电极电势", "Variable": "φ_theta_Cu_298K", "Value": φ_theta_Cu_298K},
            {"Description": "锌的标准电极电势", "Variable": "φ_theta_Zn", "Value": φ_theta_Zn},
            {"Description": "转化为298K时的锌的标准电极电势", "Variable": "φ_theta_Zn_298K", "Value": φ_theta_Zn_298K},
            {"Description": "铜锌原电池电动势测量值", "Variable": "e_measure", "Value": e_measure},
            {"Description": "计算所得铜锌原电池标准电动势", "Variable": "e_standard", "Value": e_standard},
            {"Description": "相对误差", "Variable": "Er", "Value": Er}
        ])

    def plot_results(self, save_path):
        """
        绘制结果图表

        参数:
        save_path (str): 保存图表的路径
        """
        # 设定绘图参数
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False

        # 创建画布和子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        # 第一个表格：电势原始数据处理结果
        table1 = ax1.table(cellText=self.ans_data_preprocess_df.values,
                           colLabels=self.ans_data_preprocess_df.columns,
                           cellLoc='center', loc='upper center')

        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1.2, 1.2)

        ax1.axis('off')
        ax1.set_title('电势原始数据处理结果', fontsize=12)

        # 第二个表格：计算结果
        table2 = ax2.table(cellText=self.ans_parameter_values_df.values,
                           colLabels=self.ans_parameter_values_df.columns,
                           cellLoc='center', loc='upper center')

        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1.2, 1.2)

        ax2.axis('off')
        ax2.set_title('电势计算结果', fontsize=12)

        # 调整子图之间的间距
        plt.subplots_adjust(hspace=-0.3)

        plt.savefig(save_path, dpi=500)
        plt.close()

    @staticmethod
    def compress_results(dir_to_zip, dir_to_save):
        """
        压缩结果文件

        参数:
        dir_to_zip (str): 待压缩的文件夹路径
        dir_to_save (str): 保存压缩文件的路径
        """
        with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dir_to_zip):
                for file in files:
                    file_dir = os.path.join(root, file)
                    arc_name = os.path.relpath(file_dir, dir_to_zip)
                    zipf.write(file_dir, arc_name)

        print(f'压缩完成，文件保存为: {dir_to_save}')

def main():
    """
    主函数
    """
    file_path = r'./电极制备及原电池电动势原始记录表(非).xlsx'
    calculator = ElectrodeCalculator(file_path)
    calculator.calculate_electrode_potential()
    calculator.plot_results(r'./拟合图结果/1.png')
    calculator.compress_results(r'./拟合图结果', r'./拟合图结果.zip')

if __name__ == "__main__":
    main()
