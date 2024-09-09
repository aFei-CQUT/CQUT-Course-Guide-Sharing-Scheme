# This project is created by aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
#   About aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

def calculate_electrode_potential(file_path):
    # 读取Excel文件
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    df = pd.read_excel(file_path, sheet_name=sheet_names[0], header=None)

    # 电势原始数据
    e = df.iloc[1:, 1:].values

    # 数据预处理:计算平均值
    e_mean = np.mean(e, axis=0)

    # 创建数据表格
    title = ["序号/项目", "e_Zn_Hg", "e_Cu_Zn", "e_Cu_Hg"]
    data_preprocess_array = np.column_stack((np.arange(1, e.shape[0] + 1), e))
    data_preprocess_array = np.vstack((data_preprocess_array, np.append(["平均值"], e_mean)))

    # 创建DataFrame
    ans_data_preprocess_df = pd.DataFrame(data_preprocess_array, columns=title)

    # 常量
    T = 16.6 + 273.15
    R = 8.314
    F = 96485
    a_Cu = 0.1000 * 0.150
    a_Zn = 0.1000 * 0.150
    α_Cu = -0.000016
    β_Cu = 0
    α_Zn = 0.0001
    β_Zn = 0.62 * 10**-6
    φ_sec = 0.2415 - 7.61 * 10**-4 * (T - 298)

    # 计算过程
    φ_Zn = φ_sec - e_mean[0]
    φ_Cu = φ_sec + e_mean[2]

    φ_theta_Cu = φ_Cu + (R * T) / (2 * F) * np.log(1 / a_Cu)
    φ_theta_Zn = φ_Zn + (R * T) / (2 * F) * np.log(1 / a_Zn)

    φ_theta_Cu_298K = φ_theta_Cu - α_Cu * (T - 298) - 1/2 * β_Cu * (T - 298)**2
    φ_theta_Zn_298K = φ_theta_Zn - α_Zn * (T - 298) - 1/2 * β_Zn * (T - 298)**2

    e_measure = e_mean[1]
    e_standard = φ_theta_Cu_298K - φ_theta_Zn_298K
    Er = np.abs(e_measure - e_standard) / e_standard * 100

    # 创建 ans_parameter_values_df 结构
    ans_parameter_values_df = pd.DataFrame([
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

    return ans_data_preprocess_df, ans_parameter_values_df

# 调用函数
file_path = r'./电极制备及原电池电动势原始记录表(非).xlsx'
ans_data_preprocess_df, ans_parameter_values_df = calculate_electrode_potential(file_path)

# 图表输出结果
import matplotlib.pyplot as plt

# 设定绘图参数
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# 第一个表格：电势原始数据处理结果
table1 = ax1.table(cellText=ans_data_preprocess_df.values,
                   colLabels=ans_data_preprocess_df.columns,
                   cellLoc='center', loc='upper center')

table1.auto_set_font_size(False)
table1.set_fontsize(10)
table1.scale(1.2, 1.2)

ax1.axis('off')
ax1.set_title('电势原始数据处理结果', fontsize=12)

# 第二个表格：计算结果
table2 = ax2.table(cellText=ans_parameter_values_df.values,
                   colLabels=ans_parameter_values_df.columns,
                   cellLoc='center', loc='upper center')

table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1.2, 1.2)

ax2.axis('off')
ax2.set_title('电势计算结果', fontsize=12)

# 调整子图之间的间距
plt.subplots_adjust(hspace=-0.3)

plt.savefig(r'./拟合图结果/1.png', dpi=300)

# 显示图表
plt.show()

# 拟合图结果压缩
import zipfile
import os

# 待压缩的文件路径
dir_to_zip = r'./拟合图结果'

# 压缩后的保存路径
dir_to_save = r'./拟合图结果.zip'

# 创建ZipFile对象
with zipfile.ZipFile(dir_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # 遍历目录
    for root, dirs, files in os.walk(dir_to_zip):
        for file in files:
            # 创建相对文件路径并将其写入zip文件
            file_dir = os.path.join(root, file)
            arc_name = os.path.relpath(file_dir, dir_to_zip)
            zipf.write(file_dir, arc_name)

print(f'压缩完成，文件保存为: {dir_to_save}')
