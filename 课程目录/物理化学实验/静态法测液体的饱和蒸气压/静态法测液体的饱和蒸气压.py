# This project is created by aFei-CQUT
# ------------------------------------------------------------------------------------------------------------------------------------
#   About aFei-CQUT
# - Interests&Hobbies: Programing,  ChatGPT,  Reading serious books,  Studying academical papers.
# - CurrentlyLearning: Mathmodeling，Python and Mathmatica (preparing for National College Mathematical Contest in Modeling).
# - Email:2039787966@qq.com
# - Pronouns: Chemical Engineering, Computer Science, Enterprising, Diligent, Hard-working, Sophomore,Chongqing Institute of Technology,
# - Looking forward to collaborating on experimental data processing of chemical engineering principle
# ------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 从Excel文件读取数据
file_dir = './静态法测液体的饱和蒸气压原始记录表(非).xlsx'
excel_data = pd.ExcelFile(file_dir)

# 原始数据记录:第一个工作表中的数据
df = pd.read_excel(excel_data, sheet_name=0)

# 原始数据:提取原始数据
temperature_array_float64_1 = df['T/℃'].values
p0_array_float64 = df['p0/kPa'].values
p_watch_array_float64 = df['p表/kPa'].values

# 数据预处理:为待拟合数据做准备
temperature_array_float64_2 = temperature_array_float64_1 + 273.15
p_array_float64 = p0_array_float64 - p_watch_array_float64

# 数据处理:计算待拟合数据
lgp_array_float64 = np.log10(p_array_float64)
temperature_minus_one_array_float64 = 1 / temperature_array_float64_2

ans_df = pd.DataFrame({
    "T/℃": temperature_array_float64_1,
    "T/K": temperature_array_float64_2,
    "p0/kPa": p0_array_float64,
    "p表/kPa": p_watch_array_float64,
    "p/kPa": p_array_float64,
    "lgp/kPa": lgp_array_float64,
    "1/T/K^-1": temperature_minus_one_array_float64
})

# 计算方差
variance_lgp = np.var(lgp_array_float64)
variance_temp_inv = np.var(temperature_minus_one_array_float64)

# 计算相关系数
correlation_coefficient = np.corrcoef(lgp_array_float64, temperature_minus_one_array_float64)[0, 1]

# 线性拟合
slope, intercept, r_value, p_value, std_err = linregress(temperature_minus_one_array_float64, lgp_array_float64)
fit_polynomial = slope * temperature_minus_one_array_float64 + intercept

# 计算残差和残差方差
residuals = lgp_array_float64 - fit_polynomial
residuals_variance = np.var(residuals)

# 计算摩尔蒸发焓变和沸点
R = 8.314
delta_vap_Hm = -slope * R * 2.303
boiling_point = 1 / (1 / (76.8 + 273.15) - (R * np.log(101.325 / 101.10)) / delta_vap_Hm) - 273.15

# 标准摩尔蒸发焓和误差计算
delta_vap_Hm0 = 30.81 * 1000
t0 = 76.8
relative_error_Hm = ((delta_vap_Hm0 - delta_vap_Hm) * 100) / delta_vap_Hm0
relative_error_boiling_point = ((boiling_point - t0) * 100) / t0

# 输出结果
print('原始数据记录:')
print()
print(df)
print()
print("所有结果展示:")
print()
print(ans_df)
print()
print(f"lgp的方差: {variance_lgp}")
print(f"1/T的方差: {variance_temp_inv}")
print(f"相关系数: {correlation_coefficient}")
print(f"拟合残差方差: {residuals_variance}")
print(f"拟合多项式: y = {slope}x + {intercept}")
print(f"斜率A: {slope}")
print(f"摩尔蒸发焓: {delta_vap_Hm} J/mol")
print(f"计算得沸点: {boiling_point} ℃")
print(f"摩尔蒸发焓相对误差: {relative_error_Hm} %")
print(f"正常沸点相对误差: {relative_error_boiling_point} %")

# 设定绘图参数
plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})

# 绘图
plt.gca().spines['top'].set_linewidth(1.25)
plt.gca().spines['bottom'].set_linewidth(1.25)
plt.gca().spines['left'].set_linewidth(1.25)
plt.gca().spines['right'].set_linewidth(1.25)

plt.scatter(temperature_minus_one_array_float64, lgp_array_float64, color='red', label='数据点')
plt.plot(temperature_minus_one_array_float64, fit_polynomial, color='black', label='拟合直线')
plt.title('$lg(p) - \\frac{1}{T}$')
plt.xlabel('$\\frac{1}{T}$    单位：$K^{-1}$')
plt.ylabel('$lg(p)$    单位：$Kpa$')
plt.legend()

plt.grid(True, which='both', linestyle='-', linewidth=1)
plt.minorticks_on()

plt.savefig(r'./拟合图结果/1.png', dpi=300)

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
