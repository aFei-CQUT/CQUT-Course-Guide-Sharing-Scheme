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
import matplotlib.pyplot as plt
import pandas as pd

# 导入数据
file_dir = r'./正丁醇的表面张力与吸附量原始记录表(非).xlsx'
xls = pd.ExcelFile(file_dir)
sheet_names = xls.sheet_names

df1 = pd.read_excel(file_dir, sheet_name=sheet_names[0], header=None)
df2 = pd.read_excel(file_dir, sheet_name=sheet_names[1], header=None)

# 选取纯水的特定数据
ΔpH2O_array = np.reshape(np.array(df1.iloc[2,2:].values, dtype=float), (1, 3))
ΔpH2O_mean = np.mean(ΔpH2O_array)

# 由纯水标定K
σ = 0.07118
K = σ / ΔpH2O_mean

# =============================================================================
# 补充的数据、公式
# 
# ρ = 0.81
# M = 74.12
# c = ρ/M
# 
# =============================================================================

# 选取正丁醇水溶液的特定数据
concentrations = np.array(df2.iloc[2:, 1].values, dtype=float)
Δp_Butanol_array = np.array(df2.iloc[2:, 2:].values, dtype=float)

# 计算
mean_values = np.mean(Δp_Butanol_array, axis=1)
σ_Butanol_array = K * mean_values

# Taylor 级数拟合 1
degree_fit1 = 2
coefficients1 = np.polyfit(concentrations, σ_Butanol_array, degree_fit1)
taylor_fit1 = np.poly1d(coefficients1)

# 导数和 \[CapitalGamma\] 的计算
derivative = np.polyder(taylor_fit1)
slope_array = np.polyval(derivative, concentrations)
gamma_array = -(concentrations * slope_array) / (8.314 * (30 + 273.15))

# Taylor 级数拟合 2
degree_fit2 = 3
coefficients2 = np.polyfit(concentrations, gamma_array, degree_fit2)
taylor_fit2 = np.poly1d(coefficients2)

# 生成更密集的浓度点
concentrations_dense = np.linspace(concentrations.min(), concentrations.max(), 100)

# 创建数据表格
ans_data_dict = {
    "Concentration (mol/L)": concentrations,
    "ΔP_1 (mmH2O)": Δp_Butanol_array[:, 0],
    "ΔP_2 (mmH2O)": Δp_Butanol_array[:, 1],
    "ΔP_3 (mmH2O)": Δp_Butanol_array[:, 2],
    "ΔP_mean": mean_values,
    "σ_Butanol array": σ_Butanol_array,
    "Slope array": slope_array,
    "Γ array": gamma_array
}

ans_data_df = pd.DataFrame(ans_data_dict)

# 显示数据表格
print("\n数据表:")
print(ans_data_df.to_string(index=False))

# 设置绘图参数
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 绘制图形
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(concentrations, σ_Butanol_array, color='red', label='数据')
plt.plot(concentrations_dense, taylor_fit1(concentrations_dense), '-', color='blue', label='Taylor 级数拟合')
plt.xlabel('$c$    单位:mol/L')
plt.ylabel('$\sigma$    单位:N/m')
plt.title('数据及其 Taylor 级数拟合')
plt.grid(True, which='both')
plt.minorticks_on()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(concentrations_dense, taylor_fit2(concentrations_dense), '-', color='blue', label='Taylor 级数拟合')
plt.scatter(concentrations, gamma_array, color='red', label='数据')
plt.xlabel('$c$    单位:mol/L')
plt.ylabel('$\Gamma$    单位:N/m')
plt.title('数据及其 Taylor 级数拟合')
plt.grid(True, which='both')
plt.minorticks_on()
plt.legend()

plt.tight_layout()

plt.savefig(r'./拟合图结果/正丁醇的表面张力与吸附量.png', dpi=300)

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

# 显示拟合多项式
print(f"\n\n表面张力 - Fitted Polynomial 1:\n {np.poly1d(taylor_fit1)}")
print(f"\n\n表面吸附 - Fitted Polynomial 2:\n {np.poly1d(taylor_fit2)}")