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
from scipy.optimize import curve_fit
from sympy import symbols, diff, solve

# 函数化处理不同苯甲酸和苯的燃烧热数据

def process_data(temperature_data, substance_name, arc_name):
    # 定义多项式拟合函数
    def polynomial_fit(x, *coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))

    # 温度数据分段
    
    # 前7个数据
    temperature_burn_before = temperature_data[:7]
    # 中间段数据（个数不定）
    temperature_burn_during = temperature_data[7:len(temperature_data)-6]
    # 后7个数据
    temperature_burn_after = temperature_data[len(temperature_data)-7:]

    # 数据坐标化，即用二维坐标点(x，y)的形式表达，便于拟合
    
    # 前7个数据点
    point_temperature_burn_before = np.array([range(1, 8), temperature_burn_before]).T
    # 中间段数据点（个数不定）
    point_temperature_burn_during = np.array([range(8, len(temperature_data)-5), temperature_burn_during]).T
    # 后7个数据点
    point_temperature_burn_after = np.array([range(len(temperature_data)-6, len(temperature_data)+1), temperature_burn_after]).T
    
    # 特殊点标记
    
    # 正数第7个数据点
    point_positive7 = [7, temperature_data[6]]
    # 倒数第7个数据点
    point_negetive7 = [len(temperature_data)-6, temperature_data[-7]]
    # 正数第7个数据点和倒数第7个数据点的中点
    point_mean = np.mean([point_positive7, point_negetive7], axis=0)

    # 拟合阶数
    
    # 燃烧前泰勒拟合阶数
    degree_before = 3
    # 燃烧时泰勒拟合阶数
    degree_during = 7
    # 燃烧后泰勒拟合阶数
    degree_after = 3

    # 拟合参数
    
    # 燃烧前泰勒拟合参数
    coeffs_before, _ = curve_fit(lambda x, *coeffs: polynomial_fit(x, *coeffs),
                                 point_temperature_burn_before[:, 0], point_temperature_burn_before[:, 1], p0=[1]*(degree_before+1))
    # 燃烧时泰勒拟合参数
    coeffs_during, _ = curve_fit(lambda x, *coeffs: polynomial_fit(x, *coeffs),
                                 point_temperature_burn_during[:, 0], point_temperature_burn_during[:, 1], p0=[1]*(degree_during+1))
    # 燃烧后泰勒拟合参数
    coeffs_after, _ = curve_fit(lambda x, *coeffs: polynomial_fit(x, *coeffs),
                                point_temperature_burn_after[:, 0], point_temperature_burn_after[:, 1], p0=[1]*(degree_after+1))

    # 整合燃烧过程的分段函数
    def piecewise_fit(x):
        # 燃烧前
        if x < 8:
            return polynomial_fit(x, *coeffs_before)
        # 燃烧时
        elif 8 <= x <= len(temperature_data)-6:
            return polynomial_fit(x, *coeffs_during)
        # 燃烧后
        elif len(temperature_data)-6 <= x <= len(temperature_data) :
            return polynomial_fit(x, *coeffs_after)

    # 对拟合得到得泰勒级表达式求导数，得到在特殊点处的斜率
    
    # 自变量符号x
    x = symbols('x')
    # 燃烧前得到的泰勒级拟合函数式
    fit_temperature_burn_before = sum(c * x**i for i, c in enumerate(coeffs_before))
    # 求得的燃烧前导数表达式
    derivative_temperature_burn_before = diff(fit_temperature_burn_before, x)
    # 燃烧后得到的泰勒级拟合函数式
    fit_temperature_burn_after = sum(c * x**i for i, c in enumerate(coeffs_after))
    # 求得的燃烧后导数表达式
    derivative_temperature_burn_after = diff(fit_temperature_burn_after, x)

    # 由导数表达式计算斜率
    
    # 求得的正数第7个点的斜率
    slope_point_positive7 = derivative_temperature_burn_before.subs(x, 7)
    # 求得的倒数第7个点的斜率
    slope_point_negetive7 = derivative_temperature_burn_after.subs(x, len(temperature_data)-6)

    # 点斜式求切线
    
    # 求正数第7个点处的切线
    tangent_point_positive7 = lambda x: slope_point_positive7 * (x - 7) + point_positive7[1]
    # 求倒数第7个点处的切线
    tangent_point_negetive7 = lambda x: slope_point_negetive7 * (x - (len(temperature_data)-6)) + point_negetive7[1]

    # 计算ΔT，此处取切线延长线上下的平均值，要用到拟合函数，或者作图得到
    sol = solve(polynomial_fit(x, *coeffs_during) - point_mean[1], x)
    point_mean = [sol[0].evalf(), point_mean[1]]
    T1 = tangent_point_positive7(point_mean[0])
    T2 = tangent_point_negetive7(point_mean[0])
    delta_T = T2 - T1

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(point_temperature_burn_before[:, 0], point_temperature_burn_before[:, 1], 'ro', label='燃烧前')
    plt.plot(point_temperature_burn_during[:, 0], point_temperature_burn_during[:, 1], 'go', label='燃烧时')
    plt.plot(point_temperature_burn_after[:, 0], point_temperature_burn_after[:, 1], 'bo', label='燃烧后')

    x_vals = np.linspace(1, len(temperature_data)+1, 1000)
    plt.plot(x_vals, [piecewise_fit(x) for x in x_vals], 'k-', label='分段拟合')

    plt.plot([7, 10], [tangent_point_positive7(7), tangent_point_positive7(10)], 'r--')
    plt.plot([8, 23], [tangent_point_negetive7(len(temperature_data)-6), tangent_point_negetive7(len(temperature_data))], 'b--')

    plt.scatter(*point_positive7, color='red')
    plt.text(point_positive7[0], point_positive7[1] - 0.1, '正数第 7 个数据点', color='red')
    plt.scatter(*point_negetive7, color='blue')
    plt.text(point_negetive7[0], point_negetive7[1] + 0.1, '倒数第 7 个数据点', color='blue')
    plt.axvline(point_mean[0], color='orange', linestyle='dashed')
    plt.scatter(*point_mean, color='green')
    plt.text(point_mean[0], point_mean[1] + 0.3, '中点值', color='green')

    
    plt.legend()
    plt.xlabel('$t$    单位：$s$')
    plt.ylabel('$T$    单位:$°C$')
    plt.title(f'温度-时间拟合图(燃烧热的测定:{substance_name})')

    plt.grid(True, which='both')
    plt.minorticks_on()

    plt.gca().spines['top'].set_linewidth(1)
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['right'].set_linewidth(1)

    plt.savefig(fr'./拟合图结果/{arc_name}', dpi=300)
    plt.show()

    # 输出结果
    print(f"中点: {tuple(point_mean)}")
    print(f"T_1: {T1} ℃")
    print(f"T_2: {T2} ℃")
    print(f"ΔT: {delta_T} ℃")


# 文件路径
file_dir = r'./燃烧热的测定原始记录表(非).xlsx'
sheet_names = pd.ExcelFile(file_dir).sheet_names

ans_benzoic_acid_origin_data_df = pd.read_excel(file_dir, sheet_name=sheet_names[0], header=None)
ans_naphthalene_origin_data_df = pd.read_excel(file_dir, sheet_name=sheet_names[1], header=None)

# 苯甲酸原始数据
data_benzoic_acid_temperature = np.array(ans_benzoic_acid_origin_data_df.iloc[3:, 2].values, dtype=float)

# 萘原始数据
data_naphthalene_temperature = np.array(ans_naphthalene_origin_data_df.iloc[3:, 2].values, dtype=float)

# 处理数据
process_data(data_benzoic_acid_temperature, '苯甲酸', '苯甲酸燃烧热的测定')
process_data(data_naphthalene_temperature, '萘', '萘燃烧热的测定')

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
