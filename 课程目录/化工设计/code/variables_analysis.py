import numpy as np
import matplotlib.pyplot as plt
from absorption_problem import AbsorptionColumnProblem

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
# 使用 seaborn 风格美化
plt.style.use('seaborn-v0_8-whitegrid')
# 固定的其他变量值
fixed_values = np.array([1000, 0.05, 0.98, 20, 120, 175, 38, 7.5, 1.40])
# 变量名称（英文）
variable_names = [
    'Gas Flow Rate', '$SO_{2}$ Mole Fraction', 'Absorption Efficiency', 
    'Temperature', 'Pressure', 'Packing Surface Area per Volume $a_{t}$', 
    'Packing Width $d$', 'Maximum Tower Height $h_{max}$', 'Minimum L/V Ratio'
]
# x 轴变量的取值范围
x_ranges = [
    np.linspace(900, 1100, 200),    # Gas Flow Rate
    np.linspace(0.01, 0.1, 200),    # SO2 Mole Fraction
    np.linspace(0.96, 0.98, 200),   # Absorption Efficiency
    np.linspace(20, 50, 200),       # Temperature
    np.linspace(100, 150, 200),     # Pressure
    np.linspace(150, 200, 200),     # Packing Surface Area per Volume a_t
    np.linspace(25, 50, 200),       # Packing Width d
    np.linspace(5, 10, 200),        # Maximum Tower Height h_max
    np.linspace(1.2, 2.0, 200)      # Minimum L/V Ratio
]
problem = AbsorptionColumnProblem()

# 归一化函数，使用 Min-Max 归一化将值缩放到 [0, 1]
def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val == 0:
        return np.zeros_like(values) + 0.5
    return (values - min_val) / (max_val - min_val)

# 移动平均函数
def moving_average(data, window_size):
    """ 计算移动平均 """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 参数敏感性分析函数
def sensitivity_analysis(problem, fixed_values, variable_names, perturbation=0.01):
    sensitivity_results = {}
    for i, name in enumerate(variable_names):
        original_value = fixed_values[i]
        perturbed_values = [original_value * (1 - perturbation), original_value * (1 + perturbation)]
        costs, safety_factors = [], []
        for value in perturbed_values:
            x = fixed_values.copy()
            x[i] = value
            cost_factor, safety_factor, _ = problem.evaluate(x)
            costs.append(cost_factor)
            safety_factors.append(safety_factor)
        
        # 计算标准差作为敏感性度量
        sensitivity_results[name] = {
            'cost_sensitivity': np.std(costs),
            'safety_factor_sensitivity': np.std(safety_factors)
        }
    
    return sensitivity_results

# 相关性分析函数
def correlation_analysis(problem, x_ranges, fixed_values):
    results = {'cost_factor': [], 'safety_factor': []}
    variable_values = []
    for i, var_range in enumerate(x_ranges):
        costs, safety_factors = [], []
        for value in var_range:
            x = fixed_values.copy()
            x[i] = value
            cost_factor, safety_factor, _ = problem.evaluate(x)
            costs.append(cost_factor)
            safety_factors.append(safety_factor)
        # 记录每个变量的结果
        results['cost_factor'].append(costs)
        results['safety_factor'].append(safety_factors)
        variable_values.append(var_range)

    return variable_values, results

# 在计算相关系数之前，检查变量数据
def safe_corrcoef(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    else:
        return np.corrcoef(x, y)[0, 1]

# 相关性分析时使用 safe_corrcoef 函数
def get_correlation_coefficients(variable_values, results, variable_names):
    correlations = {}
    for i, name in enumerate(variable_names):
        cost_corr = safe_corrcoef(variable_values[i], results['cost_factor'][i])
        safety_corr = safe_corrcoef(variable_values[i], results['safety_factor'][i])
        correlations[name] = {
            'cost_correlation': cost_corr,
            'safety_correlation': safety_corr
        }
    return correlations

# 绘制9张图，控制每个变量，其他变量固定
fig, axes = plt.subplots(3, 3, figsize=(18, 18), dpi=75)  # 调整图的大小以增加子图之间的空间
window_size = 5  # 设置移动平均窗口大小

for i in range(9):
    ax = axes[i // 3, i % 3]
    variable_values = x_ranges[i]
    costs = []
    safety_factors = []
    for value in variable_values:
        x = fixed_values.copy()
        x[i] = value
        cost_factor, safety_factor, _ = problem.evaluate(x)
        costs.append(cost_factor)
        safety_factors.append(safety_factor)
    # 归一化数据
    normalized_costs = normalize(np.array(costs))
    normalized_safety_factors = normalize(np.array(safety_factors))
    # 平滑处理
    smoothed_costs = moving_average(normalized_costs, window_size)
    smoothed_safety_factors = moving_average(normalized_safety_factors, window_size)
    # 重新计算变量值的范围，因为移动平均会减少数据点数
    smoothed_variable_values = variable_values[window_size-1:]
    # 绘制平滑后的归一化曲线
    ax.plot(smoothed_variable_values, smoothed_costs, label='Cost Factor (Smoothed)', color='purple', lw=2)
    ax.plot(smoothed_variable_values, smoothed_safety_factors, label='Safety Factor (Smoothed)', color='green', lw=2, linestyle='--')
    ax.set_title(f'{variable_names[i]} vs. Metrics (Smoothed)', fontsize=18)
    ax.set_xlabel(variable_names[i], fontsize=15)
    ax.set_ylabel('Normalized Value', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # 添加图例
    ax.legend(fontsize=12)

# 调整布局以避免重叠，并调整子图之间的间距
plt.subplots_adjust(wspace=0.3, hspace=0.4)
# 显示图表
plt.show()
# 运行相关性分析
variable_values, results = correlation_analysis(problem, x_ranges, fixed_values)
# 打印相关性系数
correlation_coefficients = get_correlation_coefficients(variable_values, results, variable_names)
for param, correlations in correlation_coefficients.items():
    print(f"{param}:")
    print(f"  Cost   Factor Correlation: {correlations['cost_correlation']:.4f}")
    print(f"  Safety Factor Correlation: {correlations['safety_correlation']:.4f}")