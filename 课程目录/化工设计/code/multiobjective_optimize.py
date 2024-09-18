import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
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

# 初始化吸收塔问题
problem_instance = AbsorptionColumnProblem()

# 相关性分析函数
def correlation_analysis(problem, x_ranges, fixed_values):
    results = {'cost_factor': [], 'safety_factor': []}
    variable_values = []

    for i, var_range in enumerate(x_ranges):
        cost_factors, safety_factors = [], []
        for value in var_range:
            x = fixed_values.copy()
            x[i] = value
            cost_factor, safety_factor, _ = problem.evaluate(x)
            cost_factors.append(cost_factor)
            safety_factors.append(safety_factor)

        # 记录每个变量的结果
        results['cost_factor'].append(cost_factors)
        results['safety_factor'].append(safety_factors)
        variable_values.append(var_range)

    return variable_values, results

# 安全相关系数计算函数
def safe_corrcoef(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    else:
        return np.corrcoef(x, y)[0, 1]

# 获取相关性系数
def get_correlation_coefficients(variable_values, results, variable_names):
    correlations = {}
    for i, name in enumerate(variable_names):
        cost_factor_corr = safe_corrcoef(variable_values[i], results['cost_factor'][i])
        safety_corr = safe_corrcoef(variable_values[i], results['safety_factor'][i])
        correlations[name] = {
            'cost_factor_correlation': cost_factor_corr,
            'safety_correlation': safety_corr
        }
    return correlations

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

# 运行相关性分析
variable_values, results = correlation_analysis(problem_instance, x_ranges, fixed_values)

# 获取相关性系数
correlation_coefficients = get_correlation_coefficients(variable_values, results, variable_names)

# 根据相关性排序，并选择相关性最高的 n 个变量，这里设置为 9，即全部考虑
n = 9
sorted_params = sorted(correlation_coefficients.items(), key=lambda item: abs(item[1]['cost_factor_correlation']), reverse=True)
top_correlated_params = [param[0] for param in sorted_params[:n]]

# 打印相关性最高的四个变量
print("选择的关键变量:", top_correlated_params)

# 获取关键变量的索引
variable_indices = [variable_names.index(param) for param in top_correlated_params]

# 变量的上下限定义（针对所有可能的变量）
bounds_dict = {
    'Gas Flow Rate': (900, 1100),
    '$SO_{2}$ Mole Fraction': (0.01, 0.1),
    'Absorption Efficiency': (0.96, 0.98),
    'Temperature': (20, 50),
    'Pressure': (100, 150),
    'Packing Surface Area per Volume $a_{t}$': (150, 300),
    'Packing Width $d$': (25, 50),
    'Maximum Tower Height $h_{max}$': (5, 10),
    'Minimum L/V Ratio': (1.2, 2.0)
}

# 获取关键变量的索引并根据这些索引提取对应的上下限
variable_indices = [variable_names.index(param) for param in top_correlated_params]
bounds = [bounds_dict[variable_names[i]] for i in variable_indices]

# 定义问题类，用于多目标优化
class AbsorptionColumnMultiObjectiveProblem(ElementwiseProblem):
    def __init__(self, fixed_values, variable_indices, bounds):
        # 设置变量范围 (bounds)，对应变量的上下限
        super().__init__(n_var=len(variable_indices), n_obj=2, n_constr=0, xl=np.array([b[0] for b in bounds]), xu=np.array([b[1] for b in bounds]))
        self.fixed_values = fixed_values
        self.variable_indices = variable_indices
        self.problem = problem_instance

    def _evaluate(self, x, out, *args, **kwargs):
        # 更新优化变量到全局变量空间
        full_x = self.fixed_values.copy()
        for i, idx in enumerate(self.variable_indices):
            full_x[idx] = x[i]
        
        # 获取各个性能指标
        cost_factor, safety_factor, _ = self.problem.evaluate(full_x)
        
        # 成本最小化，安全因子最大化 (因此取负)
        out["F"] = [cost_factor, -safety_factor]

# 创建问题实例，并传入动态生成的bounds
problem = AbsorptionColumnMultiObjectiveProblem(fixed_values, variable_indices, bounds)

# 继续使用NSGA-II算法进行优化
algorithm = NSGA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),
    mutation=PolynomialMutation(eta=20),
    eliminate_duplicates=True
)

# 使用明确的终止条件
termination = get_termination("n_gen", 100)

# 进行优化
res = minimize(problem,
               algorithm,
               termination,  # 明确的终止条件
               seed=1,
               verbose=True)

# 输出帕累托前沿解集及其目标值
print("帕累托前沿解集:")
for solution in res.X:
    print(solution)

print("对应的目标值 (成本, 安全因子):")
for objective in res.F:
    print(objective)

# 可视化帕累托前沿 - 二维图
plt.figure(figsize=(10, 7))
plt.scatter(res.F[:, 0], -res.F[:, 1], c='r', marker='o')

# 设置轴标签
plt.xlabel('Cost   Factor', fontsize=14)
plt.ylabel('Safety Factor', fontsize=14)
plt.title('Pareto Front 2D Visualization', fontsize=16)
plt.show()
