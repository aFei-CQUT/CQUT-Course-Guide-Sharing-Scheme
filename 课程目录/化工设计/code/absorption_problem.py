import numpy as np
from absorption_column import AbsorptionColumn

# 创建吸收塔问题的类
class AbsorptionColumnProblem:
    
    def __init__(self):
        pass
    
    def evaluate(self, x):
        gas_flow_rate, so2_mole_fraction, absorption_efficiency, temperature, pressure, a_t, d, h_max, L_over_V_min_ratio = x
        # 创建吸收塔对象
        absorption_column = AbsorptionColumn(
            gas_flow_rate, so2_mole_fraction,
            absorption_efficiency, temperature,pressure,
            a_t, d, h_max, L_over_V_min_ratio)
        # 执行物料平衡计算
        absorption_column.material_balance()
        # 计算塔径
        u_F, D = absorption_column.calculate_tower_diameter()
        # 圆整塔径
        D_rounded = absorption_column.round_tower_diameter(D)
        # 塔径校核
        flooding_ratio, D_over_d, U = absorption_column.tower_diameter_check(D_rounded, absorption_column.d, absorption_column.a_t)
        # 计算传质单元数
        _ = absorption_column.calculate_mass_transfer_units()
        # 计算膜吸收系数
        a_w, k_G, k_L = absorption_column.calculate_membrane_absorption_coefficient(D_rounded, absorption_column.a_t)
        # 计算体积吸收系数
        k_Ga, k_La = absorption_column.calculate_volume_absorption_coefficient()
        # 修正体积吸收系数
        k_prime_Ga, k_prime_La = absorption_column.correct_volume_absorption_coefficient()
        # 计算总传质系数
        _ = absorption_column.calculate_overall_absorption_coefficient()
        # 计算填料层高度
        H_OG, Z, Z_prime = absorption_column.calculate_packing_height(D_rounded)
        # 计算Eckert坐标值以查压降
        X_Eckert_phi_P, Y_Eckert_phi_P = absorption_column.calculate_packing_delta_pressure()
        # 分段处理
        num_sections = np.ceil(Z_prime / h_max)
        effective_height = Z_prime / num_sections
        # 计算塔的有效体积 (塔内径减去壁厚后的体积)
        width = 0.08
        internal_radius = (D_rounded / 2) - width  # 塔皮的厚度 width
        effective_volume = np.pi * internal_radius**2 * effective_height
        # 填料个数基于塔的体积来计算
        packing_cost_factor_per_unit = d * 0.5
        number_of_packings = effective_volume / (d * h_max) * num_sections  # 基于体积的填料个数
        packing_cost_factor = packing_cost_factor_per_unit * number_of_packings
        # 计算塔的总成本，包括分段成本
        base_cost = effective_volume * 10000 + packing_cost_factor * 1500
        # 每分一段的额外成本
        section_cost_per_unit = 2000
        section_cost = (num_sections - 1) * section_cost_per_unit  # 减1是因为第一个段不需要额外成本
        total_cost = base_cost + section_cost
        # 安全系数计算
        safety_factor = 1 / (flooding_ratio + 1e-5)
    
        return total_cost, safety_factor, absorption_column