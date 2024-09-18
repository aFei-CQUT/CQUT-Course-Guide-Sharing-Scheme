import logging
import numpy as np
from tabulate import tabulate

# 配置日志格式和输出级别
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

class AbsorptionColumn:
    # 单位均为注释中的第一个列出的,后面的为换算公式
    def __init__(self, gas_flow_rate, so2_mole_fraction, absorption_efficiency, temperature,
                 pressure, a_t, d, h_max, L_over_V_min_ratio):
        # 输入参数
        self.V_prime = gas_flow_rate                    # 进口气体体积流量, m³/h
        self.y1 = so2_mole_fraction                     # 二氧化硫摩尔分数
        self.eta = absorption_efficiency                # 二氧化硫吸收率
        self.T = temperature                            # 温度, °C
        self.p = pressure                               # 压力, kPa
        self.a_t = a_t                                  # 填料总比表面积, m²/m³
        self.d = d                                      # 填料宽度 mm
        self.h_max = h_max                              # 最大塔高 m
        self.L_over_V_min_ratio = L_over_V_min_ratio    # 最小液气比, 量纲1

        # 常量参数
        self.p0 = 1.03125 * 1e2                         # 标准大气压, kPa
        self.R = 8.314                                  # 理想气体常数, m³·kPa/(kmol·K)
        self.M_air = 28.95                              # 空气的摩尔质量, g/mol = kg/kmol
        self.M_so2 = 64.06                              # SO2的摩尔质量, g/mol = kg/kmol
        self.M_L = 18.02                                # 水的摩尔质量, g/mol = kg/kmol
        self.E = 3.55e3                                 # 亨利系数, kPa
        self.rho_L = 998.2                              # 水的密度, kg/m³
        self.mu_L = 1.005e-3                            # 水的黏度, Pa·s = 10³ mPa·s = 3600 kg/(m·h), 后面计算a_w/a_t时转换成 kg/(m·h)
        self.mu_V = 1.81e-5                             # 空气黏度, Pa·s = 10³ mPa·s = 3600 kg/(m·h), 后面计算a_w/a_t时转换成 kg/(m·h)
        self.D_L = 1.47e-9                              # SO2 在水中的扩散系数, m²/s = 3600 m²/h
        self.D_V = 1.08e-5                              # SO2在空气中的扩散系数, m²/s = 3600 m²/h
        self.sigma_c = 0.033                            # 材料材质的临界表面张力, N/m = kg/s² = 3600² kg/h²
        self.sigma_L = 0.0728                           # 水的表面张力, N/m = kg/s² = 3600² kg/h²
        self.g = 9.81                                   # 重力加速度, m/s² = 3600² m/h²
        self.phi_F = 550                                # 散装填料的泛点填料因子平均值,计算泛点气速的系数, m^-1
        self.phi_P = 232                                # 散装填料的压降填料因子平均值,计算填料塔压降Eckert图坐标的系数, m^-1
        self.varphi = 1                                 # 计算泛点气速的系数, 1
        self.psi = 1.45                                 # 体积吸收系数修正系数
        
        # 计算所得参数
        self.m = None                                   # 相平衡常数
        self.M_V = None                                 # 混合气体平均摩尔质量, g/mol  = kg/kmol
        self.rho_V = None                               # 混合气体的平均密度, kg/m³
        self.D_rounded = None                           # 圆整后的直径 m
        self.D_over_d = None                            # 填料径比
        self.u_actual = None                            # 实际操作气速 m/s
        self.u_F = None                                 # 泛点气速
        self.flooding_ratio = None                      # 泛点率
        self.X_Eckert_phi_F = None                      # Eckert 图的横坐标, 用于泛点气速, 计算时用到参数 Φ_F
        self.Y_Eckert_phi_F = None                      # Eckert 图的纵坐标, 用于泛点气速, 计算时用到参数 Φ_F
        self.X_Eckert_phi_P = None                      # Eckert 图的横坐标, 用于填料层压降, 计算时用到参数 Φ_P
        self.Y_Eckert_phi_P = None                      # Eckert 图的纵坐标, 用于填料层压降, 计算时用到参数 Φ_P

        # 结果存储字典
        self.results = {}

        # 计算基础物性参数
        self.compute_gas_properties()
        self.compute_liquid_properties()

    def compute_gas_properties(self):
        # 计算气体的摩尔质量、密度和相平衡常数
        self.M_V = self.y1 * self.M_so2 + (1 - self.y1) * self.M_air
        self.rho_V = (self.p * self.M_V) / (self.R * (273.15 + self.T))
        self.m = self.E / self.p

        # 存储结果
        self.results['M_V'] = self.M_V
        self.results['rho_V'] = self.rho_V
        self.results['D_V'] =self.D_V
        self.results['mu_V'] = self.mu_V
        self.results['E'] = self.E
        self.results['m'] = self.m

    def compute_liquid_properties(self):
        # 计算液相的相关参数
        self.H = self.rho_L / (self.E * self.M_L)  # kmol/(kPa·m³), self.M_L是混合液体的摩尔质量,低浓度,近似为水的摩尔质量
    
        # 存储结果
        self.results['rho_L'] = self.rho_L
        self.results['mu_L'] = self.mu_L
        self.results['sigma_L'] = self.sigma_L
        self.results['D_L'] = self.D_L
        self.results['H'] = self.H
    
    def material_balance(self):
        # 计算物料平衡,得到吸收塔中气液相流量的关系
        self.Y1 = self.y1 / (1 - self.y1)
        self.Y2 = self.Y1 * (1 - self.eta)
        self.V = (self.V_prime / 22.4) * (1 - self.y1) * (273 / (273 + self.T))
        self.L_over_V_min = (self.Y1 - self.Y2) / (self.Y1 / self.m - 0)
        self.L_over_V_min_ratio = self.L_over_V_min_ratio
        self.L_over_V = self.L_over_V_min_ratio * self.L_over_V_min
        self.L = self.L_over_V * self.V
        self.X1 = self.V * (self.Y1 - self.Y2) / self.L
    
        # 存储结果
        self.results.update({
            "Y1": self.Y1,
            "Y2": self.Y2,
            "V": self.V,
            "L_over_V_min": self.L_over_V_min,
            "L_over_V_min_ratio": self.L_over_V_min_ratio,
            "L_over_V": self.L_over_V,
            "L": self.L,
            "X1": self.X1
        })
    
        return self.results

    def calculate_tower_diameter(self):
        # 根据物料平衡结果计算塔径
        self.L = self.results["L"]
        self.W_L = self.L * self.M_L
        self.W_V = self.V_prime * self.rho_V
    
        self.X_Eckert_phi_F = (self.W_L / self.W_V) * (self.rho_V / self.rho_L) ** 0.5
        self.Y_Eckert_phi_F = 0.034
        
        # 泛点气速
        self.u_F = np.sqrt((self.Y_Eckert_phi_F * self.g * self.rho_L) / 
                           (self.phi_F * self.varphi * self.rho_V * (self.mu_L * 1000) ** 0.2))  # m/s
        
        # 设计操作气速
        self.u_design = 0.7 * self.u_F
    
        # 塔径计算
        self.V_s = self.V_prime / 3600
        self.D = np.sqrt((4 * self.V_s) / (np.pi * self.u_design))
    
        # 存储结果
        self.results.update({
            "W_L": self.W_L,
            "W_V": self.W_V,
            "X_Eckert_phi_F": self.X_Eckert_phi_F,
            "Y_Eckert_phi_F": self.Y_Eckert_phi_F,
            "u_F": self.u_F,
            "u_design": self.u_design,
            "D": self.D
        })
    
        return self.u_F, self.D

    def round_tower_diameter(self, D):
        # 圆整塔径到小数点后第一位
        self.D_rounded = np.ceil(D * 10) / 10
        
        self.results.update({
            "D_rounded": self.D_rounded
        })
        
        return self.D_rounded

    def tower_diameter_check(self, D_rounded, d, a_t):
        # 塔径校核
        self.V_s = self.V_prime / 3600
        self.u_actual = self.V_s / (np.pi * D_rounded**2 / 4)
        self.u_F = self.results["u_F"]
        
        # 泛点率参数
        self.flooding_ratio = self.u_actual / self.u_F
        
        # 填料规格校核参数
        self.D_over_d = D_rounded * 1000 / d  # 转换为mm
        
        self.L_W_min = 0.08                    # 最小润湿速率, m³/(m·h) = m²/h
        self.U_min = self.L_W_min * a_t        # 最小喷淋密度, m³/(m²·h) = m/h
        self.L = self.results["L"]
        self.V_L = self.L * self.M_L / self.rho_L
        self.A = np.pi * D_rounded**2 / 4
        
        # 液体喷淋密度
        self.U = self.V_L / self.A             # 转换为 kg/(m²·h)
        
        # 填料规格校核 + 泛点率校核 + 喷淋密度校核
        if 0.5 <= self.flooding_ratio <= 0.85 and self.D_over_d > 10 and self.U > self.U_min:
            # 存储结果
            self.results.update({
                "u_actual": self.u_actual,
                "flooding_ratio": self.flooding_ratio,
                "D_over_d": self.D_over_d,
                "U_min": self.U_min,
                "U": self.U
            })
            
        return self.flooding_ratio, self.D_over_d, self.U

    def calculate_mass_transfer_units(self):
        # 脱吸因数
        self.S = self.m / self.results["L_over_V"]  # 使用计算得到的 S = mV / L
        self.Y1 = self.results["Y1"]
        self.Y2 = self.results["Y2"]
        self.Y2_star = 0                            # 稀溶液,平衡摩尔分数近似为0
        
        # 计算气相总传质单元数 N_OG
        if (1 - self.S) != 0 and ((1 - self.S) * (self.Y1 - self.Y2_star) / (self.Y2 - self.Y2_star)) + self.S > 0:
            self.N_OG = (1 / (1 - self.S)) * np.log(((1 - self.S) * (self.Y1 - self.Y2_star) / (self.Y2 - self.Y2_star)) + self.S)
        else:
            self.N_OG = np.nan  # 或者其他适当的默认值
    
        # 存储结果
        self.results.update({
            "S": self.S,
            "N_OG": self.N_OG
        })
    
        return self.N_OG

    def calculate_membrane_absorption_coefficient(self, D_rounded, a_t):
        # 膜吸收系数的计算
        self.L = self.results["L"]
        self.W_L = self.L * self.M_L
        self.A = (np.pi * D_rounded**2) / 4
        self.U_L = self.W_L / self.A  # 转换为 kg/(m²·h)
    
        # 检查输入参数是否合理, 计算a_w/a_t和其他传质参数
        if a_t > 0 and self.mu_L > 0 and self.rho_L > 0 and self.sigma_L > 0 and self.U_L > 0:
            # 计算公式
            self.a_w_over_a_t = 1 - np.exp( (-1.45 * (self.sigma_c * 3600 * 3600 / (self.sigma_L * 3600 * 3600))**0.75 ) * \
                                         ( (self.U_L / (a_t * self.mu_L * 3600))**0.1 ) * \
                                         ( (self.U_L**2 * self.mu_L * 3600 / (self.rho_L * self.g * 3600 * 3600))**-0.05 ) * \
                                         ( (self.U_L**2 / (self.rho_L * self.sigma_L * 3600 * 3600 * a_t))**0.2 ) )
        else:
            self.a_w_over_a_t = np.nan  # 设置为无效值以避免错误

        self.a_w = self.a_w_over_a_t * a_t
        self.W_V = self.V_prime * self.rho_V
        self.U_V = self.W_V / self.A  # 转换为 kg/(m²·h)
    
        # 气膜吸收系数的公式 k_G
        self.k_G = 0.237 * ( ( (self.U_V / (a_t * self.mu_V * 3600))**0.7 ) * \
                       ( (self.mu_V * 3600 / (self.rho_V * self.D_V * 3600))**(1/3) ) * \
                       ( (a_t * self.D_V * 3600 / (self.R * (self.T + 273.15))) ) )
    
        # 液膜吸收系数的公式 k_L
        self.k_L = 0.0095 * ( (self.U_L / (self.a_w * self.mu_L * 3600))**(2/3) ) * \
                ( (self.mu_L * 3600 / (self.rho_L * self.D_L * 3600))**(-1/2) ) * \
                ( (self.mu_L * 3600 * self.g  * 3600 * 3600/ self.rho_L)**(1/3) )
    
        # 存储结果
        self.results.update({
            "U_L": self.U_L,
            "a_w_over_a_t": self.a_w_over_a_t,
            "a_w": self.a_w,
            "U_V": self.U_V,
            "k_G": self.k_G,
            "k_L": self.k_L
        })
        
        return self.a_w, self.k_G, self.k_L
    
    def calculate_volume_absorption_coefficient(self):
        # 计算体积吸收系数
        self.k_G = self.results["k_G"]
        self.k_L = self.results["k_L"]
        self.a_w = self.results["a_w"]
        
        self.k_Ga = self.k_G * self.a_w * self.psi**1.1
        self.k_La = self.k_L * self.a_w * self.psi**0.4
    
        self.results.update({
            "k_Ga": self.k_Ga,
            "k_La": self.k_La
        })
    
        return self.k_Ga, self.k_La

    def correct_volume_absorption_coefficient(self):
        # 修正体积吸收系数
        self.k_Ga = self.results["k_Ga"]
        self.k_La = self.results["k_La"]
        
        self.u_over_uF = self.u_actual / self.u_F
    
        self.k_prime_Ga = (1 + 9.5 * (self.u_over_uF - 0.5)**1.4) * self.k_Ga
        self.k_prime_La = (1 + 2.6 * (self.u_over_uF - 0.5)**2.2) * self.k_La
    
        # 存储结果
        self.results.update({
            "k_prime_Ga": self.k_prime_Ga,
            "k_prime_La": self.k_prime_La
        })
    
        return self.k_prime_Ga, self.k_prime_La
    
    def calculate_overall_absorption_coefficient(self):
        # 计算总传质系数
        self.k_prime_Ga = self.results.get("k_prime_Ga", self.results.get("k_Ga"))  # 如果 'k_prime_Ga' 不存在，使用 'k_Ga'
        self.k_prime_La = self.results.get("k_prime_La", self.results.get("k_La"))  # 如果 'k_prime_La' 不存在，使用 'k_La'
        self.H = self.results["H"]
        self.K_Ga = 1 / (1 / self.k_prime_Ga + 1 / (self.H * self.k_prime_La))
    
        # 存储结果
        self.results["K_Ga"] = self.K_Ga
        return self.K_Ga

    def calculate_packing_height(self, D_rounded):
        # 计算填料层高度 H_OG
        self.K_Ga = self.results["K_Ga"]                    # 总传质系数
        self.p = self.p                                     # 总压力
        self.V = self.results["V"]                          # 气体流量
        self.Ω = np.pi * (D_rounded / 2)**2                 # D_rounded 圆整后的直径
        self.H_OG = self.V / (self.K_Ga * self.p * self.Ω)  # Ω 是公式中的系数
        self.N_OG = self.results["N_OG"]                    # 传质单元数
    
        # 计算总高度 Z
        self.Z = self.H_OG * self.N_OG
    
        # 计算设计高度 Z_prime, 增加25%余量
        self.Z_prime = 1.25 * self.Z
    
        # 存储结果
        self.results.update({
            "H_OG": self.H_OG,
            "Z": self.Z,
            "Z_prime": self.Z_prime
        })
    
        return self.H_OG, self.Z, self.Z_prime
    
    def calculate_packing_delta_pressure(self):
        # 计算Eckert压降的适用坐标，以便查填料塔压降
        self.L = self.results["L"]
        self.W_L = self.L * self.M_L
        self.W_V = self.V_prime * self.rho_V
    
        # 计算X_Eckert_phi_P
        self.X_Eckert_phi_P = (self.W_L / self.W_V) * (self.rho_V / self.rho_L) ** 0.5
    
        # 计算Y_Eckert_phi_P
        self.Y_Eckert_phi_P = (self.u_actual**2 * self.phi_P * self.varphi) / self.g * \
                              (self.rho_V / self.rho_L) * (self.mu_L * 3600)**0.2
    
        # 存储结果
        self.results.update({
            "X_Eckert_phi_P": self.X_Eckert_phi_P,
            "Y_Eckert_phi_P": self.Y_Eckert_phi_P
        })
    
        return self.X_Eckert_phi_P, self.Y_Eckert_phi_P
    
    def display_results(self):
        # 使用表格展示计算结果
        logger.info("计算结果汇总:")
        table_data = []
        descriptions = {
            'M_V': '摩尔质量',
            'rho_V': '气体密度',
            'D_V': '气体扩散系数',
            'mu_V': '气体粘度',
            'E': '亨利常数',
            'm': '气体分子质量比',
            'rho_L': '液体密度',
            'mu_L': '液体粘度',
            'sigma_L': '表面张力',
            'D_L': '液体扩散系数',
            'H': '溶解度系数',
            'Y1': '进料气体组分',
            'Y2': '出料气体组分',
            'V': '气体流量',
            'L_over_V_min': '液气比',
            'L_over_V_min_ratio': '最小液气比',
            'L_over_V': '实际液气比',
            'L': '液体流量',
            'X1': '进料液体组分',
            'W_L': '液体质量流量',
            'W_V': '气体质量流量',
            'X_Eckert_phi_F': '泛点气速Eckert坐标X值',
            'Y_Eckert_phi_F': '泛点气速Eckert坐标Y值',
            'u_F': '泛点气速',
            'u_design': '设计气速',
            'D': '未圆整塔径',
            'D_rounded': '圆整后塔径',
            'u_actual': '实际操作气速',
            'flooding_ratio': '泛点率',
            'D_over_d': '圆整后塔径与填料直径比',
            'U_min': '最小喷淋密度',
            'U': '实际喷淋密度',
            'S': '脱吸因数',
            'N_OG': '气相总传质单元数',
            'U_L': '液相质量通量',
            'a_w_over_a_t': '有效比表面积比',
            'a_w': '有效比表面积',
            'U_V': '气相质量通量',
            'k_G': '气膜吸收系数',
            'k_L': '液膜吸收系数',
            'k_Ga': '气膜体积吸收系数',
            'k_La': '液膜体积吸收系数',
            'k_prime_Ga': '修正后的气膜体积吸收系数',
            'k_prime_La': '修正后的液膜体积吸收系数',
            'K_Ga': '总传质系数',
            'H_OG': '填料层高度',
            'Z': '总高度',
            'Z_prime': '设计高度',
            'X_Eckert_phi_P': '填料层压降Eckert坐标X值',
            'Y_Eckert_phi_P': '填料层压降Eckert坐标Y值'
        }
        units = {
            'M_V': 'g/mol', 
            'rho_V': 'kg/m³',
            'mu_V': 'Pa·s', 
            'D_V': 'm²/s',
            'E': 'kPa',
            'm': '-',
            'rho_L': 'kg/m³', 
            'mu_L': 'Pa·s', 
            'sigma_L': 'N/m',
            'D_L': 'm²/s',
            'H': 'kmol/(kPa·m³)',
            'Y1': '-',
            'Y2': '-', 
            'V': 'kmol/h',
            'L_over_V_min': '-',
            'L_over_V_min_ratio': '-',
            'L_over_V': '-',
            'L': 'kmol/h', 
            'X1': '-',
            'W_L': 'kg/h',
            'W_V': 'kg/h', 
            'X_Eckert_phi_F': '-', 
            'Y_Eckert_phi_F': '-',
            'u_F': 'm/s', 
            'u_design': 'm/s',
            'D': 'm', 
            'D_rouned': 'm',
            'u_actual': 'm/s',
            'flooding_ratio': '-',
            'D_over_d ': '-', 
            'U_min': 'm³/(m²·h)',
            'U': 'm³/(m²·h)', 
            'S': '-',
            'N_OG': '-',
            'U_L': 'kg/(m²·h)',
            'a_w_over_a_t': '-',
            'a_w': 'm²/m³',
            'U_V': 'kg/(m²·h)',
            'k_G': 'kmol/(m²·h·kPa)',
            'k_L': 'm/h',
            'k_Ga': 'kmol/(m²·h·kPa)',
            'k_La': 'm/h',
            'k_prime_Ga': 'kmol/(m²·h·kPa)',
            'k_prime_La': 'm/h',
            'K_Ga': 'kmol/(m²·h·kPa)',
            'H_OG': 'm',
            'Z': 'm',
            'Z_prime': 'm',
            'X_Eckert_phi_P': '-',
            'Y_Eckert_phi_P': '-'
        }

        for key, value in self.results.items():
                desc = descriptions.get(key, '-')
                unit = units.get(key, '-')
                table_data.append([desc, key, value, unit])
        
        logger.info("\n" + tabulate(table_data, headers=['描述信息', '参数符号', '参数值', '单位'], tablefmt='grid'))

def main():
    # 任务参数
    gas_flow_rate = 1000            # 进口气体体积流量, m³/h
    so2_mole_fraction = 0.060       # 二氧化硫摩尔分数
    absorption_efficiency = 0.98    # 二氧化硫吸收率
    temperature = 20                # 操作温度, °C
    pressure = 120                  # 操作压力, kPa
    a_t = 175                       # 填料总比表面积, m²/m³
    d = 38                          # 填料宽度 mm
    h_max = 6                       # 最大塔高 m
    L_over_V_min_ratio = 1.40       # 最小液气比 量纲1
    
    # 创建吸收塔对象
    absorption_column = AbsorptionColumn(gas_flow_rate, so2_mole_fraction, absorption_efficiency, temperature, 
                                         pressure, a_t, d, h_max, L_over_V_min_ratio)
    
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
    
    # 展示所有计算结果
    absorption_column.display_results()
    
    # 校核塔径设计合理性输出结果
    if 0.5 <= absorption_column.results['flooding_ratio'] <= 0.85 and absorption_column.results['D_over_d'] > 10 \
            and absorption_column.results['U'] > absorption_column.results['U_min']:
        logger.info("塔径校核通过,塔径设计合理")
    else:
        logger.info("塔径校核不通过,需要调整其他参数")
    
    # 填料层是否需要分段
    if absorption_column.results['Z_prime'] < absorption_column.h_max:
        logger.info(f"设计填料层高度为 {absorption_column.results['Z_prime']:.6f} m, 小于等于最大高度 {absorption_column.h_max} m, 故不进行分段。")
    else:
        logger.info(f"设计填料层高度为 {absorption_column.results['Z_prime']:.6f} m, 超过最大高度 {absorption_column.h_max} m, 需要进行分段。")
    
    absorption_column_results = absorption_column.results
    
    return absorption_column, absorption_column_results

if __name__ == "__main__":
    
    # 调用 main 函数, 获取实例化对象和计算结果汇总字典
    absorption_column, absorption_column_results = main()
