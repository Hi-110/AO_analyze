#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单模光纤耦合效率计算工具 - Python版本
基于自适应光学系统的波前误差和耦合效率计算

基于网页版ao.html中的计算模型，提供简化的输入-输出计算功能
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, cos, exp


class SingleModeCouplingCalculator:
    """单模光纤耦合效率计算器 - 基于网页版计算模型"""
    
    def __init__(self):
        # 固定参数
        self.wavelength_guide = 589  # 导星波长 (nm)
        self.D = 2.0  # 望远镜口径 (m)
        self.epsilon = 0.18  # 遮拦比
    
    def calculate_strehl_ratio(self, sigma_total):
        """计算斯特列尔比"""
        return exp(-sigma_total**2)
    
    def calculate_sr_at_wavelength(self, sigma_total, wavelength_guide, wavelength_obs):
        """计算特定波长下的斯特列尔比"""
        # 波长转换：nm → m
        wavelength_guide_m = wavelength_guide * 1e-9
        wavelength_obs_m = wavelength_obs * 1e-9
        return exp(-(sigma_total * wavelength_guide_m / wavelength_obs_m)**2)
    
    def calculate_single_mode_coupling_with_sr(self, SR_obs, coupling_params):
        """
        单模光纤耦合效率计算（直接使用SR值）
        
        参数:
        SR_obs: 观测波长下的斯特列尔比 (小数形式，如0.5)
        coupling_params: 字典，包含以下参数:
            wavelength_obs: 观测波长 (nm)
            focal_length: 系统等效焦距 (m)
            mfd: 光纤模场直径 (μm)
            jitter_sigma: 随机抖动标准差 (nrad)
            system_bias: 系统偏置角 (nrad)
        
        返回:
            dict: 包含所有计算结果的字典
        """
        # 参数转换
        wavelength_obs_m = coupling_params['wavelength_obs'] * 1e-9  # nm → m
        MFD_m = coupling_params['mfd'] * 1e-6  # μm → m
        jitter_rad = coupling_params['jitter_sigma'] * 1e-9  # nrad → rad
        system_bias_rad = coupling_params['system_bias'] * 1e-9  # nrad → rad
        focal_length = coupling_params['focal_length']
        
        # 计算模式匹配效率 (基于光纤参数和跟踪误差)
        # 使用跟踪误差作为倾斜误差
        sigma_tilt = jitter_rad
        eta_mode = exp(-(pi * MFD_m * sigma_tilt / (4 * wavelength_obs_m * focal_length))**2)
        
        # 计算角度抖动效率 (基于光斑抖动公式)
        # 使用新公式：1/[8(f*theta_jitter_rms/MFD)^2 + 1]
        f_theta_MFD = focal_length * jitter_rad / MFD_m
        eta_jitter = 1 / (1 + 8 * f_theta_MFD**2)
        
        # 总耦合效率
        eta_total = SR_obs * eta_mode * eta_jitter
        
        return {
            'total': eta_total,
            'mode_matching': eta_mode,
            'jitter_efficiency': eta_jitter,
            'strehl_ratio': SR_obs,
            'wavelength_obs': coupling_params['wavelength_obs'],
            'focal_length': coupling_params['focal_length'],
            'mfd': coupling_params['mfd'],
            'jitter_sigma': coupling_params['jitter_sigma'],
            'system_bias': coupling_params['system_bias']
        }
    
    def calculate_single_mode_coupling(self, sigma_total, coupling_params):
        """
        完整的单模耦合效率计算（包含波前误差计算）
        
        参数:
        sigma_total: 总波前误差 (rad)
        coupling_params: 字典，包含耦合效率计算参数
        
        返回:
            dict: 包含所有计算结果的字典
        """
        # 计算观测波长下的斯特列尔比
        SR_obs = self.calculate_sr_at_wavelength(sigma_total, 
                                                self.wavelength_guide,
                                                coupling_params['wavelength_obs'])
        
        # 计算耦合效率
        return self.calculate_single_mode_coupling_with_sr(SR_obs, coupling_params)
    
    def calculate_total_wavefront_error(self, params):
        """
        计算总波前误差方差（简化版，仅包含主要误差分量）
        
        参数:
        params: 字典，包含以下参数:
            r0_ref: 大气相干长度 @550nm (cm)
            f_G: Greenwood频率 (Hz)
            D: 望远镜口径 (m)
            N: 哈特曼单元数
            magnitude: 导星星等
            f_frame: 帧频 (Hz)
            k_fitting: 拟合误差系数
            sigma_miscel: 杂项误差 (rad²)
            elevation: 仰角 (°)
            tau_a: 天顶大气透过率
            tau_o: 光学系统效率
            eta: 量子效率
            sigma_R: 读出噪声 (e-)
            I_D: 暗电流 (e-/s)
            K: 子光斑像素数
        
        返回:
            float: 总波前误差方差 (rad²)
        """
        # 计算大气相干长度在当前仰角下的值
        r0_ref_m = params['r0_ref'] / 100  # 转换为m
        zenith_angle_rad = np.radians(90 - params['elevation'])  # 天顶角
        r0_actual = r0_ref_m * (cos(zenith_angle_rad))**(3/5)
        
        # 计算导星波长下的大气相干长度
        r0_guide = r0_actual * (self.wavelength_guide / 550e-9)**1.2
        
        # 计算拟合误差方差
        sigma2_fit = params['k_fitting'] * (params['D'] / r0_guide)**(5/3) * params['N']**(-5/6)
        
        # 计算时间误差方差
        zenith_angle_rad_temp = np.radians(90 - params['elevation'])
        cos_z = cos(zenith_angle_rad_temp)
        f_G_effective = params['f_G'] * (cos_z)**(3/5)
        sigma2_temp = (f_G_effective / (params['f_frame'] / 10))**(5/3)  # 假设带宽比为10
        
        # 计算信号电子数
        phi_cm2 = 4e6 * 10**(-params['magnitude'] / 2.5)  # 光子流量密度 (photons/cm²-s)
        phi = phi_cm2 * 1e4  # 转换为 photons/m²-s
        t_int = 1.0 / params['f_frame']  # 积分时间
        D_sub = params['D'] / sqrt(params['N'])  # 子孔径直径
        A_sub = pi * (D_sub / 2)**2  # 圆形孔径面积
        
        # 计算斜层大气透过率
        tau_a_actual = params['tau_a'] * cos(zenith_angle_rad)
        S_M = phi * params['eta'] * t_int * A_sub * tau_a_actual * params['tau_o']  # 信号电子数
        
        # 计算噪声方差
        S_B = 0  # 背景噪声（假设为0）
        noise_variance = S_M + S_B + params['K'] * (params['sigma_R']**2 + t_int * params['I_D'])
        
        # 计算信噪比
        SNR = S_M / sqrt(noise_variance) if noise_variance > 0 else 0
        
        # 计算噪声误差
        if SNR > 0:
            D_guide = 0.5  # 导星望远镜口径
            L = 92000  # 导星高度，92km
            D_star = 2.0  # 主望远镜口径
            
            s = (D_guide / L) * (D_star / sqrt(params['N'])) / self.wavelength_guide
            constant_term = sqrt((3/16)**2 + (s/8)**2)
            sigma2_wfs = 2 * pi**2 * constant_term / SNR
        else:
            sigma2_wfs = 1e6  # 极大值，表示无法测量
        
        # 总波前误差方差
        total_variance = (sigma2_fit + sigma2_temp + sigma2_wfs + 
                         params['sigma_miscel'])  # 简化，忽略非等晕和圆锥效应
        
        return total_variance


def plot_parameter_analysis(calculator, base_params, coupling_params, param_name, 
                           min_val, max_val, step_val, plot_type='total'):
    """
    参数影响分析绘图
    
    参数:
    calculator: 计算器实例
    base_params: 基础参数字典
    coupling_params: 耦合参数字典
    param_name: 分析参数名
    min_val, max_val, step_val: 参数范围
    plot_type: 绘图类型 ('total', 'mode', 'jitter')
    """
    # 生成参数值序列
    param_values = np.arange(min_val, max_val + step_val, step_val)
    
    # 计算不同参数值下的耦合效率
    total_efficiencies = []
    mode_efficiencies = []
    jitter_efficiencies = []
    
    for val in param_values:
        # 更新参数值
        if param_name in base_params:
            test_params = base_params.copy()
            test_params[param_name] = val
            
            # 计算总波前误差
            total_variance = calculator.calculate_total_wavefront_error(test_params)
            sigma_total = sqrt(total_variance)
            
            # 计算斯特列尔比
            SR_obs = calculator.calculate_sr_at_wavelength(sigma_total, 
                                                          calculator.wavelength_guide,
                                                          coupling_params['wavelength_obs'] * 1e-9)
        elif param_name in coupling_params:
            # 如果是耦合参数，直接使用输入的SR值
            test_coupling_params = coupling_params.copy()
            test_coupling_params[param_name] = val
            SR_obs = base_params.get('SR_obs', 0.5)  # 默认SR值
        else:
            # 如果是SR参数
            SR_obs = val / 100  # 百分比转换为小数
            test_coupling_params = coupling_params.copy()
        
        # 计算耦合效率
        result = calculator.calculate_single_mode_coupling_with_sr(SR_obs, 
                                                                  test_coupling_params)
        
        total_efficiencies.append(result['total'] * 100)  # 转换为百分比
        mode_efficiencies.append(result['mode_matching'] * 100)
        jitter_efficiencies.append(result['jitter_efficiency'] * 100)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'total' or plot_type == 'all':
        plt.plot(param_values, total_efficiencies, 'b-', linewidth=2, label='总耦合效率')
    
    if plot_type == 'mode' or plot_type == 'all':
        plt.plot(param_values, mode_efficiencies, 'g-', linewidth=2, label='模式匹配效率')
    
    if plot_type == 'jitter' or plot_type == 'all':
        plt.plot(param_values, jitter_efficiencies, 'r-', linewidth=2, label='角度抖动效率')
    
    # 设置图表
    plt.xlabel(f'{param_name} 参数值')
    plt.ylabel('效率 (%)')
    plt.title(f'{param_name} 参数对耦合效率的影响分析')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_contour_analysis(calculator, base_params, coupling_params, 
                         param1_name, param2_name, grid_points=20):
    """
    等高线分析绘图（二维参数分析）
    
    参数:
    calculator: 计算器实例
    base_params: 基础参数字典
    coupling_params: 耦合参数字典
    param1_name, param2_name: 两个分析参数名
    grid_points: 网格点数
    """
    # 生成参数网格
    if param1_name in base_params:
        param1_values = np.linspace(base_params[param1_name] * 0.5, 
                                   base_params[param1_name] * 1.5, grid_points)
    else:
        param1_values = np.linspace(coupling_params[param1_name] * 0.5, 
                                   coupling_params[param1_name] * 1.5, grid_points)
    
    if param2_name in base_params:
        param2_values = np.linspace(base_params[param2_name] * 0.5, 
                                   base_params[param2_name] * 1.5, grid_points)
    else:
        param2_values = np.linspace(coupling_params[param2_name] * 0.5, 
                                   coupling_params[param2_name] * 1.5, grid_points)
    
    # 计算耦合效率矩阵
    efficiency_matrix = np.zeros((len(param1_values), len(param2_values)))
    
    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            # 更新参数值
            if param1_name in base_params:
                test_base = base_params.copy()
                test_base[param1_name] = val1
            else:
                test_coupling = coupling_params.copy()
                test_coupling[param1_name] = val1
                test_base = base_params.copy()
            
            if param2_name in base_params:
                test_base[param2_name] = val2
            else:
                test_coupling[param2_name] = val2
            
            # 计算总波前误差和SR
            total_variance = calculator.calculate_total_wavefront_error(test_base)
            sigma_total = sqrt(total_variance)
            SR_obs = calculator.calculate_sr_at_wavelength(sigma_total, 
                                                          calculator.wavelength_guide,
                                                          coupling_params['wavelength_obs'] * 1e-9)
            
            # 计算耦合效率
            result = calculator.calculate_single_mode_coupling_with_sr(SR_obs, 
                                                                      test_coupling)
            efficiency_matrix[i, j] = result['total'] * 100  # 转换为百分比
    
    # 创建等高线图
    X, Y = np.meshgrid(param1_values, param2_values)
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, efficiency_matrix.T, levels=20, cmap='viridis')
    plt.colorbar(contour, label='总耦合效率 (%)')
    
    plt.xlabel(f'{param1_name}')
    plt.ylabel(f'{param2_name}')
    plt.title(f'{param1_name} 和 {param2_name} 对耦合效率的影响（等高线图）')
    plt.tight_layout()
    plt.show()


# 使用示例
def main():
    """主函数 - 使用示例"""
    # 创建计算器实例
    calculator = SingleModeCouplingCalculator()
    
    # 基础参数（波前误差计算用）
    base_params = {
        'r0_ref': 13.89,  # cm
        'f_G': 30,        # Hz
        'D': 2.0,         # m
        'N': 468,         # 
        'magnitude': 8.0, # mag
        'f_frame': 1000,  # Hz
        'k_fitting': 0.3, # 
        'sigma_miscel': 0.01,  # rad²
        'elevation': 60,  # °
        'tau_a': 0.9,     # 
        'tau_o': 0.8,     # 
        'eta': 0.9,       # 
        'sigma_R': 0.3,   # e-
        'I_D': 0.01,      # e-/s
        'K': 9,           # 
        'SR_obs': 0.5     # 斯特列尔比（百分比值）
    }
    
    # 耦合参数
    coupling_params = {
        'wavelength_obs': 1550,    # nm
        'focal_length': 10.0,      # m
        'mfd': 10.4,               # μm
        'jitter_sigma': 200,       # nrad
        'system_bias': 100         # nrad
    }
    
    # 计算单个耦合效率
    result = calculator.calculate_single_mode_coupling_with_sr(
        base_params['SR_obs'], coupling_params)
    
    print("=== 单模耦合效率计算结果 ===")
    print(f"总耦合效率: {result['total']*100:.1f}%")
    print(f"模式匹配效率: {result['mode_matching']*100:.1f}%")
    print(f"角度抖动效率: {result['jitter_efficiency']*100:.1f}%")
    print(f"斯特列尔比: {result['strehl_ratio']*100:.1f}%")
    print(f"匹配参数β: {result['beta']:.3f}")
    print(f"特征角度θ_eff: {result['theta_eff']*1e6:.1f} μrad")
    
    # 参数影响分析示例
    print("\n=== 参数影响分析示例 ===")
    
    # 分析焦距对耦合效率的影响
    plot_parameter_analysis(calculator, base_params, coupling_params, 
                           'focal_length', 5, 15, 0.5, 'all')
    
    # 分析随机抖动标准差的影响
    plot_parameter_analysis(calculator, base_params, coupling_params, 
                           'jitter_sigma', 50, 500, 10, 'all')
    
    # 等高线分析（焦距 vs 抖动标准差）
    plot_contour_analysis(calculator, base_params, coupling_params, 
                         'focal_length', 'jitter_sigma', grid_points=15)


if __name__ == "__main__":
    main()