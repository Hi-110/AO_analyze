import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from math import pi, sqrt, cos, exp

# 定义计算斯特列尔比的函数
def calculate_strehl_ratio(r0_cm, fG, D, N, magnitude, f_frame, 
                           k_fitting=0.3, sigma_miscel=0.05, eta=0.9,
                           tau_a=0.9, tau_o=0.8, sigma_R=0.3, I_D=0.01, K=9,
                           wavelength_guide=589e-9, wavelength_obs=1550e-9,
                           elevation=60, f_e3db_ratio=10, epsilon=0.18):
    """计算斯特列尔比"""
    
    # 计算大气相干长度在当前仰角下的值（单位：米）- 修正：根据HTML文件，应该是cos(z)^(3/5)
    r0_ref_m = r0_cm / 100  # 转换为米
    zenith_angle_rad = np.radians(90 - elevation)  # 天顶角
    r0_actual = r0_ref_m * (cos(zenith_angle_rad)) ** (3/5)
    
    # 计算导星波长下的大气相干长度 - 修正：根据HTML文件，应该是波长依赖的r0计算
    r0_guide = r0_actual * (wavelength_guide / 550e-9) ** 1.2
    
    # 计算系统带宽
    f_e3db = f_frame / f_e3db_ratio
    
    # 计算拟合误差方差 - 与HTML文件一致
    sigma2_fit = k_fitting * (D / r0_guide) ** (5/3) * N ** (-5/6)
    
    # 计算时间误差方差 - 修正：根据HTML文件，考虑仰角对Greenwood频率的影响
    zenith_angle_rad_temp = np.radians(90 - elevation)  # 天顶角
    cos_z = cos(zenith_angle_rad_temp)
    f_G_effective = fG * (cos_z) ** (3/5)
    sigma2_temp = (f_G_effective / f_e3db) ** (5/3)
    
    # 计算信号电子数 - 修正：根据HTML文件中的光子流量密度计算
    phi_cm2 = 4e6 * 10 ** (-magnitude / 2.5)  # 光子流量密度 (photons/cm²-s)
    phi = phi_cm2 * 1e4  # 转换为 photons/m²-s
    t_int = 1.0 / f_frame  # 积分时间
    D_sub = D / sqrt(N)  # 子孔径直径
    A_sub = pi * (D_sub / 2) ** 2  # 圆形孔径面积（与HTML文件一致）
    
    # 计算斜层大气透过率
    tau_a_actual = tau_a * (cos(zenith_angle_rad))
    
    S_M = phi * eta * t_int * A_sub * tau_a_actual * tau_o  # 信号电子数
    
    # 计算噪声方差 - 修正：根据HTML文件，应该包含背景噪声
    S_B = 0  # 背景噪声（假设为0）
    noise_variance = S_M + S_B + K * (sigma_R ** 2 + t_int * I_D)
    
    # 计算信噪比 - 修正：与HTML文件一致
    SNR = S_M / sqrt(noise_variance) if noise_variance > 0 else 0
    
    # 计算噪声误差 - 修正：根据HTML文件，使用正确的s参数计算
    if SNR > 0:
        # SHWFS噪声误差计算中的s参数 - 根据HTML文件公式
        # s = D_guide / L * (D*/sqrt(N)) / wavelength_guide
        D_guide = 0.5  # 导星望远镜口径
        L = 92000  # 导星高度，92km
        D_star = 2.0  # 主望远镜口径
        
        s = (D_guide / L) * (D_star / sqrt(N)) / wavelength_guide
        constant_term = sqrt((3/16) ** 2 + (s/8) ** 2)
        # HTML文件中的公式：2π² × constant_term / SNR
        sigma2_wfs = 2 * pi ** 2 * constant_term / SNR
    else:
        sigma2_wfs = 1e6  # 极大值，表示无法测量
    
    # 计算非等晕误差（假设自然导星，忽略）
    sigma2_iso = 0
    
    # 计算圆锥效应（假设自然导星，忽略）
    sigma2_cone = 0
    
    # 总波前误差方差 - 与HTML文件一致
    total_variance = sigma2_fit + sigma2_temp + sigma2_wfs + sigma2_iso + sigma2_cone + sigma_miscel
    
    # 计算斯特列尔比 - 修正：根据HTML文件中的公式
    # HTML文件：SR = exp(-σ²_total × (λ_guide/λ_obs)²)
    wavelength_ratio = wavelength_guide / wavelength_obs
    SR = exp(-total_variance * wavelength_ratio ** 2)
    
    return SR

# 主绘图函数
def plot_3x3_strehl_ratio():
    """绘制3x3子图"""
    
    # 固定参数
    elevation = 60
    magnitudes = [0, 3, 6, 8, 10, 12]  # 星等（数值越大越暗）
    f_frame_values = np.linspace(50, 2000, 100)  # 50Hz到2000Hz，线性分布
    
    # 大气条件和单元数组合
    atmosphere_conditions = [
        (15.13, 40),   # 最差大气条件
        (18.60, 30),   # 中等大气条件
        (22.24, 20)    # 最佳大气条件
    ]
    N_values = [100, 200, 400]
    
    # 固定其他参数
    D = 2.0
    wavelength_guide = 589e-9
    wavelength_obs = 1550e-9
    
    # 创建3x3子图，调整图形尺寸和间距
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    
    # 设置主标题，增加顶部间距
    fig.suptitle('不同大气条件和哈特曼单元数下的斯特列尔比随闭环频率变化关系', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(magnitudes)))
    
    # 创建一个共享的图例
    handles = []
    labels = []
    
    for i, (r0, fG) in enumerate(atmosphere_conditions):
        for j, N in enumerate(N_values):
            ax = axes[i, j]
            
            # 绘制不同星等的曲线
            for k, magnitude in enumerate(magnitudes):
                # 计算每个闭环频率下的斯特列尔比
                SR_values = np.array([calculate_strehl_ratio(r0, fG, D, N, magnitude, f_frame)
                                      for f_frame in f_frame_values])
                
                # 绘制曲线，保存图例句柄
                line, = ax.plot(f_frame_values, SR_values * 100, 
                         color=colors[k], linewidth=1.5)
                
                # 只在第一个子图保存图例信息
                if i == 0 and j == 0:
                    handles.append(line)
                    labels.append(f'Mag {magnitude}')
            
            # 简化子图标题
            ax.set_title(f'r0={r0:.1f}cm, fG={fG:.0f}Hz, N={N}', fontsize=8)
            
            # 只在边缘子图设置坐标轴标签
            if i == 2:  # 最后一行
                ax.set_xlabel('频率 (Hz)', fontsize=8)
            if j == 0:  # 第一列
                ax.set_ylabel('斯特列尔比 (%)', fontsize=8)
            
            # 设置坐标轴范围
            ax.set_xlim(50, 2000)
            ax.set_ylim(0, 100)  # 0% 到 100%
            
            # 设置y轴刻度格式为百分比
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
            
            # 添加网格
            ax.grid(True, which='both', ls='--', alpha=0.3)
            
            # 移除单个子图的图例（如果存在）
            if ax.legend_ is not None:
                ax.legend_.remove()
    
    # 添加共享图例
    fig.legend(handles, labels, loc='upper center', 
               bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=8)
    
    # 调整子图间距，使布局更加紧凑
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.98, hspace=0.4, wspace=0.4)
    plt.show()

# 运行绘图
plot_3x3_strehl_ratio()