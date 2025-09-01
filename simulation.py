# 导入依赖库
import numpy as np
import matplotlib.pyplot as plt
from math import erfc
import pandas as pd
import xgboost as xgb
import math
from numba import jit
from tqdm import tqdm
import os
import random
# 用于优化的simulation函数构造，便于调用和测试

r_pile = [0.010, 0.0125, 0.016, 0.02, 0.025, 0.0315]
rp_to_ri = {
        0.010: 0.008,
        0.0125: 0.0102,
        0.016: 0.013,
        0.02: 0.0162,
        0.025: 0.0202,
        0.0315: 0.0252
    }

def simulation(params):
    # 获取输入参数
    rp_idx = int(params.get('rp_idx'))  # 管径索引
    Np = int(params.get('N'))
    v = float(params.get('v'))
    Lp = float(params.get('Lp'))

    # 解析输入参数-管径转化
    rp = r_pile[rp_idx]
    ri = rp_to_ri[rp]
    # ===================== 不变的参数设定 =====================
    n= 4
    ks = 2.1           # 土壤导热系数 W/(m·K)
    Cv = 2_200_000      # 土壤体积热容 J/(m³·K)
    as_ = ks / Cv       # 土壤热扩散率 m²/s
    rs = 0.5            # 土壤热阻系数
    tt = np.arange(0, 3600*24*365+1, 3600)  # 一年每小时的时间向量，单位秒
    ttmax = len(tt)                          # 时间步数
    tp = 31536000        # 一年总秒数
    tt0 = 26611200      # 供暖季开始时间
    Tsavg = 14.02        # 年平均地温
    Tamp1 = 14.69        # 年周期幅值1
    Tamp2 = 1.173        # 年周期幅值2
    PL1 = 18.866*3600*24 # 动态边界模型参数1
    PL2 = -0.616*3600*24 # 动态边界模型参数2
    Dp = 0.8             # 能量桩直径 m
    H = 8                # 能量桩深度 m
    DD = 3               # 能量桩间距 m
    alpha = 1*10**-5  # 热膨胀系数，/℃
    Ep = 3*10**7  # 能量桩弹性模量 kPa
    Ls = 2               # 第一个圆环至地面距离 m
    D = Dp - 0.1         # 圆环直径 m
    Rs = D / 2           # 圆环半径 m
    Nc = math.floor((H - Ls) / Lp) + 1  # 圆环数量
    kp = 0.4             # 管材导热系数 W/(m·K)
    kb = 2               # 回填材料导热系数 W/(m·K)
    hr = 1000            # 流体与管内壁对流换热系数 W/(m²·K)
    cw = 4200            # 水的比热容 J/(kg·K)
    M = np.pi * ri ** 2 * 1000 * v  # 单根管道流量 kg/h
    # ---- 体积流量 ----
    qv = round(math.pi * ri**2 * v * 3600, 2)  # 单根管道体积流量 m³/h
    # ---- 雷诺数 ----
    rho = 1000  # 水的密度 kg/m³
    mu = 0.001  # 水的动力粘度 Pa·s
    Re = round((rho * v * (2 * ri)) / mu, 2)  # 雷诺数计算公式
    # ---- 埋管总长度 ----
    L_total = round(Np * (Nc * np.pi * D + Nc * Lp), 2)
    # ---- 响应点坐标初始化 ----
    z = np.zeros(Nc)
    z[0] = Ls
    for i in range(1, Nc):
        z[i] = z[i-1] + Lp
    xx = [Dp / 2, DD - Dp / 2, DD + Dp / 2, Dp / 2, DD - Dp / 2, DD + Dp / 2]
    yy = [0, 0, 0, DD, DD, DD]
    zz = np.linspace(Ls, H, n)
    if not np.any(np.isclose(zz, H/2)):
        zz = np.append(zz, H/2)
        zz = np.sort(np.unique(zz))
    # ---- 数据读取 ----
    df = pd.read_excel("results/data_result.xlsx", header=None)
    Q = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy()
    Q = Q[~np.isnan(Q)]  # 去除NaN，确保Q全为数值型
    # --- 能效比 ---
    def compute_s_raw(Dp, rp, ri, hr, kp, kb, ks, Rs, DD, M, cw, H, Qr=0.1):
        rb = Dp / 2
        # 单支路热阻
        Rp = 1 / (2 * np.pi * kp) * np.log(rp / ri) + 1 / (2 * np.pi * ri * hr)
        # 热阻矩阵分量
        R11 = 1 / (2 * np.pi * kb) * np.log(Rs / rp) + Rp
        Rr = 1 / (2 * np.pi * kb) * np.log(rb / Rs)
        R22 = Rr + Rp
        R12 = 1 / (2 * np.pi * kb) * np.log(Rs / rp) + 1 / (2 * np.pi * ks) * np.log(DD / rb)
        # 换算为热容比形式
        r11 = M * cw * R11 / H
        r22 = M * cw * R22 / H
        r12 = M * cw * R12 / H
        # 耦合参数
        beta = np.sqrt((1 / (r11 * r22)) + (1 / (r12 * r11)) + (1 / (r22 * r12)))
        f1 = np.cosh(beta) - (1 / beta) * ((1 / r11) + (1 / r12)) * np.sinh(beta)
        f2 = (1 / (beta * r12)) * np.sinh(beta)
        f3 = np.cosh(beta) + (1 / beta) * ((1 / r22) + (1 / r12)) * np.sinh(beta)
        s = (f3 - f1 - 2 * f2) / (f3 - f2) + Qr
        return s
    s_raw = compute_s_raw(Dp, rp, ri, hr, kp, kb, ks, Rs, DD, M, cw, H)

    # print(f"螺旋管能效比 s_raw 近似值: {s_raw:.4f}")
    # --- 修正公式 ---
    def s_corrected(s_raw, H, H_limit=40, s_min=0.4):
        """
        修正 s_raw，使其在 H <= 40 时随着 H 增加而指数增长。
        H > 40 时不修正，直接返回 s_raw。
        
        参数：
        - s_raw: 原始效能（对应 H=40m）
        - H: 当前深度
        - H_limit: 修正上限深度（默认 40m）
        - s_min: 最小效能值（默认 0.4）
        """
        if H >= H_limit:
            return s_raw  # 超过40m不修正

        # 拟合指数关系：s(H) = s_min * exp(k * H)
        # 满足：s(H_limit) = s_raw → 求解 k
        s_line = compute_s_raw(Dp, rp, ri, hr, kp, kb, ks, Rs, DD, M, cw, H_limit)
        k = np.log(s_line / s_min) / H_limit
        s = s_min * np.exp(k * H)
        
        # 保证最终值不超过 s_line
        return np.clip(s, s_min, s_line)

    s = s_corrected(s_raw, H)
    # print(f"螺旋管能效比 s_corrected 近似值: {s:.4f}")
    
    
    # --- 向量化的动态边界模型及相关函数 ---

    @jit(nopython=True)
    def dynamic_model(t, z):
        """向量化版本的动态边界模型"""
        sqrt_pi_as_tp = np.sqrt(np.pi / as_ / tp)
        sqrt_2pi_as_tp = np.sqrt(2 * np.pi / as_ / tp)
        
        term1 = Tamp1 * np.exp(-z * sqrt_pi_as_tp) * \
                np.cos(2 * np.pi / tp * (t + tt0 - PL1) - z * sqrt_pi_as_tp)
        term2 = Tamp2 * np.exp(-z * sqrt_2pi_as_tp) * \
                np.cos(4 * np.pi / tp * (t + tt0 - PL2) - z * sqrt_2pi_as_tp)
        return Tsavg - term1 - term2

    @jit(nopython=True)
    def T_Initial(tt, z):
        """优化版本的初始温度计算"""
        T_initial = np.zeros(len(tt))
        Df_max = 20
        Df_vals = np.linspace(0, Df_max, 100)
        
        # 预计算常用值
        sqrt_as_pi = np.sqrt(as_ * np.pi)
        
        for i, t in enumerate(tt):
            if t == 0:
                T_initial[i] = dynamic_model(0, z)
            else:
                # 向量化计算
                temp_vals = np.array([dynamic_model(0, Df) for Df in Df_vals])
                exponent1 = -(Df_vals - z)**2 / (4 * as_ * t)
                exponent2 = -(Df_vals + z)**2 / (4 * as_ * t)
                integrand = temp_vals * (np.exp(exponent1) - np.exp(exponent2)) / (2 * sqrt_as_pi * np.sqrt(t))
                T_initial[i] = np.trapezoid(integrand, Df_vals)
        
        return T_initial

    @jit(nopython=True)
    def T_Gs(tt, z):
        """优化版本的边界响应计算"""
        T_gs = np.zeros(len(tt))
        sqrt_as_pi = np.sqrt(as_ * np.pi)
        
        for i, t in enumerate(tt):
            if t == 0:
                T_gs[i] = 0
                continue
                
            Dt_vals = np.linspace(0, t, 100)
            gs_vals = np.zeros(len(Dt_vals))
            
            for j, Dt in enumerate(Dt_vals):
                if t == Dt:
                    gs_vals[j] = 0
                else:
                    term = dynamic_model(Dt, 0)
                    exponent3 = -z ** 2 / (4 * as_ * (t - Dt))
                    gs_vals[j] = term * np.exp(exponent3) * z / (2 * sqrt_as_pi) / (t - Dt) ** 1.5
            
            T_gs[i] = np.trapezoid(gs_vals, Dt_vals)
        
        return T_gs

    T_initial = []
    T_gs = []

    T_initial = np.array([T_Initial(tt, z_i) for z_i in zz])
    T_gs = np.array([T_Gs(tt, z_i) for z_i in zz])

    # --- 核积分函数 (circle + line), 用于并行任务 ---
    from joblib import Parallel, delayed
    import multiprocessing

    @jit(nopython=True)
    def circle(w, t, xx, yy, zz, Nc, D, as_, z):
        """并行化的圆形核函数"""
        term = 0.0
        for i in range(Nc):
            dx = D / 2 * np.cos(w) - xx
            dy = D / 2 * np.sin(w) - yy
            dz_minus = z[i] - zz
            dz_plus  = z[i] + zz
            r_minus = np.sqrt(dx**2 + dy**2 + dz_minus**2)
            r_plus  = np.sqrt(dx**2 + dy**2 + dz_plus**2)

            if r_minus < 1e-10 or r_plus < 1e-10 or t == 0.0:
                continue

            term += 1 / r_minus * erfc(r_minus / (2 * np.sqrt(as_ * t))) \
                - 1 / r_plus  * erfc(r_plus  / (2 * np.sqrt(as_ * t)))
        return term

    @jit(nopython=True)
    def line(h, t, xx, yy, zz, D, as_):
        """并行化的直线核函数"""
        dx = xx
        dy = yy
        dz_minus = h - zz
        dz_plus  = h + zz
        r_minus1 = np.sqrt(dx**2 + dy**2 + dz_minus**2)
        r_plus1  = np.sqrt(dx**2 + dy**2 + dz_plus**2)

        if r_minus1 < 1e-10 or r_plus1 < 1e-10 or t == 0.0:
            return 0.0

        return 1 / r_minus1 * erfc(r_minus1 / (2 * np.sqrt(as_ * t))) \
            - 1 / r_plus1  * erfc(r_plus1  / (2 * np.sqrt(as_ * t)))

    def compute_circle_integrals(t, xx, yy, zz, Nc, D, as_, z):
        w_vals = np.linspace(0, 2 * np.pi, 100)
        vals = np.array([circle(w, t, xx, yy, zz, Nc, D, as_, z) for w in w_vals])
        return np.trapezoid(vals, w_vals)

    def compute_line_integrals(t, xx, yy, zz, D, as_, z_range):
        h_vals = np.linspace(z_range[0], z_range[1], 100)
        vals = np.array([line(h, t, xx, yy, zz, D, as_) for h in h_vals])
        return np.trapezoid(vals, h_vals)

    # 0715版本的 (下面有个今天新写的zz并行的版本，可以二选一)
    def compute_kernels_all_points(xx_list, yy_list, zz_list, tt, z, Nc, D, as_):
        """针对5个z点，每个z点6个响应点，分别计算circle/line核积分 
        返回shape: (5, 6, len(tt))"""
        n_z = len(zz_list)         # zz_list 其实是 5 个点
        n_points = len(xx_list)    # 6 个响应点
        n_times = len(tt)
        
        circle_results = np.zeros((n_z, n_points, n_times))
        line_results = np.zeros((n_z, n_points, n_times))

        # print('并行预计算 circle / line 积分核...')
        for z_idx, zz0 in enumerate(zz_list):
            # print(f'  -> z点 {z_idx+1}/{n_z}')
            for resp_idx in range(n_points):
                xx, yy = xx_list[resp_idx], yy_list[resp_idx]
                
                circle_result = Parallel(n_jobs=-1)(
                    delayed(compute_circle_integrals)(t, xx, yy, zz0, Nc, D, as_, z) for t in tt
                )
                line_result = Parallel(n_jobs=-1)(
                    delayed(compute_line_integrals)(t, xx, yy, zz0, D, as_, [z[0], z[-1]]) for t in tt
                )
                circle_results[z_idx, resp_idx, :] = circle_result
                line_results[z_idx, resp_idx, :] = line_result

        # print('✅ 核积分计算完成')
        return circle_results, line_results
    circle_results, line_results = compute_kernels_all_points(xx, yy, zz, tt, z, Nc, D, as_)

    # 初始条件计算

    # 计算建筑热负荷和冷负荷 
    Q1 = np.zeros(ttmax)
    q = np.zeros(ttmax)
    T_circle = np.zeros((len(zz), ttmax))
    T_line = np.zeros((len(zz), ttmax))
    T_point = np.zeros((len(zz), ttmax))
    all_integral_circle_result = np.zeros((len(zz), 6))
    all_integral_line_result = np.zeros((len(zz), 6))

    T_in = np.zeros(ttmax)
    T_out = np.zeros(ttmax+1)
    COP = np.zeros(ttmax)
    PLR = np.zeros(ttmax)
    z_top    = np.zeros(ttmax)   # 桩顶位移
    sigma_max = np.zeros(ttmax)  # 最大应力

    model1 = xgb.XGBRegressor()
    model1.load_model("COP_heating.json")

    model2 = xgb.XGBRegressor()
    model2.load_model("COP_colding.json")

    idx_mid = np.argmin(np.abs(zz - H/2))
    T_out[0] = T_initial[idx_mid][0]

    from tqdm import tqdm
    n_z = len(zz) # n个深度
    n_points = len(xx) # 6个响应点

    for i in range(ttmax):
        # 热负荷输入
        if Q[i] < 0:  # 供暖
            PLR[i] = Q[i] / min(Q)
            X_input1 = np.array([[T_out[i], PLR[i]]])
            COP[i] = model1.predict(X_input1)[0]
            Q1[i] = Q[i] * (1 - 1 / COP[i]) if COP[i] > 0 else 0
        elif Q[i] > 0:  # 制冷
            PLR[i] = Q[i] / max(Q)
            X_input2 = np.array([[T_out[i], PLR[i]]])
            COP[i] = model2.predict(X_input2)[0]
            Q1[i] = Q[i] * (1 + 1 / COP[i]) if COP[i] > 0 else 0
        else:  # 过渡季
            COP[i] = 0
            Q1[i] = 0

        q[i] = Q1[i] * 1000 / L_total

        # 卷积计算 dq（负荷变化）
        dq = np.diff(q[:i+1], prepend=0)

        for z_idx in range(n_z):
            for a in range(n_points):
                all_integral_circle_result[z_idx, a] = np.dot(dq[::-1], circle_results[z_idx, a, :i+1])
                all_integral_line_result[z_idx, a] = np.dot(dq[::-1], line_results[z_idx, a, :i+1])
            # 修正：只对数组赋值
            T_circle[z_idx, i] = (D / 2) / (4 * np.pi * ks) * (
                all_integral_circle_result[z_idx, 0] + all_integral_circle_result[z_idx, 1] +
                all_integral_circle_result[z_idx, 2] + 2 * all_integral_circle_result[z_idx, 3] +
                2 * all_integral_circle_result[z_idx, 4] + all_integral_circle_result[z_idx, 5]
            )
            T_line[z_idx, i] = 1 / (4 * np.pi * ks) * (
                all_integral_line_result[z_idx, 0] + all_integral_line_result[z_idx, 1] +
                all_integral_line_result[z_idx, 2] + 2 * all_integral_line_result[z_idx, 3] +
                2 * all_integral_line_result[z_idx, 4] + all_integral_line_result[z_idx, 5]
            )
            T_point[z_idx, i] = T_circle[z_idx, i] + T_line[z_idx, i] + T_initial[z_idx, i] + T_gs[z_idx, i]
        # 只用H/2深度的T_point更新T_in/T_out
        T_in[i] = T_out[i] + (Q1[i] * 1000 / Np) / (M * cw)
        T_out[i+1] = T_point[idx_mid, i] + (Q1[i] * 1000 / Np) / (s * M * cw) - (Q1[i] * 1000 / Np) / (M * cw)
            # ---------- 能量桩力学计算 ----------
        delta_T = np.ptp(T_point[:, i])               # 当前时刻全部深度点之间的最大温差
        epsilon_free = alpha * delta_T                # 自由热应变
        z_top[i] = epsilon_free * H / 2               # 桩顶位移（沉降） 单位为m
        sigma_max[i] = Ep * (epsilon_free - z_top[i] / H)  # 最大应力 单位为 kPa
        
    T_out = T_out[:-1]

    # --- 结果统计 ---
    z_top_max = np.max(z_top) # unit: m
    sigma_max_max = np.max(sigma_max)
    T_out_max = np.max(T_out)
    T_out_min = np.min(T_out)
    COP_heating_avg = np.mean(COP[Q < 0]).round(2) if np.any(Q < 0) else 0
    COP_cooling_avg = np.mean(COP[Q > 0]).round(2) if np.any(Q > 0) else 0
    
    hours = np.arange(ttmax)
    data = {
        'hours': hours,
        'Q_ep': Q1,
        'COP': COP,
        'T_out': T_out,
        'T_in': T_in,
        'T_initial_z0': T_initial[0],
        'T_gs_z0': T_gs[0],
        'z_top': z_top,
        'sigma_max': sigma_max
    }
    for idx, z_val in enumerate(zz):
        data[f'桩壁温度_{z_val:.2f}m'] = T_point[idx]
    df_result = pd.DataFrame(data)
    df_result['Q'] = Q
    #  ===================== 冷热不平衡率计算 =====================
    Q_heating = abs(df_result[df_result['Q_ep'] < 0]['Q_ep'].sum())
    Q_cooling = abs(df_result[df_result['Q_ep'] > 0]['Q_ep'].sum())

    den = max(Q_heating, Q_cooling)
    if den == 0:          # 全年既无供冷也无供热，理论上 η 无意义
        unbalance_rate = 0.0
    else:
        unbalance_rate = abs(Q_heating - Q_cooling) / den
    
    # ===================== LCOE计算 =====================
    k_d = [5.3, 6.5, 8.2, 11.6, 14, 35]  # 管径造价系数，单位为元/m

    rp = rp  # 你可以手动指定rp或用已有变量
    if rp not in r_pile:
        raise ValueError(f"管径{rp}不在预定义列表{r_pile}中")

    # print('能量桩群内螺旋管总长L_total（m）:', round(L_total,2))
    # print('能量桩数量Np（个）:', Np)
    # print('能量桩外半径rp（m）:', rp)
    # print('每米PE管造价系数k_d（CNY/m）:', round(k_d[rp_idx],2))

    # ---- 根据 capacity_max 确定热泵初投资 HP_cost ----
    HP_cost = 120000  # 热泵初投资 CNY

    # ---- CAPEX 更新：管柱造价 + 热泵 ----
    CAPEX_pipe = round(k_d[r_pile.index(rp)] * L_total, 2)
    CAPEX = round(CAPEX_pipe + HP_cost, 2)
    # print('总投资CAPEX（CNY，含热泵）:', CAPEX)

    # CAPEX计算
    # CAPEX = round(k_d[r_pile.index(rp)] * L_total, 2)
    # print('总投资CAPEX（CNY）:', CAPEX)

    Q = df_result['Q'].values
    Q_ep = df_result['Q_ep'].values
    COP = df_result['COP'].values
    s = s

    # 年度耗电量计算（COP为0时耗电量设为0，避免nan和inf）
    Q = np.nan_to_num(Q, nan=0.0)
    COP = np.nan_to_num(COP, nan=0.0, posinf=0.0, neginf=0.0)
    Q_div_COP = np.zeros_like(Q)
    mask = COP > 0
    Q_div_COP[mask] = Q[mask] / COP[mask]
    # COP为0时耗电量直接为0
    Q_annual_el = round(np.sum(np.abs(Q_div_COP)), 2)
    # print('年度耗电Q_annual_el(kWh):', Q_annual_el)

    # 能量桩年换热量计算
    Q_annual_ep = round(np.sum(np.abs(Q_ep)) * s, 2)
    # print('能量桩年度换热Q_annual_ep(kWh):', Q_annual_ep)

    C_elec = 0.57
    OPEX = round(Q_annual_el * C_elec, 2)
    # print('每年运行费用OPEX（CNY）:', OPEX)
    MPEX = round(0.01 * OPEX, 2)
    # print('每年维护费用MPEX（CNY）:', MPEX)

    lifetime = 20
    gamma = 0.05
    cost_pv = CAPEX
    energy_pv = 0
    for y in range(1, lifetime + 1):
        cost_pv += round((OPEX + MPEX) / (1 + gamma)**y, 2)
        energy_pv += round(Q_annual_ep / (1 + gamma)**y, 2)
        # print(f'Year {y}: cost_pv={round(cost_pv,2)}, energy_pv={round(energy_pv,2)}')

    if energy_pv != 0:
        LCOE = round(cost_pv / energy_pv, 2)
    else:
        LCOE = float('inf')
    # print(f"管径 {rp} m 的 LCOE: {LCOE:.2f} 元/kWh")

    # 返回结果时全部转为Python float
    return {
        'LCOE': float(LCOE),
        'rp': float(rp),
        'ri': float(ri),
        'qv': float(qv),
        'Re': float(Re),
        's': float(np.round(s, 2)),
        'z_top_max': float(np.round(z_top_max*1000, 2)), # 返回桩顶位移，单位为mm
        'sigma_max_max': float(np.round(sigma_max_max, 2)),
        'T_out_max': float(np.round(T_out_max, 2)),
        'T_out_min': float(np.round(T_out_min, 2)),
        'unbalance_rate': float(np.round(unbalance_rate, 2)),
        'COP_heating_avg': float(COP_heating_avg),
        'COP_cooling_avg': float(COP_cooling_avg)
    }
