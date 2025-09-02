import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit import session_state
import random
import os
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
import math
from math import erfc
from numba import jit
import xgboost as xgb
import io
# import os
import datetime
#streamlit run mystreamlit.py

# 多页面逻辑：0为负荷上传，1为参数设定 初始显示page 0
if 'page' not in st.session_state:
    st.session_state['page'] = 0

# 页面总数（根据notebook cell数量调整）
total_pages = 13


# 修改页面导航函数
def page_nav():
    # 使用空容器创建安全的导航区域
    nav_container = st.container()

    with nav_container:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.session_state['page'] > 0:
                if st.button('上一页', key=f"prev_{st.session_state['page']}"):
                    st.session_state['page'] -= 1
                    st.rerun()
        with col3:
            if st.session_state['page'] < total_pages - 1:
                if st.button('下一页', key=f"next_{st.session_state['page']}"):
                    st.session_state['page'] += 1
                    st.rerun()
        st.markdown("---")

# 功能三的页面导航函数
def f3_page_nav():
    # 使用空容器创建安全的导航区域
    nav_container = st.container()

    with nav_container:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if st.session_state.f3_page > 0:
                st.button("上一页", on_click=lambda: st.session_state.update(f3_page=st.session_state.f3_page-1))
        with col2:
            if st.button("返回首页"):
                st.session_state['page'] = 0
                st.rerun()
        with col4:
            if st.session_state.f3_page < 5:
                st.button("下一页", on_click=lambda: st.session_state.update(f3_page=st.session_state.f3_page+1))
        st.markdown("---")

st.set_page_config(layout="centered")

#首页
if st.session_state['page'] == 0:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state['page'] = 0  # 保证首页正常显示
    st.title('能量桩热-力性能全年逐时模拟系统')
    st.markdown("""
    ## 欢迎使用能量桩模拟系统
    """)
    st.image("tech_road.jpg", caption="系统开发技术路线图")
    st.markdown("""
    请选择您需要的功能：
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('功能一\n\n能量桩全年热力性能计算',
                     on_click=lambda: st.session_state.update(page=1),help='包含负荷上传、参数设定、模拟计算和结果可视化'):
            pass # 跳转到原功能一的第一个页面

    with col2:
        # st.button('功能二\n\n能量桩系统运行优化')
        # 添加功能二按钮
        if st.button('功能二\n\n能量桩参数优化',
                     on_click=lambda: st.session_state.update(page=10),
                     help='使用代理模型与遗传算法优化能量桩参数'):
            st.session_state['page'] = 12 # 跳转到功能二的页面
    with col3:
        if st.button('功能三\n\n能量桩系统运行策略优化'):
            st.session_state['page'] = 13

    st.markdown("---")
    st.info("""
    **功能说明：**
    - **功能一**：完整的能量桩全年热-力性能指标模拟计算流程
    - **功能二**：能量桩设计参数优化（需要在功能一运行完成后进行）
    - **功能三**：基于稀疏识别SINDy的模型预测控制（研究展望）
    """)
    # st.markdown('<span style="color:red; font-weight:bold;">- 若进行平台功能使用体验，建议用户使用默认模板、默认参数进行提交和体验</span>', unsafe_allow_html=True)

elif st.session_state['page'] == 1:
    if st.button('返回首页', key='back_home_X'):
            st.session_state['page'] = 0
            st.rerun()

    st.title("1. 建筑冷热负荷全年逐时数据加载")
    st.markdown("### 在本页面进行建筑冷热负荷全年逐时数据上传与加载")

    # ===== 左侧栏 =====
    with st.sidebar.form("file_form"):
        st.subheader("📂 文件操作")
        st.markdown("""
        **说明：**
        - 上传全年8761小时逐时负荷数据 (CSV/Excel)
        - 或直接使用默认文件体验
        """)
        st.info("文件要求：\n- 必须为8761行、1列数据。\n- 第1行为1月1日0点，第8761行为次年1月1日0点。")
        st.markdown("""
        **⚠️ 选择默认使用文件或选择上传文件后请点击【确认选择】按钮。**
        """)

        data_option = st.radio("请选择数据来源：", ("使用默认数据", "上传文件"))
        uploaded_file = None
        if data_option == "上传文件":
            uploaded_file = st.file_uploader("上传文件 (CSV/Excel)", type=["csv", "xlsx"])

        # 保存选择到 session_state
        submitted = st.form_submit_button("确认选择")
        if submitted:
            st.session_state['data_option'] = data_option
            st.session_state['uploaded_file'] = uploaded_file

    # ===== 主界面 =====
    df = None
    st.markdown("### 📂 文件加载")
    load_button = st.button("🚀 点击进行文件加载")

    if load_button:
        option = st.session_state.get('data_option', None)
        uploaded_file = st.session_state.get('uploaded_file', None)

        if option == "使用默认数据":
            try:
                df = pd.read_excel("能量桩负荷数据.xlsx", header=None)
                df.columns = ["Load"]
                st.success("✅ 已加载默认建筑负荷文件：能量桩负荷数据.xlsx")
            except Exception as e:
                st.error(f"❌ 默认文件加载失败: {e}")

        elif option == "上传文件":
            if uploaded_file is None:
                st.warning("⚠️ 请先上传文件！")
            else:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file, header=None)
                    else:
                        df = pd.read_excel(uploaded_file, header=None)
                    df.columns = ["Load"]
                    st.success(f"✅ 文件上传成功: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ 文件读取失败: {e}")
        else:
            st.warning("⚠️ 请先在左侧选择数据来源并确认！")

    # ===== 文件预览 =====
    with st.expander("👀 点击展开文件预览", expanded=False):
        if df is None:
            st.warning("⚠️ 请先点击上方【加载文件】按钮。")
        else:
            df_preview = df.head(5).rename(columns={df.columns[0]: "Load [kW]"})
            df_preview.index.name = "Timestamp"
            st.dataframe(df_preview, use_container_width=True)

    # ===== 全年曲线 =====
    with st.expander("📈 点击展开全年曲线可视化", expanded=False):
        if df is None:
            st.warning("⚠️ 请先点击上方【加载文件】按钮。")
        else:
            load_data = df['Load'].values
            if len(load_data) != 8761:
                st.error(
                    f'❌ 检测到负荷数据长度为 {len(load_data)}，'
                    '但应为 8761（全年逐小时+次年1月1日0点）。请检查文件！'
                )
            else:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(load_data, color='blue', linewidth=1)
                ax.set_xlabel('Hours')
                ax.set_ylabel('Load [kW]')
                ax.set_title('Annual Hourly Load Curve')
                st.pyplot(fig)
                
            st.success('已成功绘制全年负荷曲线！')
            st.session_state['Q'] = load_data.astype(float).copy() if isinstance(load_data, np.ndarray) else np.array(load_data, dtype=float)
        
    page_nav()

# elif st.session_state['page'] == 2:
#         if st.button('返回首页', key='back_home_X'):
#             st.session_state['page'] = 0
#             st.rerun()
#         st.title('2. 能量桩全年负荷模拟与参数设定')
#         st.markdown("""
#     **本页内容：**
#     - 请在页面左侧输入土壤及螺旋管的相关设计参数
#         """)
#         st.markdown('<span style="color:red; font-weight:bold;">- 输入完成后将左侧参数栏拉至最底部点击"提交参数"按钮</span>', unsafe_allow_html=True)
#         st.markdown('<span style="color:red; font-weight:bold;">- 本平台支持垂直方向设定多个均匀计算节点n作为桩壁温度输出，但设置n时需保证桩壁中点被包含</span>', unsafe_allow_html=True)

#         # 预设城市数据（从Excel中读取）
#         try:
#             # 从提供的Excel内容中创建DataFrame
#             city_df = pd.DataFrame({
#                 "城市": ["大连", "郑州", "兰州", "武汉", "长沙", "杭州", "合肥", "西安", "太原"],
#                 "供暖季开始时间": [45966, 45976, 45962, 45976, 45976, 45996, 45996, 45976, 45962],
#                 "Tsavg": [12.25, 15.46, 12.58, 17.63, 17.15, 16.81, 16.08, 12.31, 12.23],
#                 "Tamp1": [13.66, 14.16, 13.41, 13.25, 12.99, 13.24, 13.36, 12.26, 13.86],
#                 "Tamp2": [0.993, 0.926, 0.933, 1.538, 1.308, 0.764, 1.277, 0.97, 0.308],
#                 "PL1": [19.167, 13.796, 12.685, 17.454, 20.868, 21.354, 18.194, 13.032, 10.485],
#                 "PL2": [-11.655, -17.384, -17.465, -20.602, -0.979, -1.99, 10.882, -0.015, -8.76]
#             })


#             # 将Excel日期序列转换为实际日期（只保留月日，忽略年份）
#             def excel_num_to_date(num):
#                 base_date = pd.Timestamp('1899-12-30')  # Excel的起始日期
#                 date_obj = base_date + pd.Timedelta(days=num)
#                 # 只保留月日部分，年份设为2023（统一基准年）
#                 return datetime.date(2023, date_obj.month, date_obj.day)


#             city_data = {}
#             for _, row in city_df.iterrows():
#                 city_name = row['城市']
#                 heating_date = excel_num_to_date(row['供暖季开始时间'])
#                 city_data[city_name] = {
#                     "供暖季开始日期": heating_date,
#                     "Tsavg": row['Tsavg'],
#                     "Tamp1": row['Tamp1'],
#                     "Tamp2": row['Tamp2'],
#                     "PL1": row['PL1'],
#                     "PL2": row['PL2']
#                 }
#         except Exception as e:
#             st.error(f"加载预设城市数据时出错: {e}")
#             city_data = {}

#         # 城市选择或自定义选项
#         use_preset = st.radio("请选择参数输入方式：",
#                               ("使用预设城市参数", "手动输入所有参数"),
#                               index=0)

#         selected_city = None
#         city_params = None

#         if use_preset == "使用预设城市参数" and city_data:
#             selected_city = st.selectbox("选择预设城市", list(city_data.keys()))
#             city_params = city_data[selected_city]

#             # 显示预设参数
#             st.info(f"已选择 **{selected_city}** 的预设参数:")
#             st.write(f"- 供暖季开始日期: {city_params['供暖季开始日期'].strftime('%m-%d')}")
#             st.write(f"- 年平均地温 Tsavg: {city_params['Tsavg']}°C")
#             st.write(f"- 年周期幅值1 Tamp1: {city_params['Tamp1']}")
#             st.write(f"- 年周期幅值2 Tamp2: {city_params['Tamp2']}")
#             st.write(f"- 动态边界模型参数1 PL1: {city_params['PL1']}天")
#             st.write(f"- 动态边界模型参数2 PL2: {city_params['PL2']}天")

#             # 使用预设的供暖季开始日期
#             heating_date = city_params['供暖季开始日期']
#         else:
#             # 供暖日期选择器（手动输入模式）- 使用2023年作为基准年
#             st.subheader("供暖季开始日期 (仅考虑月日)")
#             # 默认日期使用11月5日，保证tt0与ipynb一致
#             default_date = datetime.date(2023, 11, 5)
#             heating_date = st.date_input("选择供暖季开始日期", value=default_date)
#             # 确保年份为2023（统一基准年）
#             heating_date = datetime.date(2023, heating_date.month, heating_date.day)

#         # 计算tt0（供暖季开始时间秒数）- 使用统一的2023年1月1日作为基准
#         jan1 = datetime.date(2023, 1, 1)
#         tt0_days = (heating_date - jan1).days
#         tt0 = tt0_days * 24 * 3600  # 转换为秒数
#         tp = 31536000  # 一年秒数

#         st.info(f"供暖季开始时间 tt0 计算值: {tt0} 秒 (从1月1日开始 {tt0_days} 天)")

#         with st.sidebar.form("param_form"):
#             st.header("参数设定")

#             # 新增管式类型选择
#             st.subheader("埋管类型")
#             pipe_type = st.selectbox("选择埋管类型", ["螺旋管", "U型管", "W型管"], index=0)

#             # 根据管式类型显示不同参数
#             if pipe_type == "螺旋管":
#                 # 螺旋管参数
#                 st.subheader("螺旋管参数")
#                 Ls = st.number_input("第一个圆环至地面距离 Ls (m)", value=2, help="该数值可以取：2")
#                 delta_d = st.number_input("螺旋管圆环距离桩壁间距 delta d (m)", value=0.05, help="该数值可以取：0.05")
#                 Lp = st.number_input("螺距 Lp (m)", value=0.1, help="该数值可以取：0.1")

#                 # 管半径选择
#                 # r_pile_options = [0.010, 0.0125, 0.016, 0.02, 0.025, 0.0315]
#                 # default_rp_index = r_pile_options.index(0.0125) if 0.0125 in r_pile_options else 0
#                 # rp = st.selectbox("管外半径 rp (m)", options=r_pile_options, index=default_rp_index)

#             elif pipe_type == "U型管":
#                 # U型管参数
#                 st.subheader("U型管参数")
#                 d = st.number_input("U型管间距 d (m)", value=0.5, help="该数值可以取：0.5")

#                 # 管半径选择
#                 # r_pile_options = [0.010, 0.0125, 0.016, 0.02, 0.025, 0.0315]
#                 # default_rp_index = r_pile_options.index(0.0125) if 0.0125 in r_pile_options else 0
#                 # rp = st.selectbox("管外半径 rp (m)", options=r_pile_options, index=default_rp_index)

#             elif pipe_type == "W型管":
#                 # W型管参数
#                 st.subheader("W型管参数")
#                 d = st.number_input("U型管间距 d (m)", value=0.5, help="该数值可以取：0.5")
#                 du = st.number_input("两对U型管间距 du (m)", value=0.3, help="该数值可以取：0.3")

#                 # 管半径选择
#                 # r_pile_options = [0.010, 0.0125, 0.016, 0.02, 0.025, 0.0315]
#                 # default_rp_index = r_pile_options.index(0.0125) if 0.0125 in r_pile_options else 0
#                 # rp = st.selectbox("管外半径 rp (m)", options=r_pile_options, index=default_rp_index)

#             # 通用参数
#             st.subheader("通用参数")
#             # 将 rp 修改为下拉选择框
#             r_pile_options = [0.010, 0.0125, 0.016, 0.02, 0.025, 0.0315]
#             default_rp_index = r_pile_options.index(0.0125) if 0.0125 in r_pile_options else 0
#             rp = st.selectbox("管外半径 rp (m)", options=r_pile_options, index=default_rp_index)
#             ri = st.number_input("管内半径 ri (m)", value=0.0102, help="该数值可以取：0.0102")
#             kp = st.number_input("管材导热系数 kp (W/(m·K))", value=0.4, help="该数值可以取：0.4")
#             kb = st.number_input("回填材料导热系数 kb (W/(m·K))", value=2, help="该数值可以取：2")
#             hr = st.number_input("流体与管内壁对流换热系数 hr (W/(m²·K))", value=1000, help="该数值可以取：1000")

#             # 地温参数部分
#             if use_preset == "使用预设城市参数" and city_params:
#                 st.subheader(f"{selected_city} - 预设参数")
#                 Tsavg = st.number_input("年平均地温 Tsavg (℃)", value=city_params['Tsavg'], disabled=True)
#                 Tamp1 = st.number_input("年周期幅值1 Tamp1", value=city_params['Tamp1'], disabled=True)
#                 Tamp2 = st.number_input("年周期幅值2 Tamp2", value=city_params['Tamp2'], disabled=True)
#                 PL1_days = st.number_input("动态边界模型参数1 PL1", value=city_params['PL1'], disabled=True)
#                 PL2_days = st.number_input("动态边界模型参数2 PL2", value=city_params['PL2'], disabled=True)
#             else:
#                 st.subheader("手动输入参数")
#                 Tsavg = st.number_input("年平均地温 Tsavg (℃)", value=14.02, help="该数值可以取：14.02")
#                 Tamp1 = st.number_input("年周期幅值1 Tamp1", value=14.69, help="该数值可以取：14.69")
#                 Tamp2 = st.number_input("年周期幅值2 Tamp2", value=1.173, help="该数值可以取：1.173")
#                 PL1_days = st.number_input("动态边界模型参数1 PL1", value=18.866, help="该数值可以取：18.866")
#                 PL2_days = st.number_input("动态边界模型参数2 PL2", value=-0.616, help="该数值可以取：-0.616")

#             # 转换为秒数（用于计算）
#             PL1 = PL1_days * 3600 * 24
#             PL2 = PL2_days * 3600 * 24

#             # 土壤参数
#             st.subheader("土壤参数")
#             ks = st.number_input("土壤导热系数 ks (W/(m·K))", value=2.1, help="该数值可以取：2.1")
#             Cv = st.number_input("土壤体积热容 Cv (J/(m³·K))", value=2200000, help="该数值可以取：2200000")
#             rs = st.number_input("土壤热阻系数 rs", value=0.5, help="该数值可以取：0.5")

#             # 能量桩参数
#             st.subheader("能量桩参数")
#             Dp = st.number_input("能量桩直径 Dp (m)", value=0.8, help="该数值可以取：0.8")
#             H = st.number_input("能量桩深度 H (m)", value=8, help="该数值可以取：8")
#             Np = st.number_input("能量桩个数 Np", value=150, help="该数值可以取：150")
#             DD = st.number_input("能量桩间距 DD (m)", value=3, help="该数值可以取：3")

#             # 垂直方向计算节点设置
#             st.subheader("垂直方向计算节点")
#             n = st.number_input("分层数 n（正整数）", min_value=1, value=4, step=1)
#             # Ls = st.number_input("第一个圆环至地面距离 Ls (m)", value=2, help="该数值可以取：2")
#             # delta_d = st.number_input("螺旋管圆环距离桩壁间距 delta d (m)", value=0.05, help="该数值可以取：0.05")
#             # Lp = st.number_input("螺距 Lp (m)", value=0.1, help="该数值可以取：0.1")

#             st.subheader("能量桩混凝土参数")
#             alpha = st.number_input('热膨胀系数 alpha (/℃)', value=1e-5, format="%e", help='推荐值：1e-5')
#             Ep = st.number_input('能量桩弹性模量 Ep (kPa)', value=3e7, format="%e", help='推荐值：3e7')
#             st.session_state['alpha'] = alpha
#             st.session_state['Ep'] = Ep

#             st.subheader("热泵参数")
#             v = st.number_input("流速 v (m/s)", value=0.4, help="该数值可以取：0.4")
#             cw = st.number_input("水的比热容 cw (J/(kg·K))", value=4200, help="该数值可以取：4200")

#             submit_params = st.form_submit_button("提交参数")

#         if submit_params:
#             # 计算相关变量
#             # 存储管式类型和参数
#             if 'params' not in st.session_state:
#                 st.session_state['params'] = {}
#             st.session_state['params']['pipe_type'] = pipe_type
#             if pipe_type == "螺旋管":
#                 st.session_state['params']['Ls'] = Ls
#                 st.session_state['params']['delta_d'] = delta_d
#                 st.session_state['params']['Lp'] = Lp
#                 # 计算相关变量
#                 D = Dp - 2 * delta_d  # 圆环直径
#                 Rs = D / 2           # 圆环半径 m
#                 Nc = math.floor((H - Ls) / Lp) + 1  # 圆环数量
#                 st.session_state['params']['D'] = D
#                 st.session_state['params']['Nc'] = Nc
#                 L_total = Np * (Nc * np.pi * D + Nc * Lp)
#                 # ---- 响应点坐标初始化 ----
#                 z = np.zeros(Nc)
#                 z[0] = Ls
#                 for i in range(1, Nc):
#                     z[i] = z[i - 1] + Lp
#             elif pipe_type == "U型管":
#                 st.session_state['params']['d'] = d
#                 L_total = round(Np * H * 2, 2)
#             elif pipe_type == "W型管":
#                 st.session_state['params']['d'] = d
#                 st.session_state['params']['du'] = du
#                 L_total = round(Np * H * 4, 2)

#             M = np.pi * ri ** 2 * 1000 * v  # 单根管道流量 kg/h
#             tt = np.arange(0, 3600 * 24 * 365 + 1, 3600)  # 一年每小时的时间向量，单位秒
#             ttmax = len(tt)  # 时间步数
#             qv = math.pi * ri ** 2 * v * 3600  # 单根管道体积流量 m³/h
#             rho = 1000  # 水的密度 kg/m³
#             mu = 0.001  # 水的动力粘度 Pa·s
#             Re = (rho * v * (2 * ri)) / mu  # 雷诺数计算公式

#             xx = [Dp / 2, DD - Dp / 2, DD + Dp / 2, Dp / 2, DD - Dp / 2, DD + Dp / 2]
#             yy = [0, 0, 0, DD, DD, DD]

#             # 在 [Ls, H] 区间内均分 n 个点，包含 Ls 和 H
#             # 根据管型类型拆分zz的赋值
#             if pipe_type == "螺旋管":
#                 zz = np.linspace(Ls, H, n)
#             else:
#                 zz = np.linspace(H / n, H, n)

#             # 确保包含 H/2，如果不在，则追加后排序去重
#             if not np.any(np.isclose(zz, H / 2)):
#                 zz = np.append(zz, H / 2)
#                 zz = np.sort(np.unique(zz))

#             # 打印并检查（改为st页面显示）
#             st.write('分层深度 zz:', zz)
#             if np.any(np.isclose(zz, H / 2)):
#                 st.success('zz 已包含 H/2，H/2处的响应温度将用于计算出口水温')
#             else:
#                 st.warning('zz 未包含 H/2，建议检查分层设置')

#             # 存储所有参数和中间变量到session_state，包括响应点坐标
#             params_dict = {
#                 'pipe_type': pipe_type,
#                 'ks': ks, 'Cv': Cv, 'rs': rs,
#                 'tp': tp, 'tt0': tt0,
#                 'Tsavg': Tsavg, 'Tamp1': Tamp1, 'Tamp2': Tamp2, 'PL1': PL1, 'PL2': PL2,
#                 'Dp': Dp, 'H': H, 'Np': Np, 'DD': DD,
#                 'rp': rp, 'ri': ri, 'kp': kp, 'kb': kb, 'hr': hr,
#                 'v': v, 'cw': cw, 'M': M, 'L_total': L_total,
#                 'tt': tt, 'ttmax': ttmax,
#                 'xx': xx, 'yy': yy, 'zz': zz,
#                 'n': n,  # 分层数
#                 'Q': st.session_state.get('Q', None),  # 负荷数据
#                 'alpha': st.session_state.get('alpha', 1e-5),
#                 'Ep': st.session_state.get('Ep', 3e7),
#             }
#             if pipe_type == "螺旋管":
#                 params_dict.update({'Ls': Ls, 'delta_d': delta_d, 'D': D, 'Rs': Rs, 'Lp': Lp, 'Nc': Nc,'z': z})
#             elif pipe_type == "U型管":
#                 params_dict.update({'d': d})
#             elif pipe_type == "W型管":
#                 params_dict.update({'d': d, 'du': du})
#             st.session_state['params'] = params_dict

#             st.success("参数已提交，可用于后续模拟。")
#             st.write('根据提交参数计算得到:')
#             # st.write('圆环直径 D (m):', D)
#             # st.write('圆环半径 Rs (m):', Rs)
#             # st.write('圆环数量 Nc(个)：', Nc)
#             st.write('单根管道流量 (kg/h):', M)
#             st.write('埋管总长度 L_total:', L_total)
#             st.write('分层深度 zz:', zz)
#         st.markdown(
#             "<span style='color:white; font-size:20px; font-weight:bold;'>请检查左侧参数输入是否正确，确保所有必填项已填写。</span>",
#             unsafe_allow_html=True)
#         page_nav()

elif st.session_state['page'] == 2:
        if st.button('返回首页', key='back_home_X'):
            st.session_state['page'] = 0
            st.rerun()
        st.title('2. 能量桩全年负荷模拟与参数设定')
        st.markdown("""
    **本页内容：**
    - 请在页面左侧输入土壤及螺旋管的相关设计参数
        """)
        st.markdown('<span style="color:red; font-weight:bold;">- 输入完成后将左侧参数栏拉至最底部点击"提交参数"按钮</span>', unsafe_allow_html=True)
        st.markdown('<span style="color:red; font-weight:bold;">- 本平台支持垂直方向设定多个均匀计算节点n作为桩壁温度输出，但设置n时需保证桩壁中点被包含</span>', unsafe_allow_html=True)

       # ===== 左边栏：地理参数设定 =====
        with st.sidebar:
            st.header("地理参数设定")
            st.markdown(
                "<span style='color:red; font-size:18px; font-weight:bold;'>⚠️ 请选择参数并点击底部按钮确认，否则不会生效</span>",
                unsafe_allow_html=True
            )

            # 初始化 session_state 保存的最终参数
            if "geo_params" not in st.session_state:
                st.session_state.geo_params = None

            # 预设城市数据
            city_df = pd.DataFrame({
                "城市": ["大连", "郑州", "兰州", "武汉", "长沙", "杭州", "合肥", "西安", "太原"],
                "供暖季开始时间": [45966, 45976, 45962, 45976, 45976, 45996, 45996, 45976, 45962],
                "Tsavg": [12.25, 15.46, 12.58, 17.63, 17.15, 16.81, 16.08, 12.31, 12.23],
                "Tamp1": [13.66, 14.16, 13.41, 13.25, 12.99, 13.24, 13.36, 12.26, 13.86],
                "Tamp2": [0.993, 0.926, 0.933, 1.538, 1.308, 0.764, 1.277, 0.97, 0.308],
                "PL1": [19.167, 13.796, 12.685, 17.454, 20.868, 21.354, 18.194, 13.032, 10.485],
                "PL2": [-11.655, -17.384, -17.465, -20.602, -0.979, -1.99, 10.882, -0.015, -8.76]
            })

            def excel_num_to_date(num):
                base_date = pd.Timestamp('1899-12-30')
                date_obj = base_date + pd.Timedelta(days=num)
                return datetime.date(2023, date_obj.month, date_obj.day)

            city_data = {
                row["城市"]: {
                    "供暖季开始日期": excel_num_to_date(row["供暖季开始时间"]),
                    "Tsavg": row["Tsavg"],
                    "Tamp1": row["Tamp1"],
                    "Tamp2": row["Tamp2"],
                    "PL1": row["PL1"],
                    "PL2": row["PL2"],
                }
                for _, row in city_df.iterrows()
            }

            # ===== 临时变量（输入区始终变化） =====
            use_preset = st.radio("请选择参数输入方式：",
                                ("使用预设城市参数", "手动输入所有参数"),
                                index=0)

            if use_preset == "使用预设城市参数":
                selected_city = st.selectbox("选择预设城市", list(city_data.keys()))
                city_params = city_data[selected_city]

                st.info(f"已选择 **{selected_city}** 的参数")

                Tsavg_tmp = city_params['Tsavg']
                Tamp1_tmp = city_params['Tamp1']
                Tamp2_tmp = city_params['Tamp2']
                PL1_tmp = city_params['PL1']
                PL2_tmp = city_params['PL2']
                heating_date_tmp = city_params["供暖季开始日期"]

                st.write(f"- 供暖季开始日期: {heating_date_tmp.strftime('%m-%d')}")
                st.write(f"- 年平均地温 Tsavg: {Tsavg_tmp} ℃")
                st.write(f"- 年周期幅值1 Tamp1: {Tamp1_tmp}")
                st.write(f"- 年周期幅值2 Tamp2: {Tamp2_tmp}")
                st.write(f"- 动态边界模型参数1 PL1: {PL1_tmp} 天")
                st.write(f"- 动态边界模型参数2 PL2: {PL2_tmp} 天")

            else:
                st.subheader("供暖季开始日期 (仅考虑月日)")
                default_date = datetime.date(2023, 11, 5)
                heating_date_tmp = st.date_input("选择供暖季开始日期", value=default_date)
                heating_date_tmp = datetime.date(2023, heating_date_tmp.month, heating_date_tmp.day)

                st.subheader("手动输入参数")
                Tsavg_tmp = st.number_input("年平均地温 Tsavg (℃)", value=14.02)
                Tamp1_tmp = st.number_input("年周期幅值1 Tamp1", value=14.69)
                Tamp2_tmp = st.number_input("年周期幅值2 Tamp2", value=1.173)
                PL1_tmp = st.number_input("动态边界模型参数1 PL1 (天)", value=18.866)
                PL2_tmp = st.number_input("动态边界模型参数2 PL2 (天)", value=-0.616)

            # ===== 确认按钮：只有点击才更新 session_state =====
            if st.button("确认地理参数"):
                # 转换为秒
                PL1_sec = PL1_tmp * 3600 * 24
                PL2_sec = PL2_tmp * 3600 * 24
                jan1 = datetime.date(2023, 1, 1)
                tt0_days_tmp = (heating_date_tmp - jan1).days
                tt0_tmp = tt0_days_tmp * 24 * 3600
                tp_tmp = 31536000

                # 保存到 session_state
                st.session_state.geo_params = {
                    "Tsavg": Tsavg_tmp,
                    "Tamp1": Tamp1_tmp,
                    "Tamp2": Tamp2_tmp,
                    "PL1": PL1_sec,
                    "PL2": PL2_sec,
                    "tt0": tt0_tmp,
                    "tp": tp_tmp,
                    "heating_date": heating_date_tmp,
                    "tt0_days": tt0_days_tmp
                }
                st.success("✅ 地理参数已确认")

            # ===== 读取已确认参数 =====
            if st.session_state.geo_params:
                gp = st.session_state.geo_params
                Tsavg = gp['Tsavg']
                Tamp1 = gp['Tamp1']
                Tamp2 = gp['Tamp2']
                PL1 = gp['PL1']
                PL2 = gp['PL2']
                tt0 = gp['tt0']
                tp = gp['tp']
                heating_date = gp['heating_date']
                tt0_days = gp['tt0_days']
            else:
                st.warning("⚠️ 地理参数尚未确认，请先点击 '确认地理参数'。")
                Tsavg = 14.02
                Tamp1 = 14.69
                Tamp2 = 1.173
                PL1 = 18.866 * 3600 * 24
                PL2 = -0.616 * 3600 * 24
                tt0 = 0
                tp = 31536000
                heating_date = datetime.date(2023, 11, 5)
                tt0_days = 0

        # ===== 主页面：其他参数设定 =====
        st.header("参数设定")

        # ===== 1 管型选择与特有参数 =====
        st.subheader("埋管类型")
        pipe_type = st.selectbox("选择埋管类型", ["螺旋管", "U型管", "W型管"], index=0, key="pipe_type")

        # 管型特有参数
        if pipe_type == "螺旋管":
            st.subheader("螺旋管参数")
            Ls = st.number_input("第一个圆环至地面距离 Ls (m)", value=2, key="Ls")
            delta_d = st.number_input("螺旋管圆环距离桩壁间距 delta d (m)", value=0.05, key="delta_d")
            Lp = st.number_input("螺距 Lp (m)", value=0.1, key="Lp")
        elif pipe_type == "U型管":
            st.subheader("U型管参数")
            d = st.number_input("U型管间距 d (m)", value=0.5, key="d")
        elif pipe_type == "W型管":
            st.subheader("W型管参数")
            d = st.number_input("U型管间距 d (m)", value=0.5, key="d")
            du = st.number_input("两对U型管间距 du (m)", value=0.3, key="du")

        # ===== 2 通用参数 form =====
        with st.form("param_form"):
            st.subheader("通用参数")
            rp = st.selectbox("管外半径 rp (m)", [0.010, 0.0125, 0.016, 0.02, 0.025, 0.0315], index=1, key="rp")
            ri = st.number_input("管内半径 ri (m)", value=0.0102, key="ri")
            kp = st.number_input("管材导热系数 kp", value=0.4, key="kp")
            kb = st.number_input("回填材料导热系数 kb", value=2, key="kb")
            hr = st.number_input("对流换热系数 hr", value=1000, key="hr")

            st.subheader("土壤参数")
            ks = st.number_input("土壤导热系数 ks", value=2.1, key="ks")
            Cv = st.number_input("土壤体积热容 Cv", value=2.2e6, key="Cv")
            rs = st.number_input("土壤热阻系数 rs", value=0.5, key="rs")

            st.subheader("能量桩参数")
            Dp = st.number_input("能量桩直径 Dp (m)", value=0.8, key="Dp")
            H = st.number_input("能量桩深度 H (m)", value=8, key="H")
            Np = st.number_input("能量桩个数 Np", value=150, key="Np")
            DD = st.number_input("能量桩间距 DD (m)", value=3, key="DD")

            st.subheader("垂直方向计算节点")
            n = st.number_input("分层数 n（正整数）", min_value=1, value=4, step=1, key="n")

            st.subheader("热泵参数")
            v = st.number_input("流速 v (m/s)", value=0.4, key="v")
            cw = st.number_input("水的比热容 cw (J/(kg·K))", value=4200, key="cw")

            st.subheader("混凝土参数")
            alpha = st.number_input("热膨胀系数 alpha", value=1e-5, format="%e", key="alpha")
            Ep = st.number_input("能量桩弹性模量 Ep (kPa)", value=3e7, format="%e", key="Ep")

            submit_params = st.form_submit_button("提交参数")

        # ===== 3 提交后处理 =====
        if submit_params:
            # 管型计算
            if pipe_type == "螺旋管":
                D = Dp - 2 * delta_d
                Rs = D / 2
                Nc = math.floor((H - Ls) / Lp) + 1
                L_total = Np * (Nc * np.pi * D + Nc * Lp)
                z = np.zeros(Nc)
                z[0] = Ls
                for i in range(1, Nc):
                    z[i] = z[i - 1] + Lp
            elif pipe_type == "U型管":
                L_total = round(Np * H * 2, 2)
            elif pipe_type == "W型管":
                L_total = round(Np * H * 4, 2)

            # 其他计算
            M = np.pi * ri ** 2 * 1000 * v
            tt = np.arange(0, 3600 * 24 * 365 + 1, 3600)
            ttmax = len(tt)
            xx = [Dp / 2, DD - Dp / 2, DD + Dp / 2, Dp / 2, DD - Dp / 2, DD + Dp / 2]
            yy = [0, 0, 0, DD, DD, DD]
            zz = np.linspace(Ls if pipe_type=="螺旋管" else H/n, H, n)


            # 确保包含 H/2，如果不在，则追加后排序去重
            if not np.any(np.isclose(zz, H / 2)):
                zz = np.append(zz, H / 2)
                zz = np.sort(np.unique(zz))

            # 打印并检查（改为st页面显示）
            st.write('分层深度 zz:', zz)
            if np.any(np.isclose(zz, H / 2)):
                st.success('zz 已包含 H/2，H/2处的响应温度将用于计算出口水温')
            else:
                st.warning('zz 未包含 H/2，建议检查分层设置')
            
            params_dict = {
                            'pipe_type': pipe_type,
                            'ks': ks, 'Cv': Cv, 'rs': rs,
                            'tp': tp, 'tt0': tt0,
                            'Tsavg': Tsavg, 'Tamp1': Tamp1, 'Tamp2': Tamp2, 'PL1': PL1, 'PL2': PL2,
                            'Dp': Dp, 'H': H, 'Np': Np, 'DD': DD,
                            'rp': rp, 'ri': ri, 'kp': kp, 'kb': kb, 'hr': hr,
                            'v': v, 'cw': cw, 'M': M, 'L_total': L_total,
                            'tt': tt, 'ttmax': ttmax,
                            'xx': xx, 'yy': yy, 'zz': zz,
                            'n': n,  # 分层数
                            'Q': st.session_state.get('Q', None),  # 负荷数据
                            'alpha': alpha, 'Ep': Ep
                            }

            if pipe_type == "螺旋管":
                params_dict.update({'Ls': Ls, 'delta_d': delta_d, 'D': D, 'Rs': Rs, 'Lp': Lp, 'Nc': Nc,'z': z})
            elif pipe_type == "U型管":
                params_dict.update({'d': d})
            elif pipe_type == "W型管":
                params_dict.update({'d': d, 'du': du})
            st.session_state['params'] = params_dict

            st.success("参数已提交，可用于后续模拟。")
            st.write('根据提交参数计算得到:')
            # st.write('圆环直径 D (m):', D)
            # st.write('圆环半径 Rs (m):', Rs)
            # st.write('圆环数量 Nc(个)：', Nc)
            st.write('单根管道流量 (kg/h):', M)
            st.write('埋管总长度 L_total:', L_total)
            st.write('分层深度 zz:', zz)
        
        page_nav()

elif st.session_state['page'] == 3:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    st.header("3. 能效比计算")
    st.markdown("""
**本页内容：**
- 根据设计参数计算能效比 s 及相关热阻参数
    """)
    params = st.session_state['params']

    # 获取管式类型
    pipe_type = params.get('pipe_type', '螺旋管')

    # 取参数
    Dp = params['Dp']
    H = params['H']
    Np = params['Np']
    DD = params['DD']
    rp = params['rp']
    ri = params['ri']
    kp = params['kp']
    kb = params['kb']
    hr = params['hr']
    cw = params['cw']
    M = params['M']
    ks = params['ks']
    v = params['v']

    # 根据管式类型计算能效比
    if pipe_type == "螺旋管":
        # 螺旋管参数
        try:
            D = params['D']
            Rs = params['Rs']
        except KeyError:
            st.error("缺少螺旋管参数 D或Rs，请返回参数设定页面")
            page_nav()
            st.stop()


        # 螺旋管能效比计算函数
        def compute_s_raw_spiral(Dp, rp, ri, hr, kp, kb, ks, Rs, DD, M, cw, H, Qr=0.1):
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
        
        s_raw = compute_s_raw_spiral(Dp, rp, ri, hr, kp, kb, ks, Rs, DD, M, cw, H)

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
            s_line = compute_s_raw_spiral(Dp, rp, ri, hr, kp, kb, ks, Rs, DD, M, cw, H_limit)
            k = np.log(s_line / s_min) / H_limit
            s = s_min * np.exp(k * H)
            
            # 保证最终值不超过 s_line
            return np.clip(s, s_min, s_line)

        s = s_corrected(s_raw, H)

    elif pipe_type == "U型管":
        # U型管参数
        try:
            d = params['d']
        except KeyError:
            st.error("缺少U型管参数 d，请返回参数设定页面")
            page_nav()
            st.stop()


        # U型管能效比计算函数
        def calc_s_raw_u(Dp, rp, ri, hr, kp, kb, ks, d, M, cw, H):
            rb = Dp / 2
            sig = (kb - ks) / (kb + ks)
            Rp = 1 / (2 * np.pi * kp) * np.log(rp / ri) + 1 / (2 * np.pi * ri * hr)
            R11 = 1 / (2 * np.pi * kb) * (
                    abs(np.log(rb / rp)) + sig * abs(np.log(rb ** 2 / abs(rb ** 2 - d ** 2)))
            ) + Rp
            R12 = 1 / (2 * np.pi * kb) * (
                    abs(np.log(rb / (2 * d))) + sig * abs(np.log(rb ** 2 / abs(rb ** 2 + d ** 2)))
            )
            RR1 = (R11 ** 2 - R12 ** 2) / (R11 - R12)
            RR12 = (R11 ** 2 - R12 ** 2) / R12
            r1 = M * cw * RR1 / H
            r12 = M * cw * RR12 / H
            bet = np.sqrt(1 / r1 ** 2 + 2 / (r12 * r1))
            f1 = np.cosh(bet) - (1 / bet) * (1 / r1 + 1 / r12) * np.sinh(bet)
            f2 = (1 / (bet * r12)) * np.sinh(bet)
            f3 = np.cosh(bet) + (1 / bet) * (1 / r1 + 1 / r12) * np.sinh(bet)
            return (f3 - f1 - 2 * f2) / (f3 - f2)
        s_raw = calc_s_raw_u(Dp, rp, ri, hr, kp, kb, ks, d, M, cw, H)
        
        # 深度修正函数（U）
        def s_corrected(s_raw, H, H_limit=80, s_min=0.4):
            """
            修正 s_raw，使其在 H <= 80 时随着 H 增加而指数增长。
            H > H_limit 时不修正，直接返回 s_raw。
            """
            if H >= H_limit:
                return s_raw

            # 拟合指数关系：s(H) = s_min * exp(k * H)
            s_line = s_raw  # 使用原始值作为上限
            k = np.log(s_line / s_min) / H_limit
            s = s_min * np.exp(k * H)

            return np.clip(s, s_min, s_line)
        
        s = s_corrected(s_raw, H)

    elif pipe_type == "W型管":
        # W型管参数
        try:
            d = params['d']
            du = params['du']
        except KeyError:
            st.error("缺少W型管参数 d 或 du，请返回参数设定页面")
            page_nav()
            st.stop()


        # W型管能效比计算函数
        def calc_s_raw_w(Dp, rp, ri, hr, kp, kb, ks, d, M, cw, H):
            # 使用U型管计算的两倍作为近似
            rb = Dp / 2
            sig = (kb - ks) / (kb + ks)
            Rp = 1 / (2 * np.pi * kp) * np.log(rp / ri) + 1 / (2 * np.pi * ri * hr)
            R11 = 1 / (2 * np.pi * kb) * (
                    abs(np.log(rb / rp)) + sig * abs(np.log(rb ** 2 / abs(rb ** 2 - d ** 2)))
            ) + Rp
            R12 = 1 / (2 * np.pi * kb) * (
                    abs(np.log(rb / (2 * d))) + sig * abs(np.log(rb ** 2 / abs(rb ** 2 + d ** 2)))
            )
            RR1 = (R11 ** 2 - R12 ** 2) / (R11 - R12)
            RR12 = (R11 ** 2 - R12 ** 2) / R12
            r1 = M * cw * RR1 / H
            r12 = M * cw * RR12 / H
            bet = np.sqrt(1 / r1 ** 2 + 2 / (r12 * r1))
            f1 = np.cosh(bet) - (1 / bet) * (1 / r1 + 1 / r12) * np.sinh(bet)
            f2 = (1 / (bet * r12)) * np.sinh(bet)
            f3 = np.cosh(bet) + (1 / bet) * (1 / r1 + 1 / r12) * np.sinh(bet)
            return 2.0 * (f3 - f1 - 2 * f2) / (f3 - f2)  # W型管是U型管的两倍


        # W型管修正因子
        def correction_factor_D(D, a=0.2, b=5):
            """针对两对U型管中心距离的修正函数"""
            return 1 - a * np.exp(-b * D)


        s_raw = calc_s_raw_w(Dp, rp, ri, hr, kp, kb, ks, d, M, cw, H)
        s_raw = correction_factor_D(du) * s_raw


        # 深度修正函数（W）
        def s_corrected(s_raw, H, H_limit=80, s_min=0.4):
            """
            修正 s_raw，使其在 H <= 80 时随着 H 增加而指数增长。
            H > H_limit 时不修正，直接返回 s_raw。
            """
            if H >= H_limit:
                return s_raw

            # 拟合指数关系：s(H) = s_min * exp(k * H)
            s_line = s_raw  # 使用原始值作为上限
            k = np.log(s_line / s_min) / H_limit
            s = s_min * np.exp(k * H)

            return np.clip(s, s_min, s_line)
        
        s = s_corrected(s_raw, H)

    # ---- 体积流量 ----
    qv = math.pi * ri ** 2 * v * 3600  # 单根管道体积流量 m³/h

    # ---- 雷诺数 ----
    rho = 1000  # 水的密度 kg/m³
    mu = 0.001  # 水的动力粘度 Pa·s
    Re = (rho * v * (2 * ri)) / mu  # 雷诺数计算公式

    # 存储结果
    st.session_state['params']['s'] = s

    st.success(f"能效比 s 计算值: {s:.4f} (管式类型: {pipe_type})")
    st.success(f"体积流量 qv: {qv:.2f} 立方米/小时")
    st.success(f"雷诺数 Re: {Re:.2f}")

    zz = params['zz']
    st.success("分层深度 zz: " + ", ".join([f"{x:.2f}" for x in zz]))
    flag = np.any(np.isclose(zz, params['H'] / 2))
    st.success(f"确保包含 H/2: {'是' if flag else '否'}")
    page_nav()

elif st.session_state['page'] == 4:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    st.header("4. 动态边界模型及相关函数")
    st.markdown("### 📌 本页内容概览")

    with st.expander("🌡️ 动态边界模型"):
        st.markdown(
            "- 根据地温二谐波模型，定义动态边界模型 `dynamic_model`"
        )

    with st.expander("🔹 初始温度分布计算"):
        st.markdown(
            "- 根据地表温度的二谐波传播模型和一维热传导理论，计算每一深度点在时间范围内的初始温度分布 (`T_initial`)，即在未受热负荷扰动前的温场基线。"
        )

    with st.expander("⚡ 地表动态温度响应"):
        st.markdown(
            "- 求解地表温度边界条件的时间变化（年周期）对地下温度场的瞬时动态响应 (`T_gs`)，"
            "属于边界扰动传热响应问题的傅里叶积分形式，通过卷积表达地表动态温度源在深度 z 处的响应。"
        )

    with st.expander("📊 结果可视化"):
        st.markdown(
            "- 绘制 `T_initial`、`T_gs` 全年变化曲线，方便直观观察温场动态响应。"
        )

    st.markdown(
    '<span style="color:red; font-weight:bold;">⚠️ 请等待计算完成并绘制出图像后再点击"下一页"按钮</span>', 
    unsafe_allow_html=True
)
    params = st.session_state['params']
    # 取参数
    tt = params['tt']
    ttmax = params['ttmax']
    H = params['H']
    Tsavg = params['Tsavg']
    Tamp1 = params['Tamp1']
    Tamp2 = params['Tamp2']
    PL1 = params['PL1']
    PL2 = params['PL2']
    as_ = params['ks'] / params['Cv']
    tp = params['tp']
    tt0 = params['tt0']
    zz = params['zz']
    # 动态边界模型
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
                exponent1 = -(Df_vals - z) ** 2 / (4 * as_ * t)
                exponent2 = -(Df_vals + z) ** 2 / (4 * as_ * t)
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


    T_initial = np.array([T_Initial(tt, z_i) for z_i in zz])
    T_gs = np.array([T_Gs(tt, z_i) for z_i in zz])

    # 只计算一次并缓存，增加进度条
    if 'T_initial' not in st.session_state:
        def calc_T_initial():
            progress = st.progress(0, text="T_initial 计算中...")
            arr = np.zeros((len(zz), len(tt)))
            for z_idx, z_val in enumerate(zz):
                for i, t in enumerate(tt):
                    if t == 0:
                        arr[z_idx, i] = dynamic_model(0, z_val)
                    else:
                        Df_max = 20
                        Df_vals = np.linspace(0, Df_max, 100)
                        temp_vals = np.array([dynamic_model(0, Df) for Df in Df_vals])
                        exponent1 = -(Df_vals - z_val) ** 2 / (4 * as_ * t)
                        exponent2 = -(Df_vals + z_val) ** 2 / (4 * as_ * t)
                        integrand = temp_vals * (np.exp(exponent1) - np.exp(exponent2)) / (2 * np.sqrt(as_ * np.pi) * np.sqrt(t))
                        arr[z_idx, i] = np.trapezoid(integrand, Df_vals)
                progress.progress((z_idx+1)/len(zz), text=f"T_initial 计算进度: {z_idx+1}/{len(zz)}")
            progress.progress(1.0, text="T_initial 计算完成！")
            return np.round(arr, 2)
        st.session_state['T_initial'] = calc_T_initial()
    if 'T_gs' not in st.session_state:
        def calc_T_gs():
            progress = st.progress(0, text="T_gs 计算中...")
            arr = np.zeros((len(zz), len(tt)))
            for z_idx, z_val in enumerate(zz):
                for i, t in enumerate(tt):
                    if t == 0:
                        arr[z_idx, i] = 0
                        continue
                    Dt_vals = np.linspace(0, t, 100)
                    gs_vals = np.zeros(len(Dt_vals))
                    for j, Dt in enumerate(Dt_vals):
                        if t == Dt:
                            gs_vals[j] = 0
                        else:
                            term = dynamic_model(Dt, 0)
                            exponent3 = -z_val ** 2 / (4 * as_ * (t - Dt))
                            gs_vals[j] = term * np.exp(exponent3) * z_val / (2 * np.sqrt(as_ * np.pi)) / (t - Dt) ** 1.5
                    arr[z_idx, i] = np.trapezoid(gs_vals, Dt_vals)
                progress.progress((z_idx+1)/len(zz), text=f"T_gs 计算进度: {z_idx+1}/{len(zz)}")
            progress.progress(1.0, text="T_gs 计算完成！")
            return np.round(arr, 2)
        st.session_state['T_gs'] = calc_T_gs()

    # 只画 H/2 的那一条（与缓存一致）
    T_initial = st.session_state['T_initial']
    T_gs = st.session_state['T_gs']
    zz = st.session_state['params']['zz']
    H = st.session_state['params']['H']
    ttmax = st.session_state['params']['ttmax']
    idx_mid = np.argmin(np.abs(zz - H/2))
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(ttmax) / 24  # 天数
    ax.plot(x, T_initial[idx_mid], label=f'T_initial (H/2={zz[idx_mid]:.2f}m)', color='orange')
    ax.plot(x, T_gs[idx_mid], label=f'T_gs (H/2={zz[idx_mid]:.2f}m)', color='green')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('T_initial & T_gs annual Variation (H/2)')
    day_ticks = list(range(0, 366, 50))
    ax.set_xticks(day_ticks)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    page_nav()

elif st.session_state['page'] == 5:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    st.header("5. 格林函数核响应函数预计算（并行优化版）")
    st.markdown("""
**本页内容：**
- 使用并行计算计算格林函数 圆形热源核响应和线形热源核响应
- 整合全部深度点与时间步的核函数响应，用于构造热响应矩阵。
- 对所有响应点进行核函数积分预计算
    """)
    st.markdown('<span style="color:red; font-weight:bold;">- 请等待所有核响应函数计算完成后再点击"下一页"按钮</span>', unsafe_allow_html=True)

    # 获取参数
    params = st.session_state['params']
    pipe_type = params.get('pipe_type', '螺旋管')
    xx = params['xx']
    yy = params['yy']
    zz = params['zz']
    tt = params['tt']
    ttmax = params['ttmax']
    as_ = params['ks'] / params['Cv']
    H = params['H']
    Np = params['Np']

    # 检查是否已经计算过
    if 'circle_results' in st.session_state and 'line_results' in st.session_state and pipe_type == "螺旋管":
        circle_results = st.session_state['circle_results']
        line_results = st.session_state['line_results']
        st.info("检测到已缓存的螺旋管核函数积分结果，跳过计算。")
    elif 'line1_results' in st.session_state and 'line2_results' in st.session_state and pipe_type == "U型管":
        line1_results = st.session_state['line1_results']
        line2_results = st.session_state['line2_results']
        st.info("检测到已缓存的U型管核函数积分结果，跳过计算。")
    elif 'line1_results' in st.session_state and 'line2_results' in st.session_state and 'line3_results' in st.session_state and 'line4_results' in st.session_state and pipe_type == "W型管":
        line1_results = st.session_state['line1_results']
        line2_results = st.session_state['line2_results']
        line3_results = st.session_state['line3_results']
        line4_results = st.session_state['line4_results']
        st.info("检测到已缓存的W型管核函数积分结果，跳过计算。")
    else:
        from joblib import Parallel, delayed
        import multiprocessing


        # 螺旋管核函数计算
        @jit(nopython=True)
        def circle(w, t, xx, yy, zz, Nc, D, as_, z):
            term = 0.0
            for i in range(Nc):
                dx = D / 2 * np.cos(w) - xx
                dy = D / 2 * np.sin(w) - yy
                dz_minus = z[i] - zz
                dz_plus = z[i] + zz
                r_minus = np.sqrt(dx ** 2 + dy ** 2 + dz_minus ** 2)
                r_plus = np.sqrt(dx ** 2 + dy ** 2 + dz_plus ** 2)
                if r_minus < 1e-10 or r_plus < 1e-10 or t == 0.0:
                    continue
                term += 1 / r_minus * erfc(r_minus / (2 * np.sqrt(as_ * t))) - 1 / r_plus * erfc(
                    r_plus / (2 * np.sqrt(as_ * t)))
            return term


        @jit(nopython=True)
        def line(h, t, xx, yy, zz, D, as_):
            dx = xx
            dy = yy
            dz_minus = h - zz
            dz_plus = h + zz
            r_minus1 = np.sqrt(dx ** 2 + dy ** 2 + dz_minus ** 2)
            r_plus1 = np.sqrt(dx ** 2 + dy ** 2 + dz_plus ** 2)
            if r_minus1 < 1e-10 or r_plus1 < 1e-10 or t == 0.0:
                return 0.0
            return 1 / r_minus1 * erfc(r_minus1 / (2 * np.sqrt(as_ * t))) - 1 / r_plus1 * erfc(
                r_plus1 / (2 * np.sqrt(as_ * t)))


        def compute_circle_integrals(t, xx, yy, zz, Nc, D, as_, z):
            w_vals = np.linspace(0, 2 * np.pi, 100)
            vals = np.array([circle(w, t, xx, yy, zz, Nc, D, as_, z) for w in w_vals])
            return np.trapz(vals, w_vals)


        def compute_line_integrals(t, xx, yy, zz, D, as_, z_range):
            h_vals = np.linspace(z_range[0], z_range[1], 100)
            vals = np.array([line(h, t, xx, yy, zz, D, as_) for h in h_vals])
            return np.trapz(vals, h_vals)


        # U型管和W型管核函数计算
        @jit(nopython=True)
        def line1(h, t, xx, yy, zz, as_, d):
            dx = -(d / 2) - xx
            dy = yy
            dz_minus = h - zz
            dz_plus = h + zz
            r_minus1 = np.sqrt(dx ** 2 + dy ** 2 + dz_minus ** 2)
            r_plus1 = np.sqrt(dx ** 2 + dy ** 2 + dz_plus ** 2)

            if r_minus1 < 1e-10 or r_plus1 < 1e-10 or t == 0.0:
                return 0.0

            return 1 / r_minus1 * erfc(r_minus1 / (2 * np.sqrt(as_ * t))) \
                - 1 / r_plus1 * erfc(r_plus1 / (2 * np.sqrt(as_ * t)))


        def line2(h, t, xx, yy, zz, as_, d):
            dx = (d / 2) - xx
            dy = yy
            dz_minus = h - zz
            dz_plus = h + zz
            r_minus1 = np.sqrt(dx ** 2 + dy ** 2 + dz_minus ** 2)
            r_plus1 = np.sqrt(dx ** 2 + dy ** 2 + dz_plus ** 2)

            if r_minus1 < 1e-10 or r_plus1 < 1e-10 or t == 0.0:
                return 0.0

            return 1 / r_minus1 * erfc(r_minus1 / (2 * np.sqrt(as_ * t))) \
                - 1 / r_plus1 * erfc(r_plus1 / (2 * np.sqrt(as_ * t)))


        def compute_line1_integrals(t, xx, yy, zz, as_, H, d):
            h_vals = np.linspace(0, H, 100)
            vals = np.array([line1(h, t, xx, yy, zz, as_, d) for h in h_vals])
            return np.trapz(vals, h_vals)


        def compute_line2_integrals(t, xx, yy, zz, as_, H, d):
            h_vals = np.linspace(0, H, 100)
            vals = np.array([line2(h, t, xx, yy, zz, as_, d) for h in h_vals])
            return np.trapz(vals, h_vals)


        def compute_kernels_all_points(xx_list, yy_list, zz_list, tt, H, Np, as_, pipe_type, **kwargs):
            """根据管式类型计算核函数积分"""
            n_z = len(zz_list)
            n_points = len(xx_list)
            n_times = len(tt)

            if pipe_type == "螺旋管":
                # 螺旋管核函数计算
                D = kwargs.get('D')
                Nc = kwargs.get('Nc')
                z = kwargs.get('z')

                circle_results = np.zeros((n_z, n_points, n_times))
                line_results = np.zeros((n_z, n_points, n_times))

                st.info("并行预计算螺旋管 circle / line 积分核...")
                progress = st.progress(0, text="核函数积分进度...")
                total = n_z * n_points
                count = 0
                for z_idx, zz0 in enumerate(zz_list):
                    st.info(f'  -> z点 {z_idx + 1}/{n_z}')
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
                        count += 1
                        progress.progress(count / total, text=f"核函数积分进度: {count}/{total}")
                progress.progress(1.0, text="核函数积分全部完成！")
                return circle_results, line_results

            elif pipe_type == "U型管":
                # U型管核函数计算
                d = kwargs.get('d')

                line1_results = np.zeros((n_z, n_points, n_times))
                line2_results = np.zeros((n_z, n_points, n_times))

                st.info("并行预计算U型管 line1 / line2 积分核...")
                progress = st.progress(0, text="核函数积分进度...")
                total = n_z * n_points
                count = 0
                for z_idx, zz0 in enumerate(zz_list):
                    st.info(f'  -> z点 {z_idx + 1}/{n_z}')
                    for resp_idx in range(n_points):
                        xx, yy = xx_list[resp_idx], yy_list[resp_idx]
                        line1_result = Parallel(n_jobs=-1)(
                            delayed(compute_line1_integrals)(t, xx, yy, zz0, as_, H, d) for t in tt
                        )
                        line2_result = Parallel(n_jobs=-1)(
                            delayed(compute_line2_integrals)(t, xx, yy, zz0, as_, H, d) for t in tt
                        )
                        line1_results[z_idx, resp_idx, :] = line1_result
                        line2_results[z_idx, resp_idx, :] = line2_result
                        count += 1
                        progress.progress(count / total, text=f"核函数积分进度: {count}/{total}")
                progress.progress(1.0, text="核函数积分全部完成！")
                return line1_results, line2_results

            elif pipe_type == "W型管":
                # W型管核函数计算
                d = kwargs.get('d')
                du = kwargs.get('du')

                line1_results = np.zeros((n_z, n_points, n_times))
                line2_results = np.zeros((n_z, n_points, n_times))
                line3_results = np.zeros((n_z, n_points, n_times))
                line4_results = np.zeros((n_z, n_points, n_times))

                st.info("并行预计算W型管 line1 / line2 / line3 / line4 积分核...")
                progress = st.progress(0, text="核函数积分进度...")
                total = n_z * n_points
                count = 0
                for z_idx, zz0 in enumerate(zz_list):
                    st.info(f'  -> z点 {z_idx + 1}/{n_z}')
                    for resp_idx in range(n_points):
                        xx, yy = xx_list[resp_idx], yy_list[resp_idx]
                        line1_result = Parallel(n_jobs=-1)(
                            delayed(compute_line1_integrals)(t, xx, yy, zz0, as_, H, d) for t in tt
                        )
                        line2_result = Parallel(n_jobs=-1)(
                            delayed(compute_line2_integrals)(t, xx, yy, zz0, as_, H, d) for t in tt
                        )
                        # W型管使用相同的核函数计算，但在实际计算中会使用不同的组合
                        line3_result = line1_result  # 简化处理
                        line4_result = line2_result  # 简化处理

                        line1_results[z_idx, resp_idx, :] = line1_result
                        line2_results[z_idx, resp_idx, :] = line2_result
                        line3_results[z_idx, resp_idx, :] = line3_result
                        line4_results[z_idx, resp_idx, :] = line4_result
                        count += 1
                        progress.progress(count / total, text=f"核函数积分进度: {count}/{total}")
                progress.progress(1.0, text="核函数积分全部完成！")
                return line1_results, line2_results, line3_results, line4_results


        # 根据管式类型调用不同的计算函数
        if pipe_type == "螺旋管":
            try:
                D = params['D']
                Nc = params['Nc']
                z = params['z']
                circle_results, line_results = compute_kernels_all_points(
                    xx, yy, zz, tt, H, Np, as_, pipe_type, D=D, Nc=Nc, z=z
                )
                st.session_state['circle_results'] = circle_results
                st.session_state['line_results'] = line_results
                st.success('螺旋管核函数积分预计算完成！')
            except KeyError as e:
                st.error(f"缺少螺旋管必要参数: {e}")
                page_nav()
                st.stop()

        elif pipe_type == "U型管":
            try:
                d = params['d']
                line1_results, line2_results = compute_kernels_all_points(
                    xx, yy, zz, tt, H, Np, as_, pipe_type, d=d
                )
                st.session_state['line1_results'] = line1_results
                st.session_state['line2_results'] = line2_results
                st.success('U型管核函数积分预计算完成！')
            except KeyError as e:
                st.error(f"缺少U型管必要参数: {e}")
                page_nav()
                st.stop()

        elif pipe_type == "W型管":
            try:
                d = params['d']
                du = params['du']
                line1_results, line2_results, line3_results, line4_results = compute_kernels_all_points(
                    xx, yy, zz, tt, H, Np, as_, pipe_type, d=d, du=du
                )
                st.session_state['line1_results'] = line1_results
                st.session_state['line2_results'] = line2_results
                st.session_state['line3_results'] = line3_results
                st.session_state['line4_results'] = line4_results
                st.success('W型管核函数积分预计算完成！')
            except KeyError as e:
                st.error(f"缺少W型管必要参数: {e}")
                page_nav()
                st.stop()

    page_nav()
    
elif st.session_state['page'] == 6:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    st.header("6. 热泵模型加载与选择")
    st.markdown("### 📌 本页内容概览")

    with st.expander("🛠️ 自定义热泵模型"):
        st.markdown(
            "- 自定义热泵模型，根据需求调整热泵参数。"
        )

    with st.expander("📈 XGBoost训练模型"):
        st.markdown(
            "- 根据部分负荷率 `PRL`、热泵进口水温 `T_HP,in` 和 `COP` 使用 XGBoost 进行训练。"
        )

    with st.expander("📂 数据上传"):
        st.markdown(
            "- 如有数据，请上传制热和制冷工况下的两个数据文件（csv 或 xlsx），"
            "三列表头分别为 `PRL`, `T_HP,in`, `COP`。"
        )

    with st.expander("⚙️ 默认模型"):
        st.markdown(
            "- 若无数据，可直接使用系统默认热泵模型（由 XGBoost 根据已有实验数据训练得到）。"
        )

    st.markdown('<span style="color:red; font-weight:bold;">- 已有模型精度如下图所示</span>', unsafe_allow_html=True)
    st.image("HP-cop.jpg", caption="预加载热泵模型COP预测效果展示")
    import xgboost as xgb
    import os
    user_has_custom = st.radio("是否有厂家自定义热泵参数文件？", ("有", "无"), index=1)
    if user_has_custom == "有":
        st.info("请上传制热和制冷工况下的两个数据文件（csv或xlsx），三列表头分别为PRL, T_HP,in, COP")
        file_heat = st.file_uploader("上传制热工况热泵参数文件", type=["csv", "xlsx"], key="heat_file")
        file_cool = st.file_uploader("上传制冷工况热泵参数文件", type=["csv", "xlsx"], key="cool_file")
        ready = False
        if file_heat is not None and file_cool is not None:
            # 读取数据
            def read_data(f):
                if f.name.endswith(".csv"):
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)
                return df
            df_heat = read_data(file_heat)
            df_cool = read_data(file_cool)
            # 检查列名
            required_cols = ["PRL", "T_HP,in", "COP"]
            if not all(col in df_heat.columns for col in required_cols):
                st.error("制热工况文件缺少必要列，请确保有：PRL, T_HP,in, COP")
            elif not all(col in df_cool.columns for col in required_cols):
                st.error("制冷工况文件缺少必要列，请确保有：PRL, T_HP,in, COP")
            else:
                # 训练模型
                X_heat = df_heat[["PRL", "T_HP,in"]].values
                y_heat = df_heat["COP"].values
                X_cool = df_cool[["PRL", "T_HP,in"]].values
                y_cool = df_cool["COP"].values
                model1 = xgb.XGBRegressor()
                model1.fit(X_heat, y_heat)
                model2 = xgb.XGBRegressor()
                model2.fit(X_cool, y_cool)
                # 保存模型
                model1.save_model("COP_heating_custom.json")
                model2.save_model("COP_colding_custom.json")
                st.success("自定义热泵模型已训练并保存！")
                # 加载模型到session_state
                model1_loaded = xgb.XGBRegressor()
                model1_loaded.load_model("COP_heating_custom.json")
                model2_loaded = xgb.XGBRegressor()
                model2_loaded.load_model("COP_colding_custom.json")
                st.session_state['model1'] = model1_loaded
                st.session_state['model2'] = model2_loaded
                # 提供模型下载
                with open("COP_heating_custom.json", "r", encoding="utf-8") as f:
                    cop_heating_content = f.read()
                with open("COP_colding_custom.json", "r", encoding="utf-8") as f:
                    cop_colding_content = f.read()
                st.download_button('下载自定义 COP_heating.json', cop_heating_content, file_name='COP_heating_custom.json')
                st.download_button('下载自定义 COP_colding.json', cop_colding_content, file_name='COP_colding_custom.json')
                ready = True
        if ready:
            st.success("自定义热泵模型已加载，可进行下一步。")
        page_nav()
    else:
        # 使用默认模型
        model1 = xgb.XGBRegressor()
        model2 = xgb.XGBRegressor()
        try:
            model1.load_model("COP_heating.json")
            model2.load_model("COP_colding.json")
            st.info('未上传自定义参数，已加载系统默认热泵COP模型。')
            st.session_state['model1'] = model1
            st.session_state['model2'] = model2
            # 显示默认json内容
            with open("COP_heating.json", "r", encoding="utf-8") as f:
                cop_heating_content = f.read()
            with open("COP_colding.json", "r", encoding="utf-8") as f:
                cop_colding_content = f.read()
            st.markdown("**默认 COP_heating.json 内容（前1000字符）:**")
            st.code(cop_heating_content[:1000], language='json')
            st.download_button('下载默认 COP_heating.json', cop_heating_content, file_name='COP_heating.json')
            st.markdown("**默认 COP_colding.json 内容（前1000字符）:**")
            st.code(cop_colding_content[:1000], language='json')
            st.download_button('下载默认 COP_colding.json', cop_colding_content, file_name='COP_colding.json')
        except Exception as e:
            st.warning(f"无法读取默认模型文件: {e}")
        page_nav()

elif st.session_state['page'] == 7:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    st.header("7. 初始化系统计算模型")
    st.markdown("""
**本页内容：**
- 根据前面的设计参数及热泵模型，初始化全年模拟所需的变量。
- 即对计算中所涉及的多个变量进行第一步赋值以便后续计算。
- 如 COP、PLR、热泵进口水温 等。
    """)
    # 变量初始化
    try:
        ttmax = st.session_state['params']['ttmax']
        T_initial = st.session_state['T_initial']
        zz = st.session_state['params']['zz']
        H = st.session_state['params']['H']
    except Exception as e:
        st.error(f"请先完成前面页面的参数设定和初始温度计算！错误: {e}")
        page_nav()
        st.stop()
    # 初始化所有模拟变量并存入 session_state
    st.session_state['Q1'] = np.zeros(ttmax)
    st.session_state['q'] = np.zeros(ttmax)
    st.session_state['T_line'] = np.zeros(ttmax)
    st.session_state['T_circle'] = np.zeros(ttmax)
    st.session_state['all_integral_circle_result'] = np.zeros((ttmax, 6))
    st.session_state['all_integral_line_result'] = np.zeros((ttmax, 6))
    st.session_state['T_point'] = np.zeros(ttmax)
    st.session_state['T_in'] = np.zeros(ttmax)
    st.session_state['T_out'] = np.zeros(ttmax+1)
    st.session_state['COP'] = np.zeros(ttmax)
    st.session_state['PLR'] = np.zeros(ttmax)
    # 力学初始条件
    st.session_state['z_top'] = np.zeros(ttmax)
    st.session_state['sigma_max'] = np.zeros(ttmax)
    # 修正 T_out[0] 的赋值逻辑，兼容 1D/2D
    if T_initial.ndim == 2:
        idx_mid = np.argmin(np.abs(zz - H/2))
        st.session_state['T_out'][0] = T_initial[idx_mid][0]
    else:
        st.session_state['T_out'][0] = T_initial[0]
    st.success("系统变量已初始化，可进行全年模拟。")
    page_nav()

elif st.session_state['page'] == 8:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    st.header("8. 全年模拟")
    st.markdown("""
**本页内容：**
- 根据前面的设计参数及热泵模型，计算全年模拟;
- 模拟后可获得系统各时间步热-力计算结果，如：
- 桩群出口水温、不同深度桩壁温度、桩顶位移、最大热应力
    """)
    st.markdown('<span style="color:red; font-weight:bold;">- 系统热力计算输出结果验证说明：</span>', unsafe_allow_html=True)
    st.image("Tout-verification.jpg", caption="计算搭载换热解析模型桩群出口水温与TRNSYS输出对比")
    st.markdown("""
**换热解析模型支持下的力学性能验证：**
- 参考：
- 孔纲强,王成龙,刘汉龙,等.多次温度循环对能量桩桩顶位移影响分析[J].岩土力学,2017,38(04):958-964.DOI:10.16285/j.rsm.2017.04.005.
- 使用C30混凝土的线膨胀系数及弹性模量，根据不同深度的桩壁温度计算桩顶位移、最大热应力，所得结果属于满足限值条件下的同一数量级范围。
    """)
    st.markdown('<span style="color:red; font-weight:bold;">- 请等待右边"下一页"按钮由灰色变为红色后再点击"下一页"按钮</span>', unsafe_allow_html=True)
    
    # 检查是否已有缓存结果
    if 'annual_simulation_done' in st.session_state and st.session_state['annual_simulation_done']:
        st.info("检测到已缓存的全年模拟结果，跳过计算。如果需要重新计算请重新运行")
    else:
        try:
            params = st.session_state['params']
            pipe_type = params.get('pipe_type', '螺旋管')  # 获取管式类型
            Q = params['Q']
            ttmax = params['ttmax']
            ks = params['ks']
            Np = params['Np']
            M = params['M']
            cw = params['cw']
            s = params['s']
            L_total = params['L_total']
            T_initial = st.session_state['T_initial']
            T_gs = st.session_state['T_gs']
            model1 = st.session_state['model1']
            model2 = st.session_state['model2']
            zz = params['zz']
            xx = params['xx']

            # 结果变量
            n_z = len(zz)
            n_points = len(xx)
            Q1 = np.zeros(ttmax)
            q = np.zeros(ttmax)

            # 根据管式类型初始化不同的温度变量
            if pipe_type == "螺旋管":
                circle_results = st.session_state['circle_results']
                line_results = st.session_state['line_results']
                T_circle = np.zeros((n_z, ttmax))
                T_line = np.zeros((n_z, ttmax))
            elif pipe_type == "U型管":
                line1_results = st.session_state['line1_results']
                line2_results = st.session_state['line2_results']
                T_line1 = np.zeros((n_z, ttmax))
                T_line2 = np.zeros((n_z, ttmax))
            elif pipe_type == "W型管":
                line1_results = st.session_state['line1_results']
                line2_results = st.session_state['line2_results']
                line3_results = st.session_state['line3_results']
                line4_results = st.session_state['line4_results']
                T_line1 = np.zeros((n_z, ttmax))
                T_line2 = np.zeros((n_z, ttmax))
                T_line3 = np.zeros((n_z, ttmax))
                T_line4 = np.zeros((n_z, ttmax))

            T_point = np.zeros((n_z, ttmax))
            all_integral_circle_result = np.zeros((n_z, n_points)) if pipe_type == "螺旋管" else None
            all_integral_line_result = np.zeros((n_z, n_points)) if pipe_type == "螺旋管" else None
            all_integral_line1_result = np.zeros((n_z, n_points)) if pipe_type in ["U型管", "W型管"] else None
            all_integral_line2_result = np.zeros((n_z, n_points)) if pipe_type in ["U型管", "W型管"] else None
            all_integral_line3_result = np.zeros((n_z, n_points)) if pipe_type == "W型管" else None
            all_integral_line4_result = np.zeros((n_z, n_points)) if pipe_type == "W型管" else None

            T_in = np.zeros(ttmax)
            T_out = np.zeros(ttmax + 1)
            COP = np.zeros(ttmax)
            PLR = np.zeros(ttmax)

            # 力学初始条件
            z_top = np.zeros(ttmax)
            sigma_max = np.zeros(ttmax)

            # 只用H/2那一层的索引
            idx_mid = np.argmin(np.abs(zz - params['H'] / 2))
            T_out[0] = T_initial[0] if T_initial.ndim == 1 else T_initial[idx_mid][0]
        except Exception as e:
            st.error(f"请先完成前面页面的所有步骤！错误: {e}")
            page_nav()
            st.stop()

        progress = st.progress(0, text="全年模拟计算中...")
        for i in range(ttmax):
            # ---------- 热力学与温度场计算 ----------
            if Q[i] < 0:
                PLR[i] = Q[i] / min(Q)
                X_input1 = np.array([[T_out[i], PLR[i]]])
                COP[i] = model1.predict(X_input1)[0]
                Q1[i] = Q[i] * (1 - 1 / COP[i]) if COP[i] > 0 else 0
            elif Q[i] > 0:
                PLR[i] = Q[i] / max(Q)
                X_input2 = np.array([[T_out[i], PLR[i]]])
                COP[i] = model2.predict(X_input2)[0]
                Q1[i] = Q[i] * (1 + 1 / COP[i]) if COP[i] > 0 else 0
            else:
                COP[i] = 0
                Q1[i] = 0

            q[i] = Q1[i] * 1000 / L_total
            dq = np.diff(q[:i + 1], prepend=0)

            for z_idx in range(n_z):
                for a in range(n_points):
                    if pipe_type == "螺旋管":
                        all_integral_circle_result[z_idx, a] = np.dot(dq[::-1], circle_results[z_idx, a, :i + 1])
                        all_integral_line_result[z_idx, a] = np.dot(dq[::-1], line_results[z_idx, a, :i + 1])

                        # 螺旋管计算
                        T_circle[z_idx, i] = (params['D'] / 2) / (4 * np.pi * ks) * (
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

                    elif pipe_type == "U型管":
                        all_integral_line1_result[z_idx, a] = np.dot(dq[::-1], line1_results[z_idx, a, :i + 1])
                        all_integral_line2_result[z_idx, a] = np.dot(dq[::-1], line2_results[z_idx, a, :i + 1])

                        # U型管计算
                        T_line1[z_idx, i] = 1 / (4 * np.pi * ks) * (
                                all_integral_line1_result[z_idx, 0] + all_integral_line1_result[z_idx, 1] +
                                all_integral_line1_result[z_idx, 2] + 2 * all_integral_line1_result[z_idx, 3] +
                                2 * all_integral_line1_result[z_idx, 4] + all_integral_line1_result[z_idx, 5]
                        )
                        T_line2[z_idx, i] = 1 / (4 * np.pi * ks) * (
                                all_integral_line2_result[z_idx, 0] + all_integral_line2_result[z_idx, 1] +
                                all_integral_line2_result[z_idx, 2] + 2 * all_integral_line2_result[z_idx, 3] +
                                2 * all_integral_line2_result[z_idx, 4] + all_integral_line2_result[z_idx, 5]
                        )
                        T_point[z_idx, i] = T_line1[z_idx, i] + T_line2[z_idx, i] + T_initial[z_idx, i] + T_gs[z_idx, i]

                    elif pipe_type == "W型管":
                        all_integral_line1_result[z_idx, a] = np.dot(dq[::-1], line1_results[z_idx, a, :i + 1])
                        all_integral_line2_result[z_idx, a] = np.dot(dq[::-1], line2_results[z_idx, a, :i + 1])
                        all_integral_line3_result[z_idx, a] = np.dot(dq[::-1], line3_results[z_idx, a, :i + 1])
                        all_integral_line4_result[z_idx, a] = np.dot(dq[::-1], line4_results[z_idx, a, :i + 1])

                        # W型管计算
                        T_line1[z_idx, i] = 1 / (4 * np.pi * ks) * (
                                all_integral_line1_result[z_idx, 0] + all_integral_line1_result[z_idx, 1] +
                                all_integral_line1_result[z_idx, 2] + 2 * all_integral_line1_result[z_idx, 3] +
                                2 * all_integral_line1_result[z_idx, 4] + all_integral_line1_result[z_idx, 5]
                        )
                        T_line2[z_idx, i] = 1 / (4 * np.pi * ks) * (
                                all_integral_line2_result[z_idx, 0] + all_integral_line2_result[z_idx, 1] +
                                all_integral_line2_result[z_idx, 2] + 2 * all_integral_line2_result[z_idx, 3] +
                                2 * all_integral_line2_result[z_idx, 4] + all_integral_line2_result[z_idx, 5]
                        )
                        T_line3[z_idx, i] = 1 / (4 * np.pi * ks) * (
                                all_integral_line3_result[z_idx, 0] + all_integral_line3_result[z_idx, 1] +
                                all_integral_line3_result[z_idx, 2] + 2 * all_integral_line3_result[z_idx, 3] +
                                2 * all_integral_line3_result[z_idx, 4] + all_integral_line3_result[z_idx, 5]
                        )
                        T_line4[z_idx, i] = 1 / (4 * np.pi * ks) * (
                                all_integral_line4_result[z_idx, 0] + all_integral_line4_result[z_idx, 1] +
                                all_integral_line4_result[z_idx, 2] + 2 * all_integral_line4_result[z_idx, 3] +
                                2 * all_integral_line4_result[z_idx, 4] + all_integral_line4_result[z_idx, 5]
                        )
                        T_point[z_idx, i] = (T_line1[z_idx, i] + T_line2[z_idx, i] +
                                             T_line3[z_idx, i] + T_line4[z_idx, i] +
                                             T_initial[z_idx, i] + T_gs[z_idx, i])

            # 只用H/2深度的T_point更新T_in/T_out
            T_in[i] = T_out[i] + (Q1[i] * 1000 / Np) / (M * cw)
            T_out[i + 1] = T_point[idx_mid, i] + (Q1[i] * 1000 / Np) / (s * M * cw) - (Q1[i] * 1000 / Np) / (M * cw)

            # ---------- 能量桩力学计算 ----------
            alpha = params.get('alpha', 1e-5)  # 热膨胀系数
            Ep = params.get('Ep', 3e7)  # 弹性模量
            H_val = params['H']
            delta_T = np.ptp(T_point[:, i])  # 当前时刻全部深度点之间的最大温差
            epsilon_free = alpha * delta_T  # 自由热应变
            z_top[i] = epsilon_free * 1000 * H_val / 2  # 桩顶位移（沉降） 单位为mm
            sigma_max[i] = Ep * (epsilon_free - z_top[i] * 0.001 / H_val)  # 最大应力 单位为 kPa

            if i % max(1, ttmax // 100) == 0:
                progress.progress(i / ttmax, text=f"全年模拟进度: {i}/{ttmax}")
        progress.progress(1.0, text="全年模拟计算完成！")

        T_out = T_out[:-1]
        st.session_state['T_out'] = T_out

        # 结果保存，结构与ipynb一致
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
        # 多层温度输出
        for idx, z_val in enumerate(zz):
            data[f'桩壁温度_{z_val:.2f}m'] = T_point[idx]

        df_save = pd.DataFrame(data)
        # 若有原始负荷，补充Q
        if 'Q' in params and params['Q'] is not None:
            df_save['Q'] = params['Q']

            # 确保文件夹存在并保存文件
            import os

            os.makedirs("results", exist_ok=True)
            save_path = "results/data_result.xlsx"
            df_save.to_excel(save_path, index=False)
            st.session_state['annual_simulation_done'] = True
            st.success(f"全年模拟完成，结果已保存至本系统内部'{save_path}'，以便后续调用。")
    page_nav()

elif st.session_state['page'] == 9:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    st.header("9. 全年模拟结果可视化")
    st.markdown("""
**本页内容：**
- 根据全年模拟结果，将结果可视化。
- 下载全部计算数据。
    """)
    st.markdown('<span style="color:red; font-weight:bold;">- 请等待绘制完成后再点击"下一页"按钮，进入经济性计算页面</span>', unsafe_allow_html=True)
    # 读取结果文件
    try:
        df_plot = pd.read_excel('results/data_result.xlsx')
    except Exception as e:
        st.error(f"无法读取结果文件: {e}")
        page_nav()
        st.stop()

    # 统一时间轴
    if 'hours' in df_plot.columns:
        hours = df_plot['hours'].values
    else:
        hours = np.arange(len(df_plot))

    # ---------- 第一部分：不同深度桩壁温度 ----------
    temp_cols = [col for col in df_plot.columns if col.startswith('桩壁温度_')]
    col1, col2 = st.columns(2)  # 创建两列布局

    with col1:
        if temp_cols:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            for col in temp_cols:
                ax1.plot(hours / 24, df_plot[col], label=col)
            ax1.set_xlabel("Time (days)")
            ax1.set_ylabel("Temperature (°C)")
            ax1.set_title("不同深度桩壁温度全年变化")
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)
        else:
            st.info("结果文件中未找到不同深度桩壁温度数据")

    # ---------- 第二部分：进出口水温 ----------
    with col2:
        if 'T_in' in df_plot.columns or 'T_out' in df_plot.columns:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            if 'T_in' in df_plot.columns:
                ax2.plot(hours / 24, df_plot['T_in'], label='T_in (进水温度)')
            if 'T_out' in df_plot.columns:
                ax2.plot(hours / 24, df_plot['T_out'], label='T_out (出水温度)')
            ax2.set_xlabel("Time (days)")
            ax2.set_ylabel("Temperature (°C)")
            ax2.set_title("进出口水温全年变化")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
        else:
            st.info("结果文件中未找到进出口水温数据")

    # 创建第二行两列
    col3, col4 = st.columns(2)

    # ---------- 第三部分：桩顶位移（mm） ----------
    with col3:
        if 'z_top' in df_plot.columns:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.plot(hours / 24, df_plot['z_top'], label='桩顶位移')
            ax3.set_xlabel("Time (days)")
            ax3.set_ylabel("z_top (mm)")
            ax3.set_ylim(-1, 1)
            ax3.set_title("桩顶位移全年变化 (mm)")
            ax3.grid(True)
            ax3.legend()
            st.pyplot(fig3)
        else:
            st.info("结果文件中未找到桩顶位移数据")

    # ---------- 第四部分：最大应力 ----------
    with col4:
        if 'sigma_max' in df_plot.columns:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.plot(hours / 24, df_plot['sigma_max'], label='最大应力')
            ax4.axhline(y=2000, color='red', linestyle='--', label='允许最大应力 (2000 kPa)')
            ax4.set_xlabel("Time (days)")
            ax4.set_ylabel("σ_max (kPa)")
            ax4.set_title("最大应力全年变化 (kPa)")
            ax4.grid(True)
            ax4.legend()
            st.pyplot(fig4)
        else:
            st.info("结果文件中未找到最大应力数据")

    # 提供结果文件下载
    with io.BytesIO() as towrite:
        df_plot.to_excel(towrite, index=False)
        towrite.seek(0)
        st.download_button(
            label="下载全部模拟结果Excel到本地",
            data=towrite.getvalue(),
            file_name="能量桩数据download from streamlit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    page_nav()


elif st.session_state['page'] == 10:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    st.header("10. LCOE (平准化供能成本) 计算")
    st.markdown("""
**本页内容：**
- 根据模拟结果和经济参数计算能量桩系统的平准化供能（热和冷）成本 (LCOE)。
    """)

    try:
        # 加载模拟结果
        df_result = pd.read_excel('results/data_result.xlsx')

        # 从 session_state 获取参数
        params = st.session_state['params']
        L_total = params['L_total']
        Np = params['Np']
        s = params['s']

        # LCOE 计算参数
        st.subheader("经济参数设定")

        # 预定义管径和造价系数（5个选项）
        r_pile = [0.010, 0.0125, 0.016, 0.02, 0.025]  # 5个管径选项
        k_d = [5.3, 6.5, 8.2, 11.6, 14]  # 对应的价格 元/m

# 预定义管径和造价系数
        # 注意：此列表应与参数设定页的管径选项保持一致
        r_pile_prices = {
            0.010: 5.3,
            0.0125: 6.5,
            0.016: 8.2,
            0.020: 11.6,
            0.025: 14.0,
            0.0315: 35.0  # 假设的价格
        }

        # 从参数中获取已设定的管径
        rp = params.get('rp')
        # 查找对应的价格，如果找不到则返回0或提示
        kd = r_pile_prices.get(rp, 0)

        st.markdown("**管径及造价**")
        st.info(f"系统采用的管径 (rp): {rp} m")
        if kd > 0:
            st.info(f"对应的每米造价: {kd} 元/m")
        else:
            st.warning(f"未找到管径 {rp} m 对应的预设价格，请检查经济参数设置。")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            HP_cost = st.number_input("热泵初投资 (元)", value=120000)
            C_elec = st.number_input("电价 (元/kWh)", value=0.57)
        with col2:
            lifetime = st.number_input("系统寿命 (年)", value=20, min_value=1, step=1)
            gamma = st.number_input("折现率", value=0.05, format="%.2f")
        with col3:
            MPEX_ratio = st.number_input("年维护费用占运营费用比例", value=0.01, format="%.2f")

        # 计算按钮
        if st.button("计算LCOE", key="calculate_lcoe"):
            st.subheader("LCOE计算过程")

            # ---- 基本检查 ----
            if rp not in r_pile:
                st.error(f"管径{rp}不在预定义列表{r_pile}中")
            else:
                # ---- 打印基本信息 ----
                st.write('能量桩群内换热管总长L_total（m）:', round(L_total, 2))
                st.write('能量桩数量Np（个）:', Np)
                st.write('能量桩外半径rp（m）:', rp)
                st.write('每米PE管造价系数k_d（CNY/m）:', round(k_d[r_pile.index(rp)], 2))

                # ---- CAPEX 更新：管柱造价 + 热泵 ----
                CAPEX_pipe = round(kd * L_total, 2)
                CAPEX = round(CAPEX_pipe + HP_cost, 2)
                st.write('总投资CAPEX（CNY，含热泵）:', CAPEX)

                # ---- 数据提取 ----
                Q = df_result['Q'].values
                Q_ep = df_result['Q_ep'].values
                COP = df_result['COP'].values
                s = s

                # ---- 年度耗电量与换热量 ----
                Q_annual_el = round(np.sum(np.abs(np.divide(Q, COP, where=COP != 0))), 2)
                st.write('年度耗电Q_annual_el(kWh):', Q_annual_el)

                # 能量桩年换热量计算
                Q_annual_ep = round(np.sum(np.abs(Q_ep)) * s, 2)
                st.write('能量桩年度换热Q_annual_ep(kWh):', Q_annual_ep)

                # ---- OPEX & MPEX ----
                OPEX = round(Q_annual_el * C_elec, 2)
                st.write('每年运行费用OPEX（CNY）:', OPEX)

                MPEX = round(MPEX_ratio * OPEX, 2)
                st.write('每年维护费用MPEX（CNY）:', MPEX)

                # ---- LCOE 计算 ----
                cost_pv = CAPEX
                energy_pv = 0
                for y in range(1, lifetime + 1):
                    cost_pv += round((OPEX + MPEX) / (1 + gamma) ** y, 2)
                    energy_pv += round(Q_annual_ep / (1 + gamma) ** y, 2)

                if energy_pv != 0:
                    LCOE = round(cost_pv / energy_pv, 2)
                else:
                    LCOE = float('inf')

                st.write(f"管径 {rp} m 的 LCOE: {LCOE:.2f} 元/kWh")

                # 保存结果用于下一页
                st.session_state['lcoe_results'] = {
                    'rp': rp,
                    'kd': kd,
                    'CAPEX_pipe': CAPEX_pipe,
                    'CAPEX': CAPEX,
                    'Q_annual_el': Q_annual_el,
                    'Q_annual_ep': Q_annual_ep,
                    'OPEX': OPEX,
                    'MPEX': MPEX,
                    'LCOE': LCOE,
                    'lifetime': lifetime,
                    'gamma': gamma
                }

                st.success("LCOE计算完成！点击'下一页'查看投资回收期分析。")

    except FileNotFoundError:
        st.error("未找到结果文件 'results/data_result.xlsx'。请先完成前面的全年模拟。")
    except Exception as e:
        st.error(f"计算时发生错误: {e}")

    page_nav()

elif st.session_state['page'] == 11:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    st.header("11. 投资回收期计算")
    st.markdown("""
**本页内容：**
- 根据LCOE计算结果，计算系统的静态和动态投资回收期。
    """)

    # 检查是否有LCOE计算结果
    if 'lcoe_results' not in st.session_state:
        st.error("请先完成LCOE计算（页面10）。")
        page_nav()
        st.stop()

    # 获取LCOE计算结果
    results = st.session_state['lcoe_results']

    st.subheader("投资回收期参数设定")
    col1, col2 = st.columns(2)
    with col1:
        C_alt = st.number_input("替代能源成本 (元/kWh)", value=0.75,
                                help="若不用地源热泵，使用其他能源的成本")
    with col2:
        st.write(f"系统寿命: {results['lifetime']} 年")
        st.write(f"折现率: {results['gamma'] * 100:.1f}%")

    if st.button("计算投资回收期", key="calculate_payback"):
        st.subheader("投资回收期计算过程")

        # ---- 提取参数 ----
        rp = results['rp']
        CAPEX = results['CAPEX']
        Q_annual_ep = results['Q_annual_ep']
        OPEX = results['OPEX']
        MPEX = results['MPEX']
        lifetime = results['lifetime']
        gamma = results['gamma']

        # ---- 投资回收期计算 ----
        st.write("#  投资回收期计算 ")
        st.write(f"替代能源成本 C_alt: {C_alt} 元/kWh")

        # 计算年度现金流
        CF_annual = Q_annual_ep * C_alt - (OPEX + MPEX)  # 元/年
        st.write(f"年度现金流 CF_annual: {CF_annual:,.2f} 元/年")


        # 静态回收期
        def simple_payback(capex, annual_cf):
            if annual_cf <= 0:
                return float('inf')
            return capex / annual_cf


        payback_static = simple_payback(CAPEX, CF_annual)
        st.write(f"静态投资回收期: {payback_static:.2f} 年")


        # 动态回收期
        def discounted_payback(capex, annual_cf, years, discount_rate):
            cum_pv = 0
            for t in range(1, years + 1):
                pv = annual_cf / (1 + discount_rate) ** t
                cum_pv += pv
                if cum_pv >= capex:
                    # 线性插值提高精度
                    prev_pv = cum_pv - pv
                    return t - 1 + (capex - prev_pv) / pv
            return float('inf')


        payback_dynamic = discounted_payback(CAPEX, CF_annual, lifetime, gamma)
        st.write(f"动态投资回收期(折现率{gamma:.0%}): {payback_dynamic:.2f} 年")

        # 结果输出
        st.write(f"CAPEX: {CAPEX:,.0f} CNY")
        st.write(f"年度净现金流: {CF_annual:,.0f} CNY/年")
        st.write(f"静态投资回收期: {payback_static:.2f} 年")
        st.write(f"动态投资回收期(折现率{gamma:.0%}): {payback_dynamic:.2f} 年")
        st.write('CF_annual =', CF_annual)
        st.write('CAPEX      =', CAPEX)
        st.write('静态回收期 =', CAPEX / CF_annual if CF_annual > 0 else 'inf')

        st.success("投资回收期计算完成！")

        # 提供详细结果下载
        result_text = f"""
        投资回收期分析结果
        管径: {rp} m
        总投资 (CAPEX): {CAPEX:,.2f} 元
        年度净现金流: {CF_annual:,.2f} 元/年
        替代能源成本: {C_alt:.2f} 元/kWh
        ------------------------------------------
        静态投资回收期: {payback_static} 年
        动态投资回收期 (折现率 {gamma * 100:.1f}%): {payback_dynamic} 年
        """
        st.download_button('下载投资回收期分析报告', result_text, file_name=f'投资回收期分析_{rp}m.txt')

    page_nav()

    # ====================== 功能二：GBR+GA优化 ======================

# 功能二页面 - 使用独立的页面状态管理
elif st.session_state['page'] == 12:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    # 初始化功能二的页面状态
    if 'f2_page' not in st.session_state:
        st.session_state.f2_page = 0


    # 功能二独立的页面导航函数
    def f2_page_nav():
        nav_container = st.container()
        with nav_container:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.session_state.f2_page > 0:
                    if st.button('上一页', key=f"f2_prev_{st.session_state.f2_page}"):
                        st.session_state.f2_page -= 1
                        st.rerun()
                # 返回主菜单按钮
                if st.button('返回主菜单', key="f2_back_to_main"):
                    st.session_state['page'] = 0
                    st.rerun()

            with col3:
                # 最大页面根据当前功能调整
                if st.session_state.f2_page < 3:
                    if st.button('下一页', key=f"f2_next_{st.session_state.f2_page}"):
                        st.session_state.f2_page += 1
                        st.rerun()
            st.markdown("---")

    # 功能二页面0：模型选择与数据上传
    if st.session_state.f2_page == 0:
        import joblib
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import MinMaxScaler

        st.title("能量桩参数优化 - 模型选择")
        st.markdown("""
        **本页内容：**
        - 选择使用预训练模型或进行自定义数据集训练新模型
        - 提供管径对应关系参考
        - 以螺旋管为例进行设计参数优化，优化空间为管径rp、桩数Np、流速v和螺距Lp
        """)
        st.markdown('<span style="color:red; font-weight:bold;">- 如果前面基于默认文件和参数进行了计算，此处可直接使用预训练模型进行优化</span>', unsafe_allow_html=True)

        # 管径对应关系表
        st.subheader("管径对应关系")
        pipe_data = {
            "管径编号": [0, 1, 2, 3, 4, 5],
            "管外半径(m)": [0.010, 0.0125, 0.016, 0.02, 0.025, 0.0315],
            "管内半径(m)": [0.008, 0.0102, 0.013, 0.0162, 0.0202, 0.0252]
        }
        st.table(pd.DataFrame(pipe_data))

        # 模型选择
        st.subheader("模型选择")
        model_option = st.radio("请选择模型来源:",
                                ("使用预训练模型", "进行自定义数据集采样和训练新模型"),
                                index=0)

        # 预加载模型和数据
        if model_option == "使用预训练模型":
            if st.button("加载预训练模型"):
                if 'surrogate' not in st.session_state:
                    try:
                        saved = joblib.load("surrogate_os.pkl")
                        st.session_state.surrogate = saved['model']
                        st.session_state.scaler_X = saved['scaler_X']
                        st.session_state.scaler_y = saved['scaler_y']
                        st.session_state.model_loaded = True
                        st.session_state.model_source = "预训练模型"
                        st.success("预训练模型已加载！")
                    except:
                        st.session_state.model_loaded = False
                        st.warning("未找到预训练模型，请上传自定义数据")

        # 自定义模型训练
        if model_option == "进行自定义数据集采样和训练新模型":
            st.warning("使用自定义数据集训练新模型需进行数据采样，采样时间较长，请耐心等待。")
            from simulation import simulation

            n_samples = st.number_input("采样数量", min_value=1, value=100)
            N_min = st.number_input("桩数 N 最小值", min_value=0, value=0)
            N_max = st.number_input("桩数 N 最大值", min_value=1, value=500)
            v_min = st.number_input("流速 v 最小值", min_value=0.01, value=0.1)
            v_max = st.number_input("流速 v 最大值", min_value=0.01, value=1.0)
            Lp_min = st.number_input("螺距 Lp 最小值", min_value=0.01, value=0.05)
            Lp_max = st.number_input("螺距 Lp 最大值", min_value=0.01, value=0.25)

            st.markdown("""以螺旋管能量桩为例，优化空间为管径rp、桩数Np、流速v和螺距Lp""")
            sample_btn = st.button("开始采样")

            def run_simulation(x):
                params = {'rp_idx': int(round(x[0])), 'N': int(round(x[1])), 'v': float(x[2]), 'Lp': float(x[3])}
                res = simulation(params)
                g_static = [res['Re'], res['qv']]
                g_dynamic = [res['z_top_max'], res['sigma_max_max'], res['T_out_max'], res['T_out_min'], res['unbalance_rate']]
                obj = res['LCOE']
                return np.array(g_static), np.array(g_dynamic), obj

            x_path = 'results/X_samples.csv'
            y_path = 'results/Y_samples.csv'

            if sample_btn:
                import numpy as np
                import pandas as pd
                import os
                os.makedirs("results", exist_ok=True)
                X_samples, G_static_samples, G_dynamic_samples, obj_samples = [], [], [], []
                progress_bar = st.progress(0, text="采样进度")
                for i in range(int(n_samples)):
                    rp_idx = np.random.randint(0, 6)
                    N = np.random.randint(int(N_min), int(N_max)+1)
                    v = np.random.uniform(float(v_min), float(v_max))
                    Lp = np.random.uniform(float(Lp_min), float(Lp_max))
                    x = [rp_idx, N, v, Lp]
                    g_static, g_dynamic, obj = run_simulation(x)
                    X_samples.append(x)
                    G_static_samples.append(g_static)
                    G_dynamic_samples.append(g_dynamic)
                    obj_samples.append(obj)
                    progress_bar.progress((i+1)/int(n_samples), text=f"采样进度: {i+1}/{int(n_samples)}")
                
                X_samples = np.array(X_samples)
                G_static_samples = np.array(G_static_samples)
                G_dynamic_samples = np.array(G_dynamic_samples)
                obj_samples = np.array(obj_samples).reshape(-1, 1)
                Y_samples = np.hstack([G_static_samples, G_dynamic_samples, obj_samples])
                st.session_state['X_samples'] = X_samples
                st.session_state['Y_samples'] = Y_samples
                # 采样完成后自动保存数据
                pd.DataFrame(st.session_state['X_samples'], columns=['rp_idx', 'N', 'v', 'Lp']).to_csv(x_path, index=False)
                pd.DataFrame(st.session_state['Y_samples'],
                            columns=['Re', 'qv', 'z_top_max', 'sigma_max_max', 'T_out_max', 'T_out_min', 'unbalance_rate', 'LCOE']).to_csv(y_path, index=False)
                st.success("采样数据已保存至系统results文件内，分别为：X_samples.csv 和 Y_samples.csv")
                
            # 采样完成后，显示“进入代理模型构建”按钮
            # 红色提示
            st.markdown('<span style="color:red; font-weight:bold;">请在完成采样后，再点击下方进入代理模型构建</span>', unsafe_allow_html=True)
            if st.button("点击进入代理模型构建"):
                st.session_state['build_model'] = True

            # Step 1: 进入代理模型构建流程
            if st.session_state.get('build_model', False):
                try:
                    X_df = pd.read_csv(x_path)
                    Y_df = pd.read_csv(y_path)

                    if X_df.shape[0] != Y_df.shape[0]:
                        st.error("X和Y样本数量不一致！")
                    else:
                        st.success(f"成功加载数据: X样本 {X_df.shape}, Y样本 {Y_df.shape}")

                        # 数据预览
                        st.subheader("数据预览")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("X样本 (输入参数):")
                            st.dataframe(X_df.head())
                        with col2:
                            st.write("Y样本 (输出结果):")
                            st.dataframe(Y_df.head())

                        # Step 2: 进入参数设置
                        if st.button("代理模型参数设置"):
                            st.session_state['param_setting'] = True

                        if st.session_state.get('param_setting', False):
                            n_estimators = st.number_input("n_estimators", min_value=1, max_value=1000, value=200, key="n_estimators")
                            max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=4, key="max_depth")
                            learning_rate = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.05, format="%f", key="learning_rate")

                            # Step 3: 开始训练
                            if st.button("开始进行训练"):
                                from sklearn.ensemble import GradientBoostingRegressor
                                from sklearn.multioutput import MultiOutputRegressor
                                from sklearn.model_selection import train_test_split
                                from sklearn.preprocessing import StandardScaler

                                with st.spinner("正在训练代理模型，请稍候..."):
                                    mask = ~Y_df.isna().any(axis=1)
                                    Y_df_clean = Y_df[mask].reset_index(drop=True)
                                    X_df_clean = X_df[mask].reset_index(drop=True)
                                    
                                    # 归一化
                                    scaler_X = StandardScaler()
                                    scaler_y = StandardScaler()
                                    X_scaled = scaler_X.fit_transform(X_df_clean)
                                    Y_scaled = scaler_y.fit_transform(Y_df_clean)

                                    # 划分数据集
                                    X_train, X_test, Y_train, Y_test = train_test_split(
                                        X_scaled, Y_scaled, test_size=0.2, random_state=42
                                    )

                                    # 训练模型
                                    model = MultiOutputRegressor(GradientBoostingRegressor(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate
                                    ))
                                    model.fit(X_train, Y_train)
                                    score = model.score(X_test, Y_test)

                                    # 保存到 session_state
                                    st.session_state['model_score'] = score
                                    st.session_state['model'] = model
                                    st.session_state['scaler_X'] = scaler_X
                                    st.session_state['scaler_y'] = scaler_y

                                st.success("代理模型训练完成！")
                                st.info(f"模型得分（R2）：{score:.4f}")

                        # Step 4: 保存或重新训练
                        if 'model_score' in st.session_state:
                            st.info(f"模型得分（R2）：{st.session_state['model_score']:.4f}")

                            if st.button("保存模型surrogate.pkl至系统results文件内，以便后续GA调用"):
                                import joblib
                                joblib.dump({
                                    'model': st.session_state['model'],
                                    'scaler_X': st.session_state['scaler_X'],
                                    'scaler_y': st.session_state['scaler_y']
                                }, "results/surrogate.pkl")
                                st.success("模型已保存至系统results文件夹，文件名为surrogate.pkl")

                            if st.button("不满意，重新训练"):
                                st.session_state.pop('model_score', None)
                                st.session_state.pop('param_setting', None)
                                st.session_state.pop('scaler_X', None)
                                st.session_state.pop('scaler_y', None)
                                st.rerun()

                except Exception as e:
                    st.error(f"读取数据时出错: {e}")

        # 功能二独立的页面导航
        f2_page_nav()

    # 功能二页面1：参数设置
    elif st.session_state.f2_page == 1:
        from deap import creator, base, tools
        st.title("能量桩参数优化 - 参数设置")

        model_file = st.selectbox(
            "请选择要使用的模型文件:",
            ("预训练模型 surrogate_os.pkl", "自定义模型 surrogate.pkl")
        )
        if model_file == "预训练模型 surrogate_os.pkl":
            model_path = "surrogate_os.pkl"
            model_source = "预训练模型"
        else:
            model_path = "results/surrogate.pkl"
            model_source = "自定义模型"
        import joblib
        try:
            saved = joblib.load(model_path)
            st.session_state.surrogate = saved['model']
            st.session_state.scaler_X = saved.get('scaler_X', None)
            st.session_state.scaler_y = saved.get('scaler_y', None)
            st.session_state.model_loaded = True
            st.session_state.model_source = model_source
            st.success(f"{model_source}已加载！")
        except Exception as e:
            st.session_state.model_loaded = False
            st.error(f"模型文件加载失败: {e}")

        # 遗传算法参数设置
        st.subheader("遗传算法参数设置")
        col1, col2, col3 = st.columns(3)
        with col1:
            pop_size = st.number_input("种群大小", min_value=10, max_value=100, value=20)
        with col2:
            ngen = st.number_input("迭代次数", min_value=10, max_value=200, value=50)
        with col3:
            cxpb = st.number_input("交叉概率", min_value=0.1, max_value=0.9, step=0.1, value=0.5)

        col4, col5 = st.columns(2)
        with col4:
            mutpb = st.number_input("变异概率", min_value=0.1, max_value=0.9, step=0.1, value=0.3)
        with col5:
            tournsize = st.number_input("锦标赛大小", min_value=2, max_value=10, value=3)

        st.session_state.ga_params = {
            'pop_size': pop_size,
            'ngen': ngen,
            'cxpb': cxpb,
            'mutpb': mutpb,
            'tournsize': tournsize
        }

        # 约束条件设置
        st.subheader("约束条件设置")
        st.info("以下约束条件将用于优化过程")

        st.markdown("**静态约束**")
        col1, col2 = st.columns(2)
        with col1:
            re_min = st.number_input("最小雷诺数", value=4000)
        with col2:
            qv_min = st.number_input("最小体积流量(m³/h)", value=0.1)
            qv_max = st.number_input("最大体积流量(m³/h)", value=3.0)

        st.markdown("**动态约束**")
        col1, col2 = st.columns(2)
        with col1:
            z_top_max = st.number_input("最大桩顶位移(mm)", value=16.0)
            sigma_max = st.number_input("最大应力(kPa)", value=2000)
        with col2:
            t_out_max = st.number_input("夏季最高出口水温(℃)", value=40.0)
            t_out_min = st.number_input("冬季最低出口水温(℃)", value=5.0)

        st.session_state.constraints = {
            're_min': re_min,
            'qv_min': qv_min,
            'qv_max': qv_max,
            'z_top_max': z_top_max,
            'sigma_max': sigma_max,
            't_out_max': t_out_max,
            't_out_min': t_out_min
        }

        f2_page_nav()

    # 功能二页面2：优化运行
    elif st.session_state.f2_page == 2:
        from deap import creator, base, tools
        st.title("能量桩参数优化 - 优化运行")

        if not st.session_state.get('model_loaded', False):
            st.warning("请先加载或训练模型！")
            st.session_state.f2_page = 0
            st.rerun()

        st.info(f"当前使用模型: {st.session_state.get('model_source', '未知')}")

        # 初始化遗传算法
        if 'ga_initialized' not in st.session_state:
            # 创建类型
            if 'FitnessMin' in creator.__dict__:
                del creator.FitnessMin
            if 'Individual' in creator.__dict__:
                del creator.Individual

            creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
            creator.create('Individual', list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register('attr_rp', np.random.randint, 0, 5)
            toolbox.register('attr_N', np.random.randint, 0, 500)
            toolbox.register('attr_v', np.random.uniform, 0.1, 1.0)
            toolbox.register('attr_Lp', np.random.uniform, 0.05, 0.25)

            toolbox.register('individual', tools.initCycle, creator.Individual,
                             (toolbox.attr_rp, toolbox.attr_N,
                              toolbox.attr_v, toolbox.attr_Lp), n=1)
            toolbox.register('population', tools.initRepeat, list, toolbox.individual)


            # 评价函数
            def eval_ind(ind):
                # 离散变量取整
                x = np.array([int(round(ind[0])), int(round(ind[1])),
                              ind[2], ind[3]]).reshape(1, -1)
                x_scaled = st.session_state.scaler_X.transform(x)
                y_scaled = st.session_state.surrogate.predict(x_scaled)
                Re, qv, z_top, sigma_max, T_out_max, T_out_min, unbalance_rate, LCOE = \
                    st.session_state.scaler_y.inverse_transform(y_scaled)[0]

                # 约束违反量
                constraints = st.session_state.constraints
                cv = (max(0, constraints['re_min'] - Re) + \
                      max(0, constraints['qv_min'] - qv) + max(0, qv - constraints['qv_max']) + \
                      max(0, z_top - constraints['z_top_max']) + \
                      max(0, sigma_max - constraints['sigma_max']) + \
                      max(0, T_out_max - constraints['t_out_max']) + \
                      max(0, constraints['t_out_min'] - T_out_min))

                return (1e9,) if cv > 0 else (LCOE,)


            toolbox.register('evaluate', eval_ind)
            toolbox.register('mate', tools.cxUniform, indpb=0.5)
            toolbox.register('mutate', tools.mutPolynomialBounded,
                             eta=20, low=[0, 0, 0.1, 0.05],
                             up=[5, 500, 1.0, 0.25], indpb=0.3)
            toolbox.register('select', tools.selTournament, tournsize=st.session_state.ga_params['tournsize'])

            st.session_state.toolbox = toolbox
            st.session_state.ga_initialized = True

        # 运行优化
        if st.button("开始优化"):
            ga_params = st.session_state.ga_params

            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            best_fitness_chart = st.empty()

            # 用于记录最佳适应度历史
            best_fitness_history = []

            # 创建种群
            pop = st.session_state.toolbox.population(n=ga_params['pop_size'])

            # 评价初始种群
            fitnesses = list(map(st.session_state.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # 记录初始最佳个体
            best_ind = tools.selBest(pop, 1)[0]
            best_fitness_history.append(best_ind.fitness.values[0])

            # 开始进化
            for gen in range(ga_params['ngen']):
                # 选择下一代
                offspring = st.session_state.toolbox.select(pop, len(pop))

                # 克隆选中的个体
                offspring = list(map(st.session_state.toolbox.clone, offspring))

                # 交叉
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if np.random.random() < ga_params['cxpb']:
                        st.session_state.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # 变异
                for mutant in offspring:
                    if np.random.random() < ga_params['mutpb']:
                        st.session_state.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # 评价新个体
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(st.session_state.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # 替换种群
                pop[:] = offspring

                # 记录最佳个体
                best_ind_gen = tools.selBest(pop, 1)[0]
                if best_ind_gen.fitness.values[0] < best_ind.fitness.values[0]:
                    best_ind = best_ind_gen

                best_fitness_history.append(best_ind.fitness.values[0])

                # 更新进度
                progress = (gen + 1) / ga_params['ngen']
                progress_bar.progress(progress)
                status_text.text(f"迭代 {gen + 1}/{ga_params['ngen']}, 当前最佳LCOE: {best_ind.fitness.values[0]:.2f}")

                # 更新图表
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(best_fitness_history, 'b-', linewidth=2)
                ax.set_title("最佳适应度进化过程")
                ax.set_xlabel("迭代次数")
                ax.set_ylabel("LCOE (元/kWh)")
                ax.grid(True)
                best_fitness_chart.pyplot(fig)

                import time

                time.sleep(0.1)  # 稍微延迟以便显示更新

            # 保存最终结果
            best_x = [int(round(best_ind[0])), int(round(best_ind[1])),
                      best_ind[2], best_ind[3]]

            # 使用代理模型预测所有输出
            x_scaled = st.session_state.scaler_X.transform(np.array(best_x).reshape(1, -1))
            y_scaled = st.session_state.surrogate.predict(x_scaled)
            Re, qv, z_top, sigma_max, T_out_max, T_out_min, unbalance_rate, LCOE = \
                st.session_state.scaler_y.inverse_transform(y_scaled)[0]

            st.session_state.optimization_result = {
                'best_solution': best_x,
                'best_fitness': best_ind.fitness.values[0],
                'Re': Re,
                'qv': qv,
                'z_top': z_top,
                'sigma_max': sigma_max,
                'T_out_max': T_out_max,
                'T_out_min': T_out_min,
                'unbalance_rate': unbalance_rate,
                'LCOE': LCOE
            }

            st.success("优化完成！")
            st.session_state.show_results = True

        # 显示结果
        if st.session_state.get('show_results', False):
            result = st.session_state.optimization_result
            st.subheader("优化结果")

            # 显示最优参数组合
            st.markdown("### 最优参数组合")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("管径编号", result['best_solution'][0])
            with col2:
                st.metric("能量桩数量", result['best_solution'][1])
            with col3:
                st.metric("管内流速(m/s)", f"{result['best_solution'][2]:.3f}")
            with col4:
                st.metric("螺距(m)", f"{result['best_solution'][3]:.3f}")

            # 显示优化目标
            st.markdown("### 优化目标")
            st.metric("最小LCOE(元/kWh)", f"{result['LCOE']:.2f}")

            # 显示其他关键指标
            st.markdown("### 关键指标")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("桩顶位移(mm)", f"{result['z_top']:.2f}")
            with col2:
                st.metric("最大热应力(kPa)", f"{result['sigma_max']:.2f}")
            with col3:
                st.metric("夏季出口水温(℃)", f"{result['T_out_max']:.2f}")
            with col4:
                st.metric("冬季出口水温(℃)", f"{result['T_out_min']:.2f}")

            # 显示约束条件满足情况
            st.markdown("### 约束条件满足情况")
            constraints = st.session_state.constraints

            # 创建约束检查表
            constraint_data = {
                "约束条件": ["雷诺数 > 4000",
                             "体积流量在0.1-3 m³/h",
                             f"桩顶位移 ≤ {constraints['z_top_max']} mm",
                             f"最大应力 ≤ {constraints['sigma_max']} kPa",
                             f"夏季出口水温 < {constraints['t_out_max']}℃",
                             f"冬季出口水温 > {constraints['t_out_min']}℃"],
                "实际值": [f"{result['Re']:.2f}",
                           f"{result['qv']:.2f} m³/h",
                           f"{result['z_top']:.2f} mm",
                           f"{result['sigma_max']:.2f} kPa",
                           f"{result['T_out_max']:.2f}℃",
                           f"{result['T_out_min']:.2f}℃"],
                "要求值": [f"＞ {constraints['re_min']}",
                           f"{constraints['qv_min']}-{constraints['qv_max']}",
                           f"≤ {constraints['z_top_max']}",
                           f"≤ {constraints['sigma_max']}",
                           f"< {constraints['t_out_max']}",
                           f"＞ {constraints['t_out_min']}"],
                "状态": ["✅ 满足" if result['Re'] > constraints['re_min'] else "❌ 不满足",
                         "✅ 满足" if constraints['qv_min'] <= result['qv'] <= constraints['qv_max'] else "❌ 不满足",
                         "✅ 满足" if result['z_top'] <= constraints['z_top_max'] else "❌ 不满足",
                         "✅ 满足" if result['sigma_max'] <= constraints['sigma_max'] else "❌ 不满足",
                         "✅ 满足" if result['T_out_max'] < constraints['t_out_max'] else "❌ 不满足",
                         "✅ 满足" if result['T_out_min'] > constraints['t_out_min'] else "❌ 不满足"]
            }

            st.table(pd.DataFrame(constraint_data))

            # # 约束满足情况可视化
            # st.subheader("约束条件满足情况可视化")
            # fig, ax = plt.subplots(figsize=(10, 6))

            # constraint_items = [
            #     ("雷诺数", result['Re'], constraints['re_min'], ">"),
            #     ("体积流量下限", result['qv'], constraints['qv_min'], "min"),
            #     ("体积流量上限", result['qv'], constraints['qv_max'], "max"),
            #     ("桩顶位移", result['z_top'], constraints['z_top_max'], "<"),
            #     ("最大应力", result['sigma_max'], constraints['sigma_max'], "<"),
            #     ("夏季出口水温", result['T_out_max'], constraints['t_out_max'], "<"),
            #     ("冬季出口水温", result['T_out_min'], constraints['t_out_min'], ">")
            # ]

            # labels = [c[0] for c in constraint_items]
            # values = [c[1] for c in constraint_items]
            # limits = [c[2] for c in constraint_items]
            # types = [c[3] for c in constraint_items]

            # # 创建颜色编码
            # colors = []
            # for i, t in enumerate(types):
            #     if t == ">" and values[i] > limits[i]:
            #         colors.append('green')
            #     elif t == "<" and values[i] < limits[i]:
            #         colors.append('green')
            #     elif t == "min" and values[i] >= limits[i]:
            #         colors.append('green')
            #     elif t == "max" and values[i] <= limits[i]:
            #         colors.append('green')
            #     else:
            #         colors.append('red')

            # # 绘制条形图
            # y_pos = np.arange(len(labels))
            # ax.barh(y_pos, values, color=colors, alpha=0.7)

            # # 添加限制线
            # for i, limit in enumerate(limits):
            #     if types[i] in [">", "min"]:
            #         ax.axvline(x=limit, color='r', linestyle='--')
            #         ax.text(limit, i, f' 最小限制: {limit}', va='center', color='r')
            #     elif types[i] in ["<", "max"]:
            #         ax.axvline(x=limit, color='r', linestyle='--')
            #         ax.text(limit, i, f' 最大限制: {limit}', va='center', color='r')

            # ax.set_yticks(y_pos)
            # ax.set_yticklabels(labels)
            # ax.set_xlabel("值")
            # ax.set_title("约束条件满足情况")
            # ax.grid(axis='x', alpha=0.3)

            # st.pyplot(fig)

        f2_page_nav()

    # 功能二页面3：结果导出
    elif st.session_state.f2_page == 3:
        st.title("能量桩参数优化 - 结果导出")

        if not st.session_state.get('show_results', False):
            st.warning("请先完成优化运行！")
            st.session_state.f2_page = 2
            st.rerun()

        result = st.session_state.optimization_result
        constraints = st.session_state.constraints

        st.subheader("优化结果摘要")
        # 管径数据表
        pipe_data = {
            "管径编号": [0, 1, 2, 3, 4, 5],
            "管外半径(m)": [0.010, 0.0125, 0.016, 0.02, 0.025, 0.0315],
            "管内半径(m)": [0.008, 0.0102, 0.013, 0.0162, 0.0202, 0.0252]
        }
        pipe_df = pd.DataFrame(pipe_data)
        # 取出最佳解对应的管径编号
        pipe_index = result['best_solution'][0]
        pipe_row = pipe_df.loc[pipe_df["管径编号"] == pipe_index].iloc[0]
        
        st.markdown("**最优LCOE:**")
        st.write(f"{result['LCOE']:.2f} 元/kWh")
        st.markdown("**最优参数组合如下:**")
        st.write(f"管径编号 = {result['best_solution'][0]}, 外半径={pipe_row['管外半径(m)']} m, 内半径={pipe_row['管内半径(m)']} m")
        st.write(f"能量桩数量 = {result['best_solution'][1]}")
        st.write(f"流速 = {result['best_solution'][2]:.3f} m/s")
        st.write(f"螺距 = {result['best_solution'][3]:.3f} m")

        st.subheader("导出结果")

        # 构造参数表
        result_df = pd.DataFrame({
            "参数": [
                "管径编号", "外半径(m)", "内半径(m)",
                "能量桩数量", "管内流速(m/s)", "螺距(m)",
                "LCOE(元/kWh)", "雷诺数", "体积流量(m³/h)", "桩顶位移(mm)",
                "最大应力(kPa)", "夏季出口最高水温(℃)", "冬季出口最低水温(℃)"
            ],
            "值": [
                result['best_solution'][0], pipe_row['管外半径(m)'], pipe_row['管内半径(m)'],
                result['best_solution'][1], result['best_solution'][2], result['best_solution'][3],
                result['LCOE'], result['Re'], result['qv'], result['z_top'],
                result['sigma_max'], result['T_out_max'], result['T_out_min']
            ]
        })

        # 显示结果表
        st.dataframe(result_df)

        # 导出选项
        export_format = st.radio("选择导出格式:", ("CSV", "Excel", "文本报告"))

        if export_format == "CSV":
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下载CSV结果",
                data=csv,
                file_name="energy_pile_optimization_results.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            try:
                from io import BytesIO

                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='优化结果')
                st.download_button(
                    label="下载Excel结果",
                    data=buffer,
                    file_name="energy_pile_optimization_results.xlsx",
                    mime="application/vnd.ms-excel"
                )
            except ImportError:
                st.warning("需要安装openpyxl才能导出Excel格式，请运行: pip install openpyxl")
        elif export_format == "文本报告":
            report = f"""
            =============================
            能量桩系统参数优化结果报告
            =============================

            生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
            使用模型: {st.session_state.get('model_source', '未知')}

            -----------------------------
            最优参数组合:
            -----------------------------
            管径编号: {result['best_solution'][0]}
            能量桩数量: {result['best_solution'][1]}
            管内流速: {result['best_solution'][2]:.3f} m/s
            螺距: {result['best_solution'][3]:.3f} m

            -----------------------------
            优化目标:
            -----------------------------
            最小LCOE: {result['LCOE']:.2f} 元/kWh

            -----------------------------
            约束条件满足情况:
            -----------------------------
            雷诺数: {result['Re']:.2f} (要求 > {constraints['re_min']}) 
            {'满足' if result['Re'] > constraints['re_min'] else '不满足'}

            体积流量: {result['qv']:.2f} m³/h (要求 {constraints['qv_min']}-{constraints['qv_max']}) 
            {'满足' if constraints['qv_min'] <= result['qv'] <= constraints['qv_max'] else '不满足'}

            桩顶位移: {result['z_top']:.2f} mm (要求 ≤ {constraints['z_top_max']}) 
            {'满足' if result['z_top'] <= constraints['z_top_max'] else '不满足'}

            最大应力: {result['sigma_max']:.2f} kPa (要求 ≤ {constraints['sigma_max']}) 
            {'满足' if result['sigma_max'] <= constraints['sigma_max'] else '不满足'}

            夏季出口最高水温: {result['T_out_max']:.2f}℃ (要求 < {constraints['t_out_max']}) 
            {'满足' if result['T_out_max'] < constraints['t_out_max'] else '不满足'}

            冬季出口最低水温: {result['T_out_min']:.2f}℃ (要求 > {constraints['t_out_min']}) 
            {'满足' if result['T_out_min'] > constraints['t_out_min'] else '不满足'}
            """

            st.download_button(
                label="下载文本报告",
                data=report,
                file_name="energy_pile_optimization_report.txt",
                mime="text/plain"
            )

        if st.button("重新开始优化"):
            # 重置优化状态
            keys_to_reset = ['ga_initialized', 'show_results', 'optimization_result']
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.f2_page = 0
            st.rerun()

        # if st.button("返回主菜单"):
        #     st.session_state['page'] = 0
        #     st.rerun()

        f2_page_nav()




elif st.session_state['page'] == 13:
    if st.button('返回首页', key='back_home_X'):
        st.session_state['page'] = 0
        st.rerun()
    # 初始化功能三的页面状态
    if 'f3_page' not in st.session_state:
        st.session_state.f3_page = 0

    # 功能三页面0：介绍页
    if st.session_state.f3_page == 0:
        st.title("SINDy-MPC 控制策略优化框架（未来工作方向）")
        st.markdown("""
            本功能为未来工作方向，旨在为用户展示能量桩系统的 SINDy-MPC 控制策略优化框架。
            通过实测数据或虚拟仿真得到的数据（需包含系统状态量 X 和控制输入 U），
            利用稀疏识别 SINDy 建立控制方程，用于系统的模型预测控制（MPC）。
            """)
        st.markdown("""
            该 SINDy-MPC 体系可应用于如下图所示的太阳能-能量桩-土壤源热泵系统。
            系统的理想工作模式如下图所示，包含多种运行方式。
            """)

        # 使用列布局并排显示图片
        col1, col2 = st.columns(2)
        with col1:
            st.image("SA-EP-GSHP.png", caption="太阳能-能量桩-土壤源热泵系统示意图")
        with col2:
            st.image("operating-modes.jpg", caption="系统理想工作模式")

        st.markdown("""
            **SINDy-MPC 核心优势：**
            1. 数据驱动建模：从系统运行数据中自动识别动力学模型
            2. 物理启发性：模型具有物理可解释性
            3. 高效优化：实现系统运行模式和控制变量的最优调度
            4. 实时反馈：构建可实时反馈的最优运行策略
            """)

        f3_page_nav()

    # 功能三页面1：数据读取
    elif st.session_state.f3_page == 1:
        st.title("实验数据或仿真数据读取")
        st.markdown("""
            **本页内容：**
            - 读取用户上传的数据文件
            - 数据文件应包含系统状态量 X 和控制输入 U
            - 由于该功能仅作未来工作方向展示，实际使用模板数据
            """)

        # 上传文件区域
        uploaded_file = st.file_uploader('上传数据文件（CSV或Excel）', type=['csv', 'xlsx'])

        if uploaded_file is not None:
            # 读取数据
            if uploaded_file.name.endswith('.csv'):
                result_df = pd.read_csv(uploaded_file)
            else:
                result_df = pd.read_excel(uploaded_file)

            st.write("数据预览：")
            st.dataframe(result_df.head())

            # 检查必要列
            required_columns = ['Q', 'COP', 'PLR', 'T_in', 'T_initial', 'T_gs', 'T_out', 'T_point']
            missing_columns = [col for col in required_columns if col not in result_df.columns]

            if missing_columns:
                st.error(f"数据文件缺少必要列: {', '.join(missing_columns)}")
            else:
                # 提取数据
                Q = result_df['Q'].values
                COP = result_df['COP'].values
                PLR = result_df['PLR'].values
                T_in = result_df['T_in'].values
                T_initial = result_df['T_initial'].values
                T_gs = result_df['T_gs'].values
                T_out = result_df['T_out'].values
                T_point = result_df['T_point'].values

                # 确保数据长度一致
                min_length = min(len(Q), len(COP), len(PLR), len(T_in), len(T_initial),
                                 len(T_gs), len(T_out), len(T_point))
                Q = Q[:min_length]
                COP = COP[:min_length]
                PLR = PLR[:min_length]
                T_in = T_in[:min_length]
                T_initial = T_initial[:min_length]
                T_gs = T_gs[:min_length]
                T_out = T_out[:min_length]
                T_point = T_point[:min_length]

                # 构建X和U
                X = np.vstack([T_point, T_out, T_in]).T
                U = np.vstack([Q, PLR, T_initial, T_gs]).T

                # 存入session_state
                st.session_state.X = X
                st.session_state.U = U

                st.success("数据读取成功！")
                st.write(f"X 维度: {X.shape}")
                st.write(f"U 维度: {U.shape}")

                # 显示数据统计信息
                st.subheader("数据统计摘要")
                st.write(pd.DataFrame(X, columns=['T_point', 'T_out', 'T_in']).describe())
        else:
            st.info("请上传数据文件或使用示例数据")
            if st.button("使用示例数据"):
                # 加载示例数据
                result_df = pd.read_excel("data_sindy.xlsx")
                Q = result_df['Q'].values
                COP = result_df['COP'].values
                PLR = result_df['PLR'].values
                T_in = result_df['T_in'].values
                T_initial = result_df['T_initial'].values
                T_gs = result_df['T_gs'].values
                T_out = result_df['T_out'].values
                T_point = result_df['T_point'].values

                # 截取Q长度与仿真数据一致
                Q = Q[:len(T_out)]
                X = np.vstack([T_point, T_out, T_in]).T
                U = np.vstack([Q, PLR, T_initial, T_gs]).T

                st.session_state.X = X
                st.session_state.U = U

                st.success("示例数据加载成功！")
                st.write(f"X 维度: {X.shape}")
                st.write(f"U 维度: {U.shape}")

                # 显示数据统计信息
                st.subheader("数据统计摘要")
                st.write(pd.DataFrame(X, columns=['T_point', 'T_out', 'T_in']).describe())

        st.markdown("""
            **数据说明：**
            1. 状态变量 X 包含能量桩桩群进口温度 T_in、能量桩桩群出口温度 T_out、桩壁中点（垂直方向）温度 T_point
            2. 控制输入变量 U 包含建筑逐时负荷 Q，热泵部分负荷率 PLR，土体初始温度分布 T_initial，边界温度扰动相应 T_gs
            """)

        f3_page_nav()

    # 功能三页面2：SINDy建模过程展示
    elif st.session_state.f3_page == 2:
        st.title("SINDy建模过程展示")

        if 'X' not in st.session_state or 'U' not in st.session_state:
            st.warning("请先上传或加载数据")
            st.button("返回数据上传页面", on_click=lambda: st.session_state.update(f3_page=1))
            f3_page_nav()
            st.stop()

        X = st.session_state.X
        U = st.session_state.U

        st.markdown("""
            **已成功读取状态变量 X 和控制输入 U，可用于后续 SINDy 建模**
            """)
        st.write(f"X 维度: {X.shape}")
        st.write(f"U 维度: {U.shape}")

        # 数据可视化
        st.subheader("状态变量可视化")
        fig, ax = plt.subplots(3, 1, figsize=(12, 8))
        ax[0].plot(X[:, 0], label='T_point')
        ax[0].set_ylabel("温度 (°C)")
        ax[0].legend()

        ax[1].plot(X[:, 1], label='T_out')
        ax[1].set_ylabel("温度 (°C)")
        ax[1].legend()

        ax[2].plot(X[:, 2], label='T_in')
        ax[2].set_ylabel("温度 (°C)")
        ax[2].set_xlabel("时间 (小时)")
        ax[2].legend()

        st.pyplot(fig)

        st.subheader("控制输入可视化")
        fig, ax = plt.subplots(4, 1, figsize=(12, 10))
        ax[0].plot(U[:, 0], label='Q')
        ax[0].set_ylabel("负荷 (kW)")
        ax[0].legend()

        ax[1].plot(U[:, 1], label='PLR')
        ax[1].set_ylabel("部分负荷率")
        ax[1].legend()

        ax[2].plot(U[:, 2], label='T_initial')
        ax[2].set_ylabel("温度 (°C)")
        ax[2].legend()

        ax[3].plot(U[:, 3], label='T_gs')
        ax[3].set_ylabel("温度 (°C)")
        ax[3].set_xlabel("时间 (小时)")
        ax[3].legend()

        st.pyplot(fig)

        # 数据集划分
        st.subheader("训练集和测试集划分")
        st.markdown("按照 7:3 比例划分训练集和测试集")

        if st.button("划分数据集"):
            n_total = len(X)
            n_train = int(n_total * 0.7)
            X_train, X_test = X[:n_train], X[n_train:]
            U_train, U_test = U[:n_train], U[n_train:]

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.U_train = U_train
            st.session_state.U_test = U_test

            st.success(f"数据集划分完成！训练集样本数: {n_train}，测试集样本数: {n_total - n_train}")
            st.write("训练集统计摘要:")
            st.write(pd.DataFrame(X_train, columns=['T_point', 'T_out', 'T_in']).describe())

        # 数据归一化
        if 'X_train' in st.session_state:
            st.subheader("数据归一化")
            st.markdown("对训练集和测试集分别进行归一化，防止信息泄露")

            if st.button("执行归一化"):
                from sklearn.preprocessing import MinMaxScaler

                scaler_X = MinMaxScaler(feature_range=(-1, 1))
                scaler_U = MinMaxScaler(feature_range=(-1, 1))

                X_train_scaled = scaler_X.fit_transform(st.session_state.X_train)
                U_train_scaled = scaler_U.fit_transform(st.session_state.U_train)

                X_test_scaled = scaler_X.transform(st.session_state.X_test)
                U_test_scaled = scaler_U.transform(st.session_state.U_test)

                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.U_train_scaled = U_train_scaled
                st.session_state.X_test_scaled = X_test_scaled
                st.session_state.U_test_scaled = U_test_scaled
                st.session_state.scaler_X = scaler_X
                st.session_state.scaler_U = scaler_U

                st.success("归一化完成！")
                st.write("归一化后的训练集示例:")
                st.write(X_train_scaled[:5])

        f3_page_nav()

    # 功能三页面3：SINDy建模
    elif st.session_state.f3_page == 3:
        st.title("SINDy建模（离散时间递推）")

        if 'X_train_scaled' not in st.session_state:
            st.warning("请先完成数据预处理")
            st.button("返回数据预处理页面", on_click=lambda: st.session_state.update(f3_page=2))
            f3_page_nav()
            st.stop()

        st.markdown("""
            **SINDy建模与模型训练**
            采用稀疏识别 SINDy 方法，在Python环境中使用pysindy进行建模。
            采用 STLSQ 优化器和有限差分法进行建模。
            """)

        # 展示建模参数
        st.subheader("建模参数选择")
        st.markdown("""
            - 优化器 optimizer: `ps.STLSQ`
            - threshold: `0.02`
            - differentiation_method: `ps.FiniteDifference(drop_endpoints=True)`
            """)

        if st.button("执行SINDy建模"):
            # 创建进度条占位符 - 只创建一次
            progress_bar = st.progress(0, text="模型训练进度: 0%")
            status_text = st.empty()

            with st.spinner("SINDy模型训练中..."):
                import time

                # 使用同一个进度条进行更新
                for i in range(1, 6):
                    time.sleep(1)
                    progress = i * 20
                    progress_bar.progress(progress, text=f"模型训练进度: {progress}%")
                    status_text.text(f"当前阶段: 特征工程 ({i}/5)")

                # 模拟建模结果
                st.success("SINDy建模与训练完成！")
                progress_bar.empty()  # 完成后移除进度条
                st.subheader("模型结构")
                st.code("""
                    (x0)' = 0.262 x0 + -0.609 x1 + 0.350 x2 + -0.209 x0 u1 + -0.084 x0 u3 + 0.339 x1 u1 + 0.088 x1 u3 + -0.128 x2 u1
                    (x1)' = 0.621 x0 + -1.238 x1 + 0.618 x2 + -0.229 x0 u1 + 0.103 x0 u2 + 0.033 x0 u3 + 0.387 x1 u1 + -0.133 x1 u2 + -0.035 x1 u3 + -0.155 x2 u1 + 0.025 x2 u2
                    (x2)' = -0.005 1 + 1.109 x0 + -1.942 x1 + 0.819 x2 + 0.024 u0 + 0.046 x0 u1 + -0.019 x0 u2 + -0.149 x0 u3 + -0.053 x1 u1 + -0.023 x1 u2 + 0.203 x1 u3 + 0.001 x2 u0 + 0.047 x2 u2 + -0.050 x2 u3 + -0.007 u0 u2
                    """, language="python")

                st.subheader("模型评估")
                st.write("模型得分: 0.817117")

                # 保存模型（模拟）
                st.session_state.sindy_model = {
                    'equation': """
                        (x0)' = 0.262 x0 + -0.609 x1 + 0.350 x2 + -0.209 x0 u1 + -0.084 x0 u3 + 0.339 x1 u1 + 0.088 x1 u3 + -0.128 x2 u1
                        (x1)' = 0.621 x0 + -1.238 x1 + 0.618 x2 + -0.229 x0 u1 + 0.103 x0 u2 + 0.033 x0 u3 + 0.387 x1 u1 + -0.133 x1 u2 + -0.035 x1 u3 + -0.155 x2 u1 + 0.025 x2 u2
                        (x2)' = -0.005 1 + 1.109 x0 + -1.942 x1 + 0.819 x2 + 0.024 u0 + 0.046 x0 u1 + -0.019 x0 u2 + -0.149 x0 u3 + -0.053 x1 u1 + -0.023 x1 u2 + 0.203 x1 u3 + 0.001 x2 u0 + 0.047 x2 u2 + -0.050 x2 u3 + -0.007 u0 u2
                        """,
                    'score': 0.817117
                }

        f3_page_nav()

    # 功能三页面4：SINDy仿真
    elif st.session_state.f3_page == 4:
        st.title("SINDy仿真与结果可视化")

        if 'sindy_model' not in st.session_state:
            st.warning("请先完成SINDy建模")
            st.button("返回建模页面", on_click=lambda: st.session_state.update(f3_page=3))
            f3_page_nav()
            st.stop()

        st.markdown("""
        **SINDy仿真（simulation）与训练集结果对比**
        采用积分器参数 integrator_kws 进行仿真。
        """)

        st.subheader("积分器参数")
        st.markdown("""
        - method: "RK45"
        - rtol: 1e-6
        - atol: 1e-6
        """)

        if st.button("执行SINDy仿真"):
            # 创建单一进度条
            progress_bar = st.progress(0, text="仿真进度: 0%")
            status_text = st.empty()

            with st.spinner("仿真进行中..."):
                import time

                # 使用同一个进度条进行更新
                for i in range(1, 11):
                    time.sleep(0.3)
                    progress = i * 10
                    progress_bar.progress(progress, text=f"仿真进度: {progress}%")
                    status_text.text(f"当前步骤: {i}/10")

                st.success("仿真完成！")
                progress_bar.empty()  # 完成后移除进度条

                # 显示仿真结果
                st.subheader("T_out温度对比")
                st.image("T_out.png", caption="T_out Temperature Comparison")

                

        f3_page_nav()

    # 功能三页面5：SINDy-MPC体系展示
    elif st.session_state.f3_page == 5:
        st.title("SINDy-MPC体系展示")
        st.markdown("""
            1. **SINDy-MPC体系介绍：**
            训练获得的SINDy模型可作为控制方程，用于model predictive control体系的建立。在系统参数优化的基础上，开展运行策略优化，以协调能量桩-太阳能-地源热泵系统在长期运行过程中的效率与热平衡。为此，采用基于稀疏识别的模型预测控制（SINDy-MPC）方法，实现系统运行模式和控制变量的最优调度。该方法通过从系统运行数据中自动识别动力学模型，再将识别模型嵌入MPC预测优化框架中，构建可实时反馈的最优运行策略。SINDy-MPC兼具数据驱动与物理启发性，在运行优化中具备良好的泛化能力与解释性。
            """)

        st.markdown("2. **MPC可表达为：**")
        st.image("mpc-equation.jpg", caption="MPC数学表达式")

        st.markdown("3. **对于SA-EP-GSHP系统，成本函数可设计为：**")
        st.image("cost-function.jpg", caption="成本函数设计")

        # st.markdown("""
        #     4. **使用训练所得的SINDy模型作为 $X_{k+1}=f(x_k,u_k)$，进执行后续步骤：**
        #     - i. 约束条件设计：保证运行安全（如力学形变 $z_{top}$、$\sigma_{max}$ 和能量桩群出口水温 T4）以及减少长期热不平衡（土壤平均温度波动控制在初始温度 ±1℃范围内，防止热失衡）
        #     - ii. 控制优化：将上述目标与约束构成优化问题，采用粒子群优化算法（Particle Swarm Optimization, PSO）求解最优控制变量序列
        #     - iii. 策略执行与反馈：实施当前时间步的最优控制输入，进入下一个滚动周期 $u_t$。
        #     """)
        st.markdown(r"""
            4. **使用训练所得的SINDy模型作为 $X_{k+1}=f(x_k,u_k)$，进执行后续步骤：**
            - i. 约束条件设计：保证运行安全（如力学形变 $z_{top}$、$\sigma_{max}$ 和能量桩群出口水温 T4）以及减少长期热不平衡（土壤平均温度波动控制在初始温度 ±1℃范围内，防止热失衡）
            - ii. 控制优化：将上述目标与约束构成优化问题，采用粒子群优化算法（Particle Swarm Optimization, PSO）求解最优控制变量序列
            - iii. 策略执行与反馈：实施当前时间步的最优控制输入，进入下一个滚动周期 $u_t$。
            """)

        st.markdown("""
            5. **最终实现：**
            SINDy-MPC融合数据驱动建模与前馈-反馈控制机制，特别适用于本类非线性、耦合复杂、具有工程约束的建筑能源系统。在不依赖详细物理模型的前提下，SINDy能够提取系统核心动力学规律，并以简洁的形式嵌入MPC框架中，实现高效、可解释的智能运行控制策略，为能量桩-太阳能-热泵复合系统的全年稳定运行提供了有效支撑。
            """)

        if st.button("点击查看平台最终设计目标"):
            st.image("final_constructure.png", caption="平台最终设计目标展示")

        f3_page_nav()