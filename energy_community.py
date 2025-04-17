import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import datetime
import pytz

# 设置页面布局
st.set_page_config(
    page_title="能源社区智能管理系统",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .plot-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background: #2c3e50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# 生成模拟数据
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="h")  # 修改这里 H→h
    households = [f"家庭{i:03d}" for i in range(1, 101)]

    data = []
    for date in dates:
        base_load = np.random.normal(3, 0.5, 100)  # 基础用电负荷
        solar_gen = np.random.uniform(0, 5, 100) * (1 + 0.5 * np.sin(date.hour / 24 * 2 * np.pi))  # 太阳能发电
        battery_level = np.random.uniform(0, 10, 100)  # 储能系统电量
        ev_charging = np.random.poisson(0.5, 100) * (date.hour in [7, 8, 18, 19])  # 充电桩使用

        for i in range(100):
            data.append({
                "时间": date,
                "家庭ID": households[i],
                "用电量(kWh)": max(0, base_load[i] + 0.2 * ev_charging[i]),
                "太阳能发电量(kWh)": max(0, solar_gen[i]),
                "储能系统电量(kWh)": max(0, min(10, battery_level[i])),
                "充电桩使用量(kWh)": ev_charging[i]
            })

    return pd.DataFrame(data)


# 加载数据
@st.cache_data
def load_data():
    return generate_data()


df = load_data()


# 预测模型
def train_model(data):
    # 准备特征
    data['小时'] = data['时间'].dt.hour
    data['星期'] = data['时间'].dt.dayofweek
    data['是否周末'] = data['星期'] >= 5

    # 训练模型
    X = data[['小时', '星期', '是否周末', '太阳能发电量(kWh)', '储能系统电量(kWh)']]
    y = data['用电量(kWh)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mae


model, mae = train_model(df.copy())


# 预测未来负荷
def predict_future(model, last_date):
    future_dates = pd.date_range(start=last_date + datetime.timedelta(hours=1),
                                 periods=24, freq="h")  # 修改这里 H→h

    predictions = []
    for date in future_dates:
        # 假设太阳能和储能的简单模式
        solar = 3 * (1 + 0.5 * np.sin(date.hour / 24 * 2 * np.pi))
        battery = 5 + 2 * np.sin(date.hour / 12 * np.pi)

        features = pd.DataFrame({
            '小时': [date.hour],
            '星期': [date.dayofweek],
            '是否周末': [date.dayofweek >= 5],
            '太阳能发电量(kWh)': [solar],
            '储能系统电量(kWh)': [battery]
        })

        pred = model.predict(features)[0]
        predictions.append({
            '时间': date,
            '预测用电量(kWh)': max(0, pred)
        })

    return pd.DataFrame(predictions)


# 侧边栏
with st.sidebar:
    st.title("🌍 能源社区控制面板")
    selected_household = st.selectbox("选择家庭", df['家庭ID'].unique())
    selected_date = st.date_input("选择日期", datetime.date(2023, 1, 15))
    selected_metric = st.selectbox("查看指标", ["用电量", "太阳能发电量", "储能系统电量", "充电桩使用量"])

    st.markdown("---")
    st.markdown("### 系统信息")
    st.markdown(f"**数据时间范围:** {df['时间'].min().date()} 至 {df['时间'].max().date()}")
    st.markdown(f"**家庭数量:** {len(df['家庭ID'].unique())}")
    st.markdown(f"**预测模型MAE:** {mae:.2f} kWh")

# 主页面
st.title("🏡 能源社区智能管理系统")
st.markdown("### 社区能源使用实时监控与分析")

# 指标卡片
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_consumption = df['用电量(kWh)'].sum()
    st.metric("总用电量", f"{total_consumption:,.0f} kWh", delta="-5% vs 上月")
with col2:
    total_solar = df['太阳能发电量(kWh)'].sum()
    st.metric("总太阳能发电", f"{total_solar:,.0f} kWh", delta="+12% vs 上月")
with col3:
    avg_battery = df['储能系统电量(kWh)'].mean()
    st.metric("平均储能水平", f"{avg_battery:.1f} kWh", delta="+3% vs 上月")
with col4:
    total_ev = df['充电桩使用量(kWh)'].sum()
    st.metric("充电桩总使用量", f"{total_ev:,.0f} kWh", delta="+22% vs 上月")

# 数据筛选
selected_date = pd.Timestamp(selected_date)
filtered_df = df[(df['家庭ID'] == selected_household) &
                 (df['时间'].dt.date == selected_date.date())].copy()

community_day_df = df[df['时间'].dt.date == selected_date.date()].groupby('时间').sum().reset_index()

# 图表展示
tab1, tab2, tab3, tab4 = st.tabs(["家庭用电分析", "社区能源概况", "负荷预测", "数据详情"])

with tab1:
    st.markdown(f"### 家庭 {selected_household} - {selected_date.date()} 能源使用情况")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(filtered_df, x="时间", y="用电量(kWh)",
                      title="每小时用电量", height=300)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(filtered_df, x="时间", y="储能系统电量(kWh)",
                      title="储能系统电量变化", height=300)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(filtered_df, x="时间", y="太阳能发电量(kWh)",
                      title="太阳能发电量", height=300)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(filtered_df, x="时间", y="充电桩使用量(kWh)",
                     title="充电桩使用情况", height=300)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(f"### 社区整体能源概况 - {selected_date.date()}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(community_day_df, x="时间", y="用电量(kWh)",
                      title="社区总用电负荷", height=300)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(community_day_df, x="时间", y=["太阳能发电量(kWh)", "充电桩使用量(kWh)"],
                      title="发电与充电对比", height=300)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.area(community_day_df, x="时间", y="太阳能发电量(kWh)",
                      title="社区太阳能发电总量", height=300)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # 用电量分布
        hour_avg = df.groupby(df['时间'].dt.hour)['用电量(kWh)'].mean().reset_index()
        fig = px.bar(hour_avg, x="时间", y="用电量(kWh)",
                     title="每日平均用电模式", height=300)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 未来24小时负荷预测")

    last_date = df['时间'].max()
    future_df = predict_future(model, last_date)

    fig = px.line(future_df, x="时间", y="预测用电量(kWh)",
                  title="用电负荷预测", height=400)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 预测详情")
    st.dataframe(future_df.style.background_gradient(cmap='Blues'))

with tab4:
    st.markdown("### 原始数据浏览")
    st.dataframe(filtered_df.drop(columns=['家庭ID']).style.format({
        '用电量(kWh)': '{:.2f}',
        '太阳能发电量(kWh)': '{:.2f}',
        '储能系统电量(kWh)': '{:.2f}',
        '充电桩使用量(kWh)': '{:.2f}'
    }))

# 页脚
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d;">
        <p>能源社区智能管理系统 v1.0 · 数据更新于 {}</p>
        <p>© 2023 408工作室 · 所有权利保留</p>
    </div>
""".format(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')),
            unsafe_allow_html=True)
