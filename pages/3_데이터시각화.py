import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 페이지 기본 설정 및 다크 테마(경고 분위기) 적용
st.set_page_config(
    page_title="도박의 수학적 진실: 당신이 무조건 잃는 이유",
    page_icon="🚫",
    layout="wide"
)

# 대시보드 제목
st.title("🚫 도박의 수학적 진실: 시스템은 어떻게 당신을 파산시키는가")
st.markdown("""
본 프로젝트는 실제 온라인 크래시 게임의 데이터 로그를 기반으로, **도박이 왜 '수학적으로 절대 딸 수 없는 게임'인지**를 증명하고 경각심을 주고자 제작된 공익 대시보드입니다.
""")
st.write("---")

# 2. 데이터 로드
@st.cache_data
def load_data():
    # 데이터셋 읽기
    df = pd.read_csv("onlineCasino_cleaned_v2.csv")
    # 누적 하우스 수익 계산
    df['cumulative_house_profit'] = df['net_profit'].cumsum()
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("데이터 파일('onlineCasino_cleaned_v2.csv')을 찾을 수 없습니다. 경로를 확인해주세요.")
    st.stop()

# ==========================================
# 팩트 폭격 1: 하우스 엣지 (누적 수익 곡선)
# ==========================================
st.header("📉 FACT 1. 플레이가 길어질수록 카지노의 이익은 우상향한다")
st.markdown("""
간혹 유저가 돈을 따는 라운드도 있습니다. 하지만 수천, 수만 판이 누적되면 어떻게 될까요? 
아래 그래프는 **시간 흐름에 따른 카지노 운영사의 누적 순이익**입니다. 개개인은 돈을 딸지 몰라도, 전체 유저의 돈은 결국 카지노의 주머니로 수렴합니다.
""")

# Plotly를 이용한 고도의 인터랙티브 라인 차트
fig_cum = px.line(
    df, 
    x='time', 
    y='cumulative_house_profit',
    title='시간 경과에 따른 카지노 운영사 누적 순이익 (House Profit Line)',
    labels={'time': '시간(Time)', 'cumulative_house_profit': '카지노 누적 이익 ($)'},
    color_discrete_sequence=['#FF4B4B']
)
fig_cum.update_layout(hovermode="x unified", template="plotly_dark")
st.plotly_chart(fig_cum, use_container_width=True)


# ==========================================
# 팩트 폭격 2: 즉시 파산 확률 (Instant Crash)
# ==========================================
st.write("---")
st.header("🚨 FACT 2. 당신이 제어할 수 없는 '즉시 파산'의 함정")
st.markdown("""
크래시 게임은 시작하자마자(1.00배~1.05배) 터져버리는 **'즉시 파산(Instant Crash)'** 구간이 존재합니다. 
이 구간에서는 유저가 아무리 완벽한 타이밍에 출금(Cash-out) 버튼을 누르려고 해도 손을 쓸 수 없이 돈을 날리게 됩니다.
""")

col1, col2 = st.columns([1, 2])

with col1:
    # 1.05배 이하로 터진 확률 계산
    instant_crash_threshold = 1.05
    total_rounds = len(df)
    instant_crash_rounds = len(df[df['ticks_clean'] <= instant_crash_threshold])
    instant_crash_rate = (instant_crash_rounds / total_rounds) * 100
    
    st.metric(
        label="💥 무조건 돈을 잃는 '즉시 파산' 확률", 
        value=f"{instant_crash_rate:.2f} %",
        delta="예측 및 대응 불가능", delta_color="inverse"
    )
    st.info(f"전체 {total_rounds:,}번의 게임 중 {instant_crash_rounds:,}번은 유저가 베팅하자마자 카지노가 판돈을 몰수했습니다.")

with col2:
    # 배수(Ticks)의 분포 시각화 (초반에 극도로 몰려있음을 증명)
    fig_dist = px.histogram(
        df[df['ticks_clean'] <= 10],  # 가독성을 위해 10배 이하만 시각화
        x='ticks_clean',
        nbins=50,
        title='게임 배수(Ticks) 분포 (10배 이하 구간)',
        labels={'ticks_clean': '종료 배수'},
        color_discrete_sequence=['#FFA15A']
    )
    fig_dist.update_layout(template="plotly_dark")
    st.plotly_chart(fig_dist, use_container_width=True)


# ==========================================
# 팩트 폭격 3: 시간대별 이성 상실 구간 분석
# ==========================================
st.write("---")
st.header("🌙 FACT 3. 이성이 마비되는 심야 시간대, 더 과감하게 잃는다")
st.markdown("""
하루 24시간 중 유저들이 가장 자제력을 잃고 큰 돈을 베팅하며, 결과적으로 카지노가 가장 많은 이득을 챙기는 시간대는 언제일까요?
""")

# 시간대별 평균 판돈 및 카지노 순이익 계산
hourly_stats = df.groupby('hour').agg({
    'money_clean': 'mean',
    'net_profit': 'sum'
}).reset_index()

fig_hour = go.Figure()
# 평균 판돈 바 차트
fig_hour.add_trace(go.Bar(
    x=hourly_stats['hour'], y=hourly_stats['money_clean'],
    name='평균 판돈 ($)', marker_color='#1f77b4', yaxis='y'
))
# 카지노 총 수익 라인 차트
fig_hour.add_trace(go.Scatter(
    x=hourly_stats['hour'], y=hourly_stats['net_profit'],
    name='카지노 총 수익 ($)', marker_color='#e377c2', yaxis='y2', mode='lines+markers'
))

# 레이아웃 조정 (이중 y축)
fig_hour.update_layout(
    title='시간대별 평균 판돈 및 카지노 총 수익 분포',
    xaxis=dict(title='시간 (0시~23시)', tickmode='linear'),
    yaxis=dict(title='평균 판돈 ($)', side='left'),
    yaxis2=dict(title='카지노 총 수익 ($)', side='right', overlaying='y', showgrid=False),
    template="plotly_dark",
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig_hour, use_container_width=True)

st.markdown("""
> **💡 시각화 결론:** > 대다수의 도박 사이트는 유저들이 판단력이 흐려지는 **새벽 시간대**에 베팅 규모가 커지며, 이에 맞춰 카지노의 수익 또한 극대화되는 경향을 보입니다. 도박은 당신의 심리적 취약점을 철저하게 이용합니다.
""")

st.write("---")
st.caption("© 2026 도박 중독 예방 캠페인 대시보드 - 데이터에 기반한 팩트 전달 프로젝트")
