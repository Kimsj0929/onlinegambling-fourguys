import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 페이지 기본 설정
st.set_page_config(page_title="온라인 카지노 데이터 시각화", layout="wide")

st.title("3. 데이터 시각화 (Plotly 버전)")
st.markdown("전처리 전 `onlineCasino.csv` 데이터를 분석하고 시각화하는 페이지입니다.")

# 1. 데이터 로드 및 캐싱
@st.cache_data
def load_data():
    df = pd.read_csv("onlineCasino.csv")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("📂 'onlineCasino.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
    st.stop()

# 2. 레이아웃 구성을 위한 탭 생성
tab1, tab2, tab3, tab4 = st.tabs(["🔥 히트맵 (상관관계)", "✨ 산점도 (관계 분석)", "📊 분포 차트", "📦 박스 플롯 (비교)"])

# --- 🔥 1. 히트맵 (Heatmap) ---
with tab1:
    st.subheader("변수 간 상관관계 히트맵")
    st.markdown("수치형 변수들 간의 상관계수를 시각화하여 어떤 변수들이 서로 밀접한지 확인합니다.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_matrix = numeric_cols.corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="수치형 변수 상관관계 행렬 (Correlation Matrix)"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# --- ✨ 2. 산점도 (Scatter Plot) ---
with tab2:
    st.subheader("참여 게이머 수 vs 총 판돈(Money) 산점도")
    st.markdown("> 💡 **Tip:** 데이터가 많아 렉을 방지하고 시각화 직관성을 높이기 위해 **무작위로 2,000건을 샘플링**하여 출력합니다.")
    
    df_sample = df.sample(n=min(2000, len(df)), random_state=42)
    
    fig_scatter = px.scatter(
        df_sample,
        x="gamers",
        y="money",
        color="moderator",
        hover_data=["ticks", "outpay"],
        title="게이머 수와 판돈의 관계 (중재자 여부별 색상 구분)",
        labels={"gamers": "참여 게이머 수", "money": "총 판돈 (Money)"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- 📊 3. 분포 차트 (Distribution Chart) ---
with tab3:
    st.subheader("배당률(Ticks) 분포 차트")
    st.markdown("현재 데이터의 배당률(`ticks`)은 극단적인 이상치가 존재합니다. 슬라이더를 조절하여 원하는 분포 범위를 확인해 보세요.")
    
    max_tick_slider = st.slider("시각화할 최대 배당률(Ticks) 제한", min_value=5, max_value=200, value=30, step=5)
    filtered_df_ticks = df[df["ticks"] <= max_tick_slider]
    
    fig_dist = px.histogram(
        filtered_df_ticks,
        x="ticks",
        nbins=50,
        title=f"배당률(Ticks) 분포 (Ticks가 {max_tick_slider} 이하인 데이터 대상)",
        labels={"ticks": "배당률 (Ticks)"},
        color_discrete_sequence=["#1f77b4"]
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# --- 📦 4. 박스 플롯 (Box Plot) ---
with tab4:
    st.subheader("중재자 유무에 따른 판돈(Money) 분포 비교")
    st.markdown("중재자(`moderator`) 유무에 따라 판돈의 흐름이나 격차가 어떻게 다른지 박스 플롯으로 비교합니다.")
    
    show_all = st.checkbox("아웃라이어(이상치)를 포함한 전체 데이터로 보기", value=False)
    
    if not show_all:
        q_95 = df["money"].quantile(0.95)
        filtered_df_box = df[df["money"] <= q_95]
        box_title = f"중재자 유무별 판돈 분포 (상위 5% 이상치 제외, Money <= {q_95:.1f})"
    else:
        filtered_df_box = df
        box_title = "중재자 유무별 판돈 분포 (전체 데이터)"
        
    fig_box = px.box(
        filtered_df_box,
        x="moderator",
        y="money",
        color="moderator",
        title=box_title,
        labels={"moderator": "중재자 여부 (Moderator)", "money": "총 판돈 (Money)"}
    )
    st.plotly_chart(fig_box, use_container_width=True)
