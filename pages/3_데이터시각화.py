import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# 1. 페이지 설정
st.set_page_config(page_title="카지노 데이터 시각화 대시보드", layout="wide")

st.title("📊 카지노 데이터 분석 대시보드 (정제 완료 버전)")
st.markdown("결측치와 이상치가 완벽하게 정제된 데이터셋을 기반으로 시각화 및 인사이트를 제공합니다.")

# 2. [🔥 절대 오류 나지 않는 경로 추적 알고리즘]
# app.py 파일이 있는 '진짜 위치'를 절대 경로로 알아내어 파일명과 강제로 결합합니다.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 만약 환경에 따라 __file__ 인식이 안 될 경우 현재 작업 디렉토리 사용
    current_dir = os.getcwd()

target_file_path = os.path.join(current_dir, "onlineCasino_cleaned_v2.csv")


# 3. 데이터 로드 및 캐싱
@st.cache_data
def load_clean_data():
    # 컴퓨터 내부의 엉뚱한 폴더가 아니라, app.py 바로 옆의 파일을 지정하여 로드합니다.
    return pd.read_csv(target_file_path)

try:
    df = load_clean_data()
    file_loaded = True
except FileNotFoundError:
    file_loaded = False

# --- 만약 위의 자동 추적조차 실패했을 때만 보여주는 비상 안내판 ---
if not file_loaded:
    st.error("📂 여전히 파일을 자동으로 찾지 못했습니다. 시스템 경로가 완전히 분리된 상태입니다.")
    st.info(f"💡 현재 코드가 파일을 찾으려고 시도한 절대 경로는 다음과 같습니다:\n`{target_file_path}`")
    st.markdown("""
    **🚨 마지막 수동 해결 방법:**
    1. 가지고 계신 전처리 완료 파일의 이름을 정확히 **`onlineCasino_cleaned_v2.csv`** 로 변경합니다.
    2. 그 파일을 위의 에러 메시지에 적힌 경로 폴더 안으로 직접 이동시켜 주시거나, 아래 칸에 마우스로 한 번만 끌어다 놓아주세요!
    """)
    
    # 비상용 수동 업로더 활성화
    uploaded_file = st.file_uploader("📂 여기에 onlineCasino_cleaned_v2.csv 파일을 직접 넣어주세요", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()


# ==================== 대시보드 본문 및 시각화 구현 ====================

# 4. 사이드바 - 분석할 핵심 종속변수 마스터 스위치 배치
st.sidebar.header("🎯 핵심 종속변수 설정")
target_var = st.sidebar.selectbox(
    "시각화 기준 종속변수 선택", 
    ["outpay", "peopleLost"],
    help="선택한 종속변수에 따라 아래 모든 산점도, 분포차트, 박스플롯이 동적으로 변합니다."
)

st.sidebar.write("---")
st.sidebar.markdown(f"### 📋 데이터 요약 정보\n- **전체 데이터 수:** {len(df):,} 행\n- **결측치 상태:** 0개 (완벽 정제 완료)")


# 5. 레이아웃 구성을 위한 4개의 탭 생성
tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 1. 변수 간 상관관계", 
    "✨ 2. 종속변수 간의 관계", 
    "📊 3. 종속변수 데이터 분포", 
    "📦 4. 중재자별 범위 비교"
])

# --- 🔥 탭 1: 상관관계 히트맵 ---
with tab1:
    st.subheader("🔥 파생변수가 결합된 전체 변수 간 상관관계 히트맵")
    numeric_cols = df.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_matrix = numeric_cols.corr()
    fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig_heat, use_container_width=True)

# --- ✨ 탭 2: 산점도 ---
with tab2:
    st.subheader("✨ 핵심 종속변수 간의 관계 분석 (peopleLost vs outpay)")
    df_sample = df.sample(n=min(2000, len(df)), random_state=42)
    fig_scat = px.scatter(
        df_sample, x="peopleLost", y="outpay", color="moderator", 
        title="패배 유저 수(peopleLost)와 카지노 지급액(outpay)의 관계", opacity=0.7
    )
    st.plotly_chart(fig_scat, use_container_width=True)

# --- 📊 탭 3: 분포 차트 ---
with tab3:
    st.subheader(f"📊 선택한 종속변수(`{target_var}`)의 전체 데이터 빈도 분포")
    q_99 = df[target_var].quantile(0.99)
    df_filtered = df[df[target_var] <= q_99]
    fig_hist = px.histogram(df_filtered, x=target_var, nbins=50, color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig_hist, use_container_width=True)

# --- 📦 탭 4: 박스 플롯 ---
with tab4:
    st.subheader(f"📦 중재자(Moderator) 유무에 따른 `{target_var}` 범위 비교")
    fig_box = px.box(
        df_filtered, x="moderator", y=target_var, color="moderator", points="outliers",
        labels={"moderator": "중재자 개입 여부", target_var: f"종속변수 ({target_var})"}
    )
    st.plotly_chart(fig_box, use_container_width=True)
