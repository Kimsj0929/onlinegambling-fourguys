import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# 1. 페이지 설정
st.set_page_config(page_title="카지노 정제 데이터 분석 대시보드", layout="wide")

st.title("📊 카지노 데이터 분석 대시보드 (정제 완료 버전)")
st.markdown("결측치와 이상치가 완벽하게 정제된 데이터셋을 기반으로 시각화 및 인사이트를 제공합니다.")

# 2. [🔥 GitHub 최상위 루트 폴더 추적 알고리즘]
@st.cache_data
def load_clean_data():
    # 현재 코드가 pages/ 폴더 안에 있으므로, 한 단계 위로 올라가서(..) 최상위 루트의 파일을 찾습니다.
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    
    # 후보 1: 상위 폴더(루트)에 있는 csv 파일 경로 계산
    path_parent_root = os.path.abspath(os.path.join(current_dir, "..", "onlineCasino_cleaned_v2.csv"))
    
    # 후보 2: Streamlit Cloud 기본 루트 기준 상대 경로
    path_pure_root = "onlineCasino_cleaned_v2.csv"
    
    # 후보 3: 현재 코드가 있는 pages/ 폴더 안 자체 (혹시 나중에 옮기실까봐 방어 코드로 추가)
    path_side = os.path.join(current_dir, "onlineCasino_cleaned_v2.csv")
    
    # 존재하는 경로를 순서대로 탐색하여 로드
    if os.path.exists(path_parent_root):
        return pd.read_csv(path_parent_root)
    elif os.path.exists(path_pure_root):
        return pd.read_csv(path_pure_root)
    elif os.path.exists(path_side):
        return pd.read_csv(path_side)
    else:
        raise FileNotFoundError("GitHub 내 상위(루트) 폴더에서 파일을 찾지 못했습니다.")

try:
    df = load_clean_data()
    file_loaded = True
except FileNotFoundError:
    file_loaded = False

# --- 만약 경로가 완전 뒤틀려 실패했을 때만 작동하는 비상 수동 화면 ---
if not file_loaded:
    st.error("📂 [서버 파일 매칭 실패] GitHub 내에서 데이터를 찾을 수 없습니다.")
    st.markdown("""
    **🛠️ 최종 조치 방법:**
    서버 경로 추적이 차단된 상태입니다. 대시보드를 즉시 정상 작동시키려면 아래 칸에 파일을 한 번만 직접 마우스로 끌어다(Drag & Drop) 넣어주세요!
    """)
    uploaded_file = st.file_uploader("📂 테스트용 onlineCasino_cleaned_v2.csv 직접 넣기", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()


# ==================== 대시보드 본문 및 시각화 구현 ====================

# 3. 사이드바 - 분석할 핵심 종속변수 마스터 스위치 배치
st.sidebar.header("🎯 핵심 종속변수 설정")
target_var = st.sidebar.selectbox(
    "시각화 기준 종속변수 선택", 
    ["outpay", "peopleLost"],
    help="선택한 종속변수에 따라 아래 모든 산점도, 분포차트, 박스플롯이 동적으로 변합니다."
)

st.sidebar.write("---")
st.sidebar.markdown(f"### 📋 데이터 요약 정보\n- **전체 데이터 수:** {len(df):,} 행\n- **결측치 상태:** 0개 (완벽 정제 완료)")


# 4. 레이아웃 구성을 위한 4개의 탭 생성
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
