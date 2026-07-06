import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. 페이지 설정 (화면을 넓게 쓰는 와이드 모드)
st.set_page_config(page_title="카지노 데이터 시각화 대시보드", layout="wide")

st.title("📊 카지노 데이터 분석 대시보드 (정제 완료 버전)")
st.markdown("결측치와 이상치가 완벽하게 정제된 데이터셋을 기반으로 시각화 및 인사이트를 제공합니다.")

# 2. 데이터 자동 로드 및 캐싱 (성능 최적화)
@st.cache_data
def load_clean_data():
    # 전처리 완료된 파일 하나만 깔끔하게 로드합니다.
    return pd.read_csv("onlineCasino_cleaned_v2.csv")

try:
    df = load_clean_data()
    file_loaded = True
except FileNotFoundError:
    file_loaded = False

# 파일 로드 실패 시에만 안내 메시지 출력
if not file_loaded:
    st.error("📂 'onlineCasino_cleaned_v2.csv' 파일을 찾을 수 없습니다.")
    st.markdown("""
    **🛠️ 확인해 보세요:**
    이 `app.py` 파일이 저장된 **동일한 폴더(경로)** 안에 **`onlineCasino_cleaned_v2.csv`** 파일이 정확한 이름으로 들어있는지 꼭 확인해 주세요!
    """)
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


# 4. 레이아웃 구성을 위한 4개의 탭 생성 (깔끔한 UI 구성)
tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 1. 변수 간 상관관계", 
    "✨ 2. 종속변수 간의 관계", 
    "📊 3. 종속변수 데이터 분포", 
    "📦 4. 중재자별 범위 비교"
])


# --- 🔥 탭 1: 상관관계 히트맵 ---
with tab1:
    st.subheader("🔥 파생변수가 결합된 전체 변수 간 상관관계 히트맵")
    st.markdown("정제 완료된 수치형 변수들과 새로 생성한 파생변수들이 종속변수들과 어떠한 선형 관계를 가지는지 분석합니다.")
    
    # ID 컬럼을 제외한 수치형 데이터 상관계수 계산
    numeric_cols = df.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_matrix = numeric_cols.corr()
    
    fig_heat = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale="RdBu_r", 
        zmin=-1, zmax=1,
        title="수치형 변수 간 상관계수 시각화"
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# --- ✨ 탭 2: 산점도 (종속변수 2개의 관계 비교) ---
with tab2:
    st.subheader("✨ 핵심 종속변수 간의 관계 분석 (peopleLost vs outpay)")
    st.markdown("방에서 잃은 유저 수(`peopleLost`)가 많아질수록 카지노가 유저들에게 총 지급하는 금액(`outpay`)이 어떻게 변화하는지 분석합니다.")
    st.caption("💡 대용량 데이터로 인한 웹 브라우저 렉(느려짐) 방지를 위해 무작위 2,000건을 샘플링하여 시각화합니다.")
    
    # 웹 로딩 최적화를 위한 데이터 샘플링
    df_sample = df.sample(n=min(2000, len(df)), random_state=42)
    
    fig_scat = px.scatter(
        df_sample, 
        x="peopleLost", 
        y="outpay", 
        color="moderator",  # 중재자 여부로 색상 구분
        title="패배 유저 수(peopleLost)와 카지노 지급액(outpay)의 산점도 관계 분포",
        labels={"peopleLost": "패배 유저 수 (명)", "outpay": "총 지급액 (outpay)"},
        opacity=0.7
    )
    st.plotly_chart(fig_scat, use_container_width=True)


# --- 📊 탭 3: 분포 차트 (히스토그램) ---
with tab3:
    st.subheader(f"📊 선택한 종속변수(`{target_var}`)의 전체 데이터 빈도 분포")
    st.markdown(f"현재 사이드바에서 선택된 종속변수 **`{target_var}`**의 전체적인 데이터 밀도와 빈도를 관찰합니다.")
    
    # 이상치(상위 1% 극단치)로 인해 스케일이 아래로 뭉개지는 것을 방지하는 안전 필터링 적용
    q_99 = df[target_var].quantile(0.99)
    df_filtered = df[df[target_var] <= q_99]
    
    fig_hist = px.histogram(
        df_filtered, 
        x=target_var, 
        nbins=50,
        title=f"{target_var} 데이터 분포 차트 (상위 1% 극단치 제외 버전)",
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# --- 📦 탭 4: 박스 플롯 ---
with tab4:
    st.subheader(f"📦 중재자(Moderator) 유무에 따른 `{target_var}` 범위 비교")
    st.markdown(f"방에 중재자가 참여하고 개입했을 때(`True`)와 없을 때(`False`), 우리가 정한 종속변수 **`{target_var}`**의 중앙값 및 편차 범위가 어떻게 바뀌는지 비교·분석합니다.")
    
    # 탭 3에서 극단치를 제외하고 정의한 동일한 필터링 데이터를 사용해 선명도 극대화
    fig_box = px.box(
        df_filtered, 
        x="moderator", 
        y=target_var, 
        color="moderator",
        points="outliers",  # 정제된 버전의 이상치를 점으로 따로 표시
        title=f"중재자 개입 여부별 {target_var} 박스플롯 분포 비교",
        labels={"moderator": "중재자 개입 여부", target_var: f"종속변수 ({target_var})"}
    )
    st.plotly_chart(fig_box, use_container_width=True)
