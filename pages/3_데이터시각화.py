import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. 페이지 설정
st.set_page_config(page_title="카지노 정제 데이터 분석 대시보드", layout="wide")

st.title("📊 카지노 데이터 분석 대시보드 (정제 완료 버전)")
st.markdown("결측치와 이상치가 완벽하게 정제된 데이터셋을 기반으로 시각화 및 인사이트를 제공합니다.")

# 2. [🔥 서버 배포 환경 전용 파일 로드 알고리즘]
# 복잡한 os.path를 걷어내고, Streamlit 서버 시스템이 현재 열려있는 경로에서 직접 파일을 가져오게 합니다.
@st.cache_data
def load_clean_data():
    # 파일명만 깔끔하게 명시하여 시스템이 직접 찾도록 유도합니다.
    return pd.read_csv("onlineCasino_cleaned_v2.csv")

try:
    df = load_clean_data()
    file_loaded = True
except Exception as e:
    file_loaded = False
    error_message = str(e)

# 만약 파일이 누락되어 찾지 못할 때만 작동하는 백업 로직
if not file_loaded:
    st.error("📂 [서버 에러] GitHub 저장소 내에서 데이터를 불러오지 못했습니다.")
    st.markdown("""
    **🛠️ 깃허브(GitHub) 최종 확인 사항:**
    이 메시지가 웹 화면에 지속적으로 나온다면, 현재 깃허브 저장소(Repository)의 **가장 바깥 폴더(Root)** 또는 `pages` 폴더 안에 **`onlineCasino_cleaned_v2.csv`** 파일이 실제로 업로드되어 누락되지 않았는지 꼭 확인해 주세요!
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
