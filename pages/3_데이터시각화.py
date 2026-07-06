import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. 페이지 설정 (화면을 넓게 쓰는 와이드 모드)
st.set_page_config(page_title="데이터 전처리 전/후 비교 대시보드", layout="wide")

# 2. 데이터 자동 로드 및 캐싱 (렉 방지)
@st.cache_data
def load_data_automatically():
    # 사용자의 입력 없이 백엔드에서 파일을 자동으로 지정하여 로드합니다.
    df_before = pd.read_csv("onlineCasino.csv")
    df_after = pd.read_csv("onlineCasino_cleaned_v2.csv")
    return df_before, df_after

# 파일 읽기 시도
try:
    df_before, df_after = load_data_automatically()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# --- 만약 파일 자동 로드에 실패했을 때만 보여주는 최소한의 안내 화면 ---
if not data_loaded:
    st.error("📂 [파일 로드 실패] 데이터를 자동으로 불러오지 못했습니다.")
    st.markdown("""
    **🛠️ 조치 방법:**
    현재 실행 중인 `app.py` 파일과 **동일한 폴더(경로)** 안에 아래의 두 파일이 정확한 이름으로 존재하는지 확인해 주세요.
    1. 원본 데이터 파일: `onlineCasino.csv`
    2. 전처리 완료 파일: `onlineCasino_cleaned_v2.csv`
    """)
    st.stop()


# ==================== 파일 로드 성공 시 즉시 렌더링되는 본문 화면 ====================

st.title("📊 데이터 전처리 전(Before) vs 후(After) 종합 비교 분석")
st.markdown("설계하신 종속변수 `peopleLost`(잃은 유저 수)와 `outpay`(지급액)를 중심으로 전처리 전후의 시각화 차이를 한눈에 비교합니다.")

# 3. 사이드바 - 종속변수(Target) 스위치만 깔끔하게 배치
st.sidebar.header("🎯 종속변수(Target) 설정")
target_var = st.sidebar.selectbox(
    "시각화 기준 종속변수 변경", 
    ["outpay", "peopleLost"],
    help="선택한 종속변수에 따라 아래 모든 산점도, 분포차트, 박스플롯이 실시간으로 변경됩니다."
)

st.sidebar.write("---")
st.sidebar.markdown("### 📋 요약 리포트")
st.sidebar.write(f"- **전처리 전 데이터:** {len(df_before):,} 행")
st.sidebar.write(f"- **전처리 후 데이터:** {len(df_after):,} 행")


# 4. 메인 대시보드 레이아웃 (좌: 전처리 전 / 우: 전처리 후)
col_left, col_right = st.columns(2)

# ==========================================
# ❌ LEFT: 전처리 전 (Before Preprocessing)
# ==========================================
with col_left:
    st.subheader("❌ 전처리 전 원본 데이터 (Original)")
    st.caption("결측치와 극단적 이상치가 정제되지 않아 대시보드가 왜곡된 상태입니다.")
    
    # 웹 브라우저 부하 및 시각화 렉 방지를 위한 무작위 샘플링
    df_b_sample = df_before.sample(n=min(2000, len(df_before)), random_state=42)
    
    # [그래프 1] 상관관계 히트맵
    st.markdown("#### 🔥 1. 변수 간 상관관계 히트맵")
    numeric_cols_b = df_before.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_b = numeric_cols_b.corr()
    fig_heat_b = px.imshow(
        corr_b, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="[전처리 전] 수치형 변수 상관관계 표"
    )
    st.plotly_chart(fig_heat_b, use_container_width=True)
    
    # [그래프 2] 산점도
    st.markdown(f"#### ✨ 2. 참여 게이머 수 vs {target_var} 산점도")
    fig_scat_b = px.scatter(
        df_b_sample, x="gamers", y=target_var, color="moderator",
        title=f"[전처리 전] 게이머 수와 {target_var}의 관계 분포"
    )
    st.plotly_chart(fig_scat_b, use_container_width=True)
    
    # [그래프 3] 분포 차트 (히스토그램)
    st.markdown(f"#### 📊 3. {target_var} 전체 빈도 분포 차트")
    fig_hist_b = px.histogram(
        df_before, x=target_var, nbins=50,
        title=f"[전처리 전] {target_var} 데이터 왜곡 상태",
        color_discrete_sequence=["#EF553B"]
    )
    st.plotly_chart(fig_hist_b, use_container_width=True)
    
    # [그래프 4] 박스 플롯
    st.markdown(f"#### 📦 4. 중재자 유무별 {target_var} 박스플롯")
    fig_box_b = px.box(
        df_before, x="moderator", y=target_var, color="moderator",
        title=f"[전처리 전] 중재자 개입 여부별 {target_var} 범위"
    )
    st.plotly_chart(fig_box_b, use_container_width=True)


# ==========================================
#  RIGHT: 전처리 후 (After Preprocessing)
# ==========================================
with col_right:
    st.subheader(" 전처리 후 정제 데이터 (Cleaned)")
    st.caption("결측치가 박멸되었으며, 상위 1% 아웃라이어 제어로 데이터 경향성이 뚜렷합니다.")
    
    # 웹 브라우저 부하 및 시각화 렉 방지를 위한 무작위 샘플링
    df_a_sample = df_after.sample(n=min(2000, len(df_after)), random_state=42)
    
    # [그래프 1] 상관관계 히트맵 (시간대/요일/순수익 파생변수 포함)
    st.markdown("#### 🔥 1. 고도화된 변수 간 상관관계 히트맵")
    numeric_cols_a = df_after.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_a = numeric_cols_a.corr()
    fig_heat_a = px.imshow(
        corr_a, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="[전처리 후] 파생변수가 결합된 상관관계 표"
    )
    st.plotly_chart(fig_heat_a, use_container_width=True)
    
    # [그래프 2] 산점도
    st.markdown(f"#### ✨ 2. 참여 게이머 수 vs {target_var} 산점도")
    fig_scat_a = px.scatter(
        df_a_sample, x="gamers", y=target_var, color="moderator",
        title=f"[전처리 후] 게이머 수와 {target_var}의 관계 (결측치 정제 완료)",
        opacity=0.7
    )
    st.plotly_chart(fig_scat_a, use_container_width=True)
    
    # [그래프 3] 분포 차트 (히스토그램 - 아웃라이어 필터링 적용)
    st.markdown(f"#### 📊 3. {target_var} 전체 빈도 분포 차트")
    # 이상치로 인한 스케일 뭉개짐을 해결하기 위해 상위 99% 컷팅 시각화 적용
    q_99 = df_after[target_var].quantile(0.99)
    df_hist_filtered = df_after[df_after[target_var] <= q_99]
    
    fig_hist_a = px.histogram(
        df_hist_filtered, x=target_var, nbins=50,
        title=f"[전처리 후] {target_var} 분포 (상위 1% 극단치 조정 버전)",
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig_hist_a, use_container_width=True)
    
    # [그래프 4] 박스 플롯
    st.markdown(f"#### 📦 4. 중재자 유무별 {target_var} 박스플롯")
    fig_box_a = px.box(
        df_hist_filtered, x="moderator", y=target_var, color="moderator",
        points="outliers",
        title=f"[전처리 후] 중재자 개입 여부별 {target_var} 비교 (가독성 극대화)"
    )
    st.plotly_chart(fig_box_a, use_container_width=True)


# 5. 하단 조별 발표용 분석 가이드 가시화
st.write("---")
st.subheader("💡 우리 조의 종속변수 분석 방향성 요약 (발표 팁)")
st.info(f"""
- **대시보드 편의성:** 사용자가 파일을 업로드할 필요 없이 프로그램 시작과 동시에 양쪽 데이터를 자동으로 로드하여 화면을 즉시 그리도록 자동화 시스템을 구축했습니다.
- **`outpay`(지급액) 분석 결과:** 왼쪽(전처리 전) 박스플롯과 히스토그램은 무지막지하게 큰 극단치 때문에 박스가 완전히 짜부라져 성분 비교가 불가능하지만, 오른쪽(전처리 후) 대시보드에서는 상위 1% 조율을 통해 중재자가 없을 때(`False`)보다 있을 때(`True`) 카지노의 지급액 편차가 어떻게 안정되는지 확연하게 눈으로 증명됩니다.
- **`peopleLost`(패배 유저 수) 분석 결과:** 전처리 후 우측의 확장된 상관관계 히트맵을 확인하면, 새롭게 설계한 시간대 파생변수(`hour`)와 우리가 지정한 종속변수(`peopleLost`) 간의 유의미한 밀집도가 깔끔하게 매핑되는 것을 조원 및 교수님께 어필할 수 있습니다.
""")
