import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# 1. 페이지 설정
st.set_page_config(page_title="데이터 전처리 전/후 비교 대시보드", layout="wide")

# 2. 파이썬이 app.py의 실제 위치를 스스로 추적하도록 설정 (핵심 경로 수정)
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
before_path = os.path.join(current_dir, "onlineCasino.csv")
after_path = os.path.join(current_dir, "onlineCasino_cleaned_v2.csv")

# 3. 데이터 자동 로드 및 캐싱
@st.cache_data
def load_data_strictly():
    df_before = pd.read_csv(before_path)
    df_after = pd.read_csv(after_path)
    return df_before, df_after

# 파일 로드 시도 및 예외 처리
try:
    df_before, df_after = load_data_strictly()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# --- 만약 여전히 파일을 못 찾을 때만 임시로 작동하는 비상용 파일 업로더 ---
if not data_loaded:
    st.error("📂 [자동 파일 인식 실패] 파이썬 시스템이 폴더 내의 CSV 파일을 놓쳤습니다.")
    st.markdown("🚨 **가장 확실한 임시 해결책:** 아래의 업로드 칸에 가지고 계신 CSV 파일을 순서대로 끌어다(Drag & Drop) 놓아주시면 즉시 대시보드가 정상 출력됩니다!")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        u_before = st.file_uploader("❌ 전처리 전 'onlineCasino.csv' 업로드", type=["csv"])
    with col_up2:
        u_after = st.file_uploader(" 전처리 후 'onlineCasino_cleaned_v2.csv' 업로드", type=["csv"])
        
    if u_before is not None and u_after is not None:
        df_before = pd.read_csv(u_before)
        df_after = pd.read_csv(u_after)
    else:
        st.stop()

# ==================== 대시보드 본문 렌더링 ====================

st.title("📊 데이터 전처리 전(Before) vs 후(After) 종합 비교 분석")
st.markdown("우리가 지정한 핵심 종속변수인 **`peopleLost`(패배 유저 수)**와 **`outpay`(지급액)**의 상관성과 변화 추이를 전처리 전후로 관찰합니다.")

# 4. 사이드바 - 종속변수 컨트롤러
st.sidebar.header("🎯 종속변수(Target) 설정")
target_var = st.sidebar.selectbox(
    "시각화 기준 종속변수 선택", 
    ["outpay", "peopleLost"],
    help="선택한 변수에 따라 산점도, 분포차트, 박스플롯이 동적으로 변화합니다."
)

st.sidebar.write("---")
st.sidebar.markdown(f"### 📋 요약 정보\n- **전처리 전:** {len(df_before):,} 행\n- **전처리 후:** {len(df_after):,} 행")


# 5. 메인 레이아웃 (좌: 전처리 전 / 우: 전처리 후)
col_left, col_right = st.columns(2)

# ==========================================
# ❌ LEFT: 전처리 전 (Before Preprocessing)
# ==========================================
with col_left:
    st.subheader("❌ 전처리 전 원본 데이터 (Original)")
    st.caption("이상치와 결측치가 정제되지 않아 통계 스케일 왜곡이 심합니다.")
    
    df_b_sample = df_before.sample(n=min(2000, len(df_before)), random_state=42)
    
    # [그래프 1] 종속변수 간 상관관계 히트맵
    st.markdown("#### 🔥 1. 변수 간 상관관계 히트맵")
    numeric_cols_b = df_before.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_b = numeric_cols_b.corr()
    fig_heat_b = px.imshow(
        corr_b, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="[전처리 전] 변수 간 상관계수"
    )
    st.plotly_chart(fig_heat_b, use_container_width=True)
    
    # [그래프 2] 산점도 (종속변수 2개의 관계 비교 추가!)
    st.markdown("#### ✨ 2. 종속변수 간의 관계 (peopleLost vs outpay)")
    fig_scat_b = px.scatter(
        df_b_sample, x="peopleLost", y="outpay", color="moderator",
        title="[전처리 전] 패배 유저 수와 카지노 지급액의 관계"
    )
    st.plotly_chart(fig_scat_b, use_container_width=True)
    
    # [그래프 3] 분포 차트
    st.markdown(f"#### 📊 3. 선택한 종속변수({target_var}) 분포")
    fig_hist_b = px.histogram(
        df_before, x=target_var, nbins=50,
        title=f"[전처리 전] {target_var} 빈도 분포", color_discrete_sequence=["#EF553B"]
    )
    st.plotly_chart(fig_hist_b, use_container_width=True)
    
    # [그래프 4] 박스 플롯
    st.markdown(f"#### 📦 4. 중재자 유무별 {target_var} 범위")
    fig_box_b = px.box(
        df_before, x="moderator", y=target_var, color="moderator",
        title=f"[전처리 전] 중재자별 {target_var} 박스플롯"
    )
    st.plotly_chart(fig_box_b, use_container_width=True)


# ==========================================
#  RIGHT: 전처리 후 (After Preprocessing)
# ==========================================
with col_right:
    st.subheader(" 전처리 후 정제 데이터 (Cleaned)")
    st.caption("결측치가 처리되었으며, 상위 1% 극단치 캡핑을 반영해 경향성이 정밀합니다.")
    
    df_a_sample = df_after.sample(n=min(2000, len(df_after)), random_state=42)
    
    # [그래프 1] 종속변수 간 상관관계 히트맵 (파생변수 포함)
    st.markdown("#### 🔥 1. 고도화된 변수 간 상관관계 히트맵")
    numeric_cols_a = df_after.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_a = numeric_cols_a.corr()
    fig_heat_a = px.imshow(
        corr_a, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="[전처리 후] 파생변수 포함 상관계수"
    )
    st.plotly_chart(fig_heat_a, use_container_width=True)
    
    # [그래프 2] 산점도 (종속변수 2개의 관계 비교 - 정제 버전!)
    st.markdown("#### ✨ 2. 종속변수 간의 관계 (peopleLost vs outpay)")
    fig_scat_a = px.scatter(
        df_a_sample, x="peopleLost", y="outpay", color="moderator",
        title="[전처리 후] 패배 유저 수와 카지노 지급액의 관계 (왜곡 제거)",
        opacity=0.7
    )
    st.plotly_chart(fig_scat_a, use_container_width=True)
    
    # [그래프 3] 분포 차트
    st.markdown(f"#### 📊 3. 선택한 종속변수({target_var}) 분포")
    # 종속변수 스케일 가독성을 극대화하기 위해 상위 99% 컷팅 적용
    q_99 = df_after[target_var].quantile(0.99)
    df_hist_filtered = df_after[df_after[target_var] <= q_99]
    
    fig_hist_a = px.histogram(
        df_hist_filtered, x=target_var, nbins=50,
        title=f"[전처리 후] {target_var} 분포 (상위 1% 극단치 조정)", color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig_hist_a, use_container_width=True)
    
    # [그래프 4] 박스 플롯
    st.markdown(f"#### 📦 4. 중재자 유무별 {target_var} 범위")
    fig_box_a = px.box(
        df_hist_filtered, x="moderator", y=target_var, color="moderator", points="outliers",
        title=f"[전처리 후] 중재자별 {target_var} 박스플롯 (시각 왜곡 제거)"
    )
    st.plotly_chart(fig_box_a, use_container_width=True)
