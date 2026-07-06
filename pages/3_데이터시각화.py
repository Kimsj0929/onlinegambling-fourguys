import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# 1. 페이지 설정
st.set_page_config(page_title="데이터 전처리 전/후 비교 대시보드", layout="wide")

# 2. [🔥 핵심 수정] 파이썬이 내 컴퓨터 내부를 뒤져서 자동으로 파일을 찾아내는 함수
@st.cache_data
def super_auto_load():
    file1 = "onlineCasino.csv"
    file2 = "onlineCasino_cleaned_v2.csv"
    
    # 시스템이 탐색할 후보 경로들 목록 (현재폴더, 상위폴더, 실행폴더 등)
    possible_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd(),
        os.path.join(os.getcwd(), '..'),
    ]
    
    df_b, df_a = None, None
    
    # 1단계: 주변 폴더 자동 탐색
    for d in possible_dirs:
        p1 = os.path.join(d, file1)
        p2 = os.path.join(d, file2)
        if os.path.exists(p1) and df_b is None:
            df_b = pd.read_csv(p1)
        if os.path.exists(p2) and df_a is None:
            df_a = pd.read_csv(p2)
            
    # 2단계: 만약 특정 환경 때문에 위에서 못 찾았을 경우, 같은 폴더 내 리스트 전체 매칭 (방어적 코드)
    if df_b is None or df_a is None:
        for root, dirs, files in os.walk(os.getcwd()):
            if file1 in files and df_b is None:
                df_b = pd.read_csv(os.path.join(root, file1))
            if file2 in files and df_a is None:
                df_a = pd.read_csv(os.path.join(root, file2))
            if df_b is not None and df_a is not None:
                break
                
    # 둘 다 찾았다면 반환, 하나라도 없으면 에러 발생시킴
    if df_b is not None and df_a is not None:
        return df_b, df_a
    else:
        raise FileNotFoundError("두 파일 중 하나를 찾을 수 없습니다.")

# 파일 로드 실행
try:
    df_before, df_after = super_auto_load()
    data_loaded = True
except Exception:
    data_loaded = False

# --- [비상 상황] 컴퓨터 환경이 완전 차단되어 자동 인식이 절대 불가능할 때만 뜨는 백업 자동화창 ---
if not data_loaded:
    st.error("📂 [시스템 안내] 파일 자동 인식 경로가 막혔습니다.")
    st.markdown("💡 **1초 해결법:** 이 메시지가 보인다면 아래 파일 이름을 다시 한번만 체크하시거나, 아래 칸에 파일을 한 번만 넣어주세요. 이후에는 자동으로 세션이 유지됩니다.")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        u_before = st.file_uploader("❌ onlineCasino.csv 넣어주기", type=["csv"], key="b_up")
    with col_u2:
        u_after = st.file_uploader(" onlineCasino_cleaned_v2.csv 넣어주기", type=["csv"], key="a_up")
        
    if u_before is not None and u_after is not None:
        df_before = pd.read_csv(u_before)
        df_after = pd.read_csv(u_after)
    else:
        st.stop()

# ==================== 대시보드 본문 즉시 출력 화면 ====================

st.title("📊 데이터 전처리 전(Before) vs 후(After) 종합 비교 분석")
st.markdown("종속변수인 **`peopleLost`(패배 유저 수)**와 **`outpay`(지급액)**를 축으로 전처리 효과를 시각적으로 즉시 확인합니다.")

# 3. 사이드바 컨트롤러 (종속변수 타겟 마스터 스위치)
st.sidebar.header("🎯 종속변수(Target) 설정")
target_var = st.sidebar.selectbox(
    "시각화 기준 종속변수 선택", 
    ["outpay", "peopleLost"],
    help="선택한 변수에 따라 산점도, 분포차트, 박스플롯이 실시간 동성 변화합니다."
)

st.sidebar.write("---")
st.sidebar.markdown(f"### 📋 요약 정보\n- **전처리 전:** {len(df_before):,} 행\n- **전처리 후:** {len(df_after):,} 행")

# 4. 메인 화면 레이아웃 (좌 2단 배치)
col_left, col_right = st.columns(2)

# ==========================================
# ❌ LEFT: 전처리 전 (Before Preprocessing)
# ==========================================
with col_left:
    st.subheader("❌ 전처리 전 원본 데이터")
    st.caption("결측치와 극단치 아웃라이어가 섞여 스케일이 왜곡된 상태입니다.")
    
    df_b_sample = df_before.sample(n=min(2000, len(df_before)), random_state=42)
    
    st.markdown("#### 🔥 1. 변수 간 상관관계 히트맵")
    numeric_cols_b = df_before.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    fig_heat_b = px.imshow(numeric_cols_b.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig_heat_b, use_container_width=True)
    
    st.markdown("#### ✨ 2. 종속변수 간의 관계 (peopleLost vs outpay)")
    fig_scat_b = px.scatter(df_b_sample, x="peopleLost", y="outpay", color="moderator", title="[전처리 전] 관계 산점도")
    st.plotly_chart(fig_scat_b, use_container_width=True)
    
    st.markdown(f"#### 📊 3. 선택한 종속변수({target_var}) 분포")
    fig_hist_b = px.histogram(df_before, x=target_var, nbins=50, color_discrete_sequence=["#EF553B"])
    st.plotly_chart(fig_hist_b, use_container_width=True)
    
    st.markdown(f"#### 📦 4. 중재자 유무별 {target_var} 범위")
    fig_box_b = px.box(df_before, x="moderator", y=target_var, color="moderator")
    st.plotly_chart(fig_box_b, use_container_width=True)


# ==========================================
#  RIGHT: 전처리 후 (After Preprocessing)
# ==========================================
with col_right:
    st.subheader(" 전처리 후 정제 데이터")
    st.caption("상위 1% 아웃라이어 조율 및 인덱스 정렬이 완료되어 데이터가 깔끔합니다.")
    
    df_a_sample = df_after.sample(n=min(2000, len(df_after)), random_state=42)
    
    st.markdown("#### 🔥 1. 고도화된 변수 간 상관관계 히트맵")
    numeric_cols_a = df_after.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    fig_heat_a = px.imshow(numeric_cols_a.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig_heat_a, use_container_width=True)
    
    st.markdown("#### ✨ 2. 종속변수 간의 관계 (peopleLost vs outpay)")
    fig_scat_a = px.scatter(df_a_sample, x="peopleLost", y="outpay", color="moderator", title="[전처리 후] 관계 산점도 (이상치 제거)", opacity=0.7)
    st.plotly_chart(fig_scat_a, use_container_width=True)
    
    st.markdown(f"#### 📊 3. 선택한 종속변수({target_var}) 분포")
    q_99 = df_after[target_var].quantile(0.99)
    df_hist_filtered = df_after[df_after[target_var] <= q_99]
    fig_hist_a = px.histogram(df_hist_filtered, x=target_var, nbins=50, color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig_hist_a, use_container_width=True)
    
    st.markdown(f"#### 📦 4. 중재자 유무별 {target_var} 범위")
    fig_box_a = px.box(df_hist_filtered, x="moderator", y=target_var, color="moderator", points="outliers")
    st.plotly_chart(fig_box_a, use_container_width=True)
