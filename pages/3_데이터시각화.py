import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. 페이지 설정
st.set_page_config(page_title="데이터 전처리 전/후 비교 대시보드", layout="wide")

st.title("📊 데이터 전처리 전(Before) vs 후(After) 종합 비교 분석")
st.markdown("설계하신 종속변수 `peopleLost`(잃은 유저 수)와 `outpay`(지급액)를 중심으로 전처리 효과를 분석합니다.")

# 2. 사이드바 구성 및 데이터 로드 가이드
st.sidebar.header("📂 1. 데이터셋 불러오기 설정")
st.sidebar.markdown("""
기본적으로 같은 폴더 내의 `onlineCasino.csv`와 `onlineCasino_cleaned_v2.csv` 파일을 자동으로 탐색합니다.
만약 파일 로드 실패 오류가 발생하면 아래 **업로드 기능**을 이용해 주세요!
""")

# 파일 업로더 제공 (서버에 파일이 없을 때를 대비한 수동 업로드 장치)
uploaded_before = st.sidebar.file_uploader("❌ 전처리 전 CSV 파일 업로드 (선택)", type=["csv"])
uploaded_after = st.sidebar.file_uploader(" 전처리 후 CSV 파일 업로드 (선택)", type=["csv"])

# 3. 데이터 로드 로직 (자동 탐색 및 업로드 연동)
df_before = None
df_after = None

# [전처리 전 데이터 로드]
if uploaded_before is not None:
    df_before = pd.read_csv(uploaded_before)
else:
    try:
        df_before = pd.read_csv("onlineCasino.csv")
    except FileNotFoundError:
        pass

# [전처리 후 데이터 로드]
if uploaded_after is not None:
    df_after = pd.read_csv(uploaded_after)
else:
    try:
        df_after = pd.read_csv("onlineCasino_cleaned_v2.csv")
    except FileNotFoundError:
        pass

# 두 파일 중 하나라도 로드되지 않았을 때의 예외 처리 화면
if df_before is None or df_after is None:
    st.warning("⚠️ 데이터를 로드하는 중입니다. 아래 안내 사항을 확인해 주세요.")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        if df_before is None:
            st.error("❌ **전처리 전 데이터(`onlineCasino.csv`)**를 찾을 수 없습니다.")
            st.markdown("- **해결책 1:** 이 `app.py` 파일과 **같은 폴더**에 `onlineCasino.csv` 파일이 있는지 확인해 주세요.")
            st.markdown("- **해결책 2:** 왼쪽 사이드바의 **'전처리 전 CSV 파일 업로드'** 버튼을 눌러 파일을 직접 넣어주세요.")
        else:
            st.success("❌ 전처리 전 데이터 로드 성공!")
            
    with col_info2:
        if df_after is None:
            st.error(" **전처리 후 데이터(`onlineCasino_cleaned_v2.csv`)**를 찾을 수 없습니다.")
            st.markdown("- **해결책 1:** 이 `app.py` 파일과 **같은 폴더**에 `onlineCasino_cleaned_v2.csv` 파일이 있는지 확인해 주세요.")
            st.markdown("- **해결책 2:** 왼쪽 사이드바의 **'전처리 후 CSV 파일 업로드'** 버튼을 눌러 파일을 직접 넣어주세요.")
    st.stop()  # 파일이 다 준비될 때까지 아래 시각화 코드는 실행하지 않고 멈춤


# ==================== 데이터 로드 완료 후 시각화 시작 ====================

# 4. 사이드바 - 종속변수 타겟 설정
st.sidebar.write("---")
st.sidebar.header("🎯 2. 종속변수(Target) 설정")
target_var = st.sidebar.selectbox("시각화 기준 종속변수 선택", ["outpay", "peopleLost"])

st.sidebar.write("---")
st.sidebar.markdown("### 📋 현재 데이터 구조 요약")
st.sidebar.write(f"- ❌ **전처리 전:** {len(df_before):,} 행")
st.sidebar.write(f"-  **전처리 후:** {len(df_after):,} 행")


# 5. 메인 화면 레이아웃 분할 (좌: 전처리 전 / 우: 전처리 후)
col_left, col_right = st.columns(2)

# ==========================================
# ❌ LEFT: 전처리 전 (Before Preprocessing)
# ==========================================
with col_left:
    st.header("❌ 전처리 전 데이터 (Original)")
    st.caption("이상치와 결측치가 정제되지 않아 그래프 왜곡이 발생할 수 있습니다.")
    
    # 렉 방지용 샘플링
    df_b_sample = df_before.sample(n=min(2000, len(df_before)), random_state=42)
    
    # [그래프 1] 상관관계 히트맵
    st.subheader("🔥 1. 변수 간 상관관계 히트맵")
    numeric_cols_b = df_before.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_b = numeric_cols_b.corr()
    fig_heat_b = px.imshow(
        corr_b, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="전처리 전 수치형 변수 상관관계"
    )
    st.plotly_chart(fig_heat_b, use_container_width=True)
    
    # [그래프 2] 산점도 (gamers vs 종속변수)
    st.subheader(f"✨ 2. 참여 게이머 수 vs 종속변수({target_var}) 산점도")
    fig_scat_b = px.scatter(
        df_b_sample, x="gamers", y=target_var, color="moderator",
        title=f"[전처리 전] 게이머 수와 {target_var}의 관계"
    )
    st.plotly_chart(fig_scat_b, use_container_width=True)
    
    # [그래프 3] 분포 차트 (히스토그램)
    st.subheader(f"📊 3. 종속변수({target_var}) 분포 차트")
    fig_hist_b = px.histogram(
        df_before, x=target_var, nbins=50,
        title=f"[전처리 전] {target_var} 전체 빈도 분포",
        color_discrete_sequence=["#EF553B"]
    )
    st.plotly_chart(fig_hist_b, use_container_width=True)
    
    # [그래프 4] 박스 플롯 (중재자별 비교)
    st.subheader(f"📦 4. 중재자 유무별 종속변수({target_var}) 박스플롯")
    fig_box_b = px.box(
        df_before, x="moderator", y=target_var, color="moderator",
        title=f"[전처리 전] 중재자 개입 여부별 {target_var} 분포"
    )
    st.plotly_chart(fig_box_b, use_container_width=True)


# ==========================================
#  RIGHT: 전처리 후 (After Preprocessing)
# ==========================================
with col_right:
    st.header(" 전처리 후 데이터 (Cleaned)")
    st.caption("결측치가 처리되었으며, 이상치 제어로 데이터 경향성이 투렷합니다.")
    
    # 렉 방지용 샘플링
    df_a_sample = df_after.sample(n=min(2000, len(df_after)), random_state=42)
    
    # [그래프 1] 상관관계 히트맵 (파생변수 포함)
    st.subheader("🔥 1. 고도화된 변수 간 상관관계 히트맵")
    numeric_cols_a = df_after.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_a = numeric_cols_a.corr()
    fig_heat_a = px.imshow(
        corr_a, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="전처리 후 수치형 변수 상관관계 (파생변수 포함)"
    )
    st.plotly_chart(fig_heat_a, use_container_width=True)
    
    # [그래프 2] 산점도 (gamers vs 종속변수 - 정제 버전)
    st.subheader(f"✨ 2. 참여 게이머 수 vs 종속변수({target_var}) 산점도")
    fig_scat_a = px.scatter(
        df_a_sample, x="gamers", y=target_var, color="moderator",
        title=f"[전처리 후] 게이머 수와 {target_var}의 관계 (결측치 제거 완료)",
        opacity=0.7
    )
    st.plotly_chart(fig_scat_a, use_container_width=True)
    
    # [그래프 3] 분포 차트 (종속변수 히스토그램 - 극단치 조정)
    st.subheader(f"📊 3. 종속변수({target_var}) 분포 차트")
    # 이상치 스케일 뭉개짐 방지를 위해 상위 99% 선에서 컷팅하여 가독성 증대
    q_99 = df_after[target_var].quantile(0.99)
    df_hist_filtered = df_after[df_after[target_var] <= q_99]
    
    fig_hist_a = px.histogram(
        df_hist_filtered, x=target_var, nbins=50,
        title=f"[전처리 후] {target_var} 분포 (상위 1% 극단치 제외 버전)",
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig_hist_a, use_container_width=True)
    
    # [그래프 4] 박스 플롯 (중재자별 비교 - 시각적 시인성 극대화)
    st.subheader(f"📦 4. 중재자 유무별 종속변수({target_var}) 박스플롯")
    fig_box_a = px.box(
        df_hist_filtered, x="moderator", y=target_var, color="moderator",
        points="outliers",
        title=f"[전처리 후] 중재자 개입 여부별 {target_var} 분포 (시각 왜곡 제거)"
    )
    st.plotly_chart(fig_box_a, use_container_width=True)


# 6. 하단 조별 발표 및 인사이트 요약 가이드
st.write("---")
st.subheader("💡 종속변수(`outpay`, `peopleLost`) 관점의 전처리 효과 가이드")
st.info(f"""
- **파일 로드 안전성 확보:** 현재 페이지는 파일 경로 에러가 발생하더라도 사이드바에서 직접 수동 업로드를 통해 정상 작동할 수 있도록 복구 가이드 레이아웃이 적용되었습니다.
- **`outpay` (플레이어 지급액) 결과 해석:** 전처리 전(왼쪽) 분포 및 박스플롯은 대형 아웃라이어로 인해 찌그러져 있으나, 전처리 후(오른쪽)에는 중재자(`moderator`) 유무에 따른 지급 분포 격차가 뚜렷하게 가시화됩니다.
- **`peopleLost` (패배 유저 수) 결과 해석:** 전처리 후 우측 히트맵을 보면 새로 추가된 시간대별 파생변수(`hour`)와 결합되어 특정 시간에 패배하는 유저 수(`peopleLost`)의 군집 현상을 정밀하게 추적할 수 있습니다.
""")
