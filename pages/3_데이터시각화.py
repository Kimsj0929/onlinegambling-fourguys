import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. 페이지 설정
st.set_page_config(page_title="데이터 전처리 전/후 비교 대시보드", layout="wide")

st.title("📊 데이터 전처리 전(Before) vs 후(After) 종합 비교 분석")
st.markdown("설계하신 종속변수 `peopleLost`(잃은 유저 수)와 `outpay`(지급액)를 중심으로 전처리 효과를 분석합니다.")

# 2. 데이터 로드 및 캐싱
@st.cache_data
def load_data():
    # 전처리 전 원본 데이터와 전처리 후 데이터를 모두 불러옵니다.
    df_before = pd.read_csv("onlineCasino.csv")
    df_after = pd.read_csv("onlineCasino_cleaned_v2.csv")
    return df_before, df_after

try:
    df_before, df_after = load_data()
except FileNotFoundError:
    st.error("📂 파일 로드 실패! 'onlineCasino.csv'와 'onlineCasino_cleaned_v2.csv' 파일이 같은 경로에 있는지 확인해 주세요.")
    st.stop()


# 3. 사이드바 (Sidebar) - 종속변수 타겟 선택 및 요약 정보
st.sidebar.header("🎯 종속변수(Target) 설정")
target_var = st.sidebar.selectbox("시각화 기준 종속변수 선택", ["outpay", "peopleLost"])
st.sidebar.markdown(f"현재 선택된 종속변수: **`{target_var}`**")

st.sidebar.write("---")
st.sidebar.markdown("### 📋 데이터 크기 비교")
st.sidebar.write(f"- ❌ **전처리 전:** {len(df_before):,} 행 (time 결측치 10건 포함)")
st.sidebar.write(f"-  **전처리 후:** {len(df_after):,} 행 (결측치 완전 박멸)")


# 4. 메인 화면 - 전처리 전 / 후로 레이아웃 분할
# 사용자가 한 화면에서 직관적으로 비교할 수 있도록 왼쪽-오른쪽 2단 컬럼 구성을 사용합니다.
col_left, col_right = st.columns(2)

# ==========================================
# ❌ LEFT: 전처리 전 (Before Preprocessing)
# ==========================================
with col_left:
    st.header("❌ 전처리 전 데이터 (Original)")
    st.caption("이상치와 결측치가 정제되지 않아 그래프 왜곡이 발생할 수 있습니다.")
    
    # 전처리 전의 대용량 렉 방지 샘플링
    df_b_sample = df_before.sample(n=min(2000, len(df_before)), random_state=42)
    
    # [그래프 1] 상관관계 히트맵
    st.subheader("🔥 1. 변수 간 상관관계 히트맵")
    # 종속변수들과 수치형 변수들 간의 관계 분석
    numeric_cols_b = df_before.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_b = numeric_cols_b.corr()
    fig_heat_b = px.imshow(
        corr_b, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="전처리 전 수치형 변수 상관관계"
    )
    st.plotly_chart(fig_heat_b, use_container_width=True)
    
    # [그래프 2] 산점도 (독립변수 gamers vs 선택한 종속변수)
    st.subheader(f"✨ 2. 참여 게이머 수 vs 종속변수({target_var}) 산점도")
    fig_scat_b = px.scatter(
        df_b_sample, x="gamers", y=target_var, color="moderator",
        title=f"[전처리 전] 게이머 수와 {target_var}의 관계",
        labels={"gamers": "참여 게이머 수"}
    )
    st.plotly_chart(fig_scat_b, use_container_width=True)
    
    # [그래프 3] 분포 차트 (종속변수 히스토그램)
    st.subheader(f"📊 3. 종속변수({target_var}) 분포 차트")
    fig_hist_b = px.histogram(
        df_before, x=target_var, nbins=50,
        title=f"[전처리 전] {target_var} 전체 데이터 빈도 분포",
        color_discrete_sequence=["#EF553B"]
    )
    st.plotly_chart(fig_hist_b, use_container_width=True)
    
    # [그래프 4] 박스 플롯 (중재자 유무에 따른 종속변수 비교)
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
    st.caption("파생변수가 추가되었으며, 상위 1% 아웃라이어 조정을 통해 데이터 경향성이 뚜렷합니다.")
    
    # 전처리 후의 대용량 렉 방지 샘플링
    df_a_sample = df_after.sample(n=min(2000, len(df_after)), random_state=42)
    
    # [그래프 1] 상관관계 히트맵 (파생변수가 대거 추가되어 더 풍부한 인사이트)
    st.subheader("🔥 1. 고도화된 변수 간 상관관계 히트맵")
    numeric_cols_a = df_after.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
    corr_a = numeric_cols_a.corr()
    fig_heat_a = px.imshow(
        corr_a, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="전처리 후 수치형 변수 상관관계 (파생변수 포함)"
    )
    st.plotly_chart(fig_heat_a, use_container_width=True)
    
    # [그래프 2] 산점도 (독립변수 gamers vs 선택한 종속변수 - 정제 버전)
    st.subheader(f"✨ 2. 참여 게이머 수 vs 종속변수({target_var}) 산점도")
    fig_scat_a = px.scatter(
        df_a_sample, x="gamers", y=target_var, color="moderator",
        title=f"[전처리 후] 게이머 수와 {target_var}의 관계 (아웃라이어 제어)",
        labels={"gamers": "참여 게이머 수"},
        opacity=0.7
    )
    st.plotly_chart(fig_scat_a, use_container_width=True)
    
    # [그래프 3] 분포 차트 (종속변수 히스토그램 - 안정된 스케일)
    st.subheader(f"📊 3. 종속변수({target_var}) 분포 차트")
    # 극단치 스케일 왜곡 방지를 위해 상위 99% 컷팅 후 시각화
    q_99 = df_after[target_var].quantile(0.99)
    df_hist_filtered = df_after[df_after[target_var] <= q_99]
    
    fig_hist_a = px.histogram(
        df_hist_filtered, x=target_var, nbins=50,
        title=f"[전처리 후] {target_var} 분포 (상위 1% 극단치 제외 버전)",
        color_discrete_sequence=["#636EFA"]
    )
    st.plotly_chart(fig_hist_a, use_container_width=True)
    
    # [그래프 4] 박스 플롯 (중재자 유무에 따른 종속변수 비교 - 시각적 왜곡 제거)
    st.subheader(f"📦 4. 중재자 유무별 종속변수({target_var}) 박스플롯")
    fig_box_a = px.box(
        df_hist_filtered, x="moderator", y=target_var, color="moderator",
        points="outliers",
        title=f"[전처리 후] 중재자 개입 여부별 {target_var} 분포 (비교 가독성 극대화)"
    )
    st.plotly_chart(fig_box_a, use_container_width=True)


# 5. 하단 종속변수 모델링 가이드 섹션 (조별 발표용)
st.write("---")
st.subheader("💡 종속변수(`outpay`, `peopleLost`) 관점의 전처리 차이점 요약 (발표 팁)")
st.info(f"""
- **분석의 흐름:** 좌측(전처리 전) 그래프들은 극단적인 몇몇 최고 배당률(`ticks`)과 총판돈(`money`)의 이상치 때문에 데이터 분포가 아래쪽으로 뭉개져 스케일 왜곡이 심합니다.
- **`outpay` (플레이어 지급액) 관점:** 전처리 후 오른쪽 분포차트와 박스플롯을 보면, 몇몇 이상치를 적절히 조율함으로써 중재자(`moderator`) 유무에 따라 카지노가 유저에게 주는 배당금 분포 격차가 어떻게 변하는지 왜곡 없이 깨끗하게 관찰됩니다.
- **`peopleLost` (패배 유저 수) 관점:** 전처리 후 히트맵을 확인하시면, 새로 추가된 시간대별 파생변수(`hour`)와 결합했을 때 특정 피크타임에 패배하는 유저 수(`peopleLost`)의 밀도와 상관성이 더욱 뚜렷하게 증명됩니다.
""")
