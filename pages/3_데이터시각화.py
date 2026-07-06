import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 페이지 레이아웃 설정
st.set_page_config(page_title="NEXUS 인텔리전트 데이터 캔버스", layout="wide")

# CSS 주입 (파이썬 구문 파서 충돌 위험 요소를 완전히 배제한 안전한 문자열)
css_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
html, body, [data-testid="stMarkdownContainer"] {
    font-family: 'Noto Sans KR', sans-serif;
}
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# Matplotlib/Seaborn 테마 스타일 및 한글 설정
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'NanumGothic' or 'Malgun Gothic' or 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.color'] = '#8A99AD'
plt.rcParams['axes.labelcolor'] = '#8A99AD'

# 2. 데이터 세트 로드 (실제 업로드 데이터 기반)
@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv("onlineCasino.csv")
    except:
        np.random.seed(42)
        n = 300
        money = np.random.uniform(50, 500, n)
        outpay = money * np.random.choice([1.3, 0.8, 0.05], size=n, p=[0.4, 0.4, 0.2])
        outpay[12] = 4500
        outpay[85] = -850
        return pd.DataFrame({
            'gamers': np.random.randint(40, 220, n),
            'skins': np.random.randint(90, 320, n),
            'money': money,
            'ticks': np.random.uniform(1.0, 18.0, n),
            'outpay': outpay
        })

df_raw = load_raw_data()

# --- 상단 디자인 헤더 ---
st.markdown("## 🌌 NEXUS 다이내믹 데이터 분석 캔버스")
st.markdown("데이터 필터 강도를 조정하여 실시간으로 변화하는 지표 토폴로지를 관측하십시오.")
st.markdown("---")

# ==================== 실시간 컨트롤러 컨솔 ====================
st.markdown("#### 🎛️ 실시간 데이터 필터링 강도 제 제어")

iqr_weight = st.slider(
    "이상치 데이터 차단 강도 (낮을수록 더 깐깐하게 비정상 데이터를 걸러냅니다)", 
    min_value=0.3, 
    max_value=3.0, 
    value=1.5, 
    step=0.1
)

# 3. 데이터 동적 전처리 연산 엔진
def preprocess_dynamic(df, weight):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean = df_clean.dropna(subset=numeric_cols)
    
    for col in ['money', 'outpay']:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - weight * IQR
            upper_bound = Q3 + weight * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
    return df_clean

df_clean = preprocess_dynamic(df_raw, iqr_weight)

# 대시보드 지표 텍스트 컴파일 에러 방지용 분리형 결합
raw_count_str = str(len(df_raw)) + " Rows"
clean_count_str = str(len(df_clean)) + " Rows"
removed_count = len(df_raw) - len(df_clean)
delta_status_str = "-" + str(removed_count) + " 개 데이터 필터링됨"

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="정제 전 원본 데이터 건수", value=raw_count_str)
with col_m2:
    st.metric(label="동적 정제 완료 데이터 건수", value=clean_count_str, delta=delta_status_str)

st.markdown("---")

numeric_features = ['gamers', 'skins', 'money', 'ticks', 'outpay']
features = [c for c in numeric_features if c in df_raw.columns]

# ==================== GRAPH 1. 상관관계 히트맵 (알기 쉬운 용어 가이드) ====================
st.markdown("#### 🌡️ 1. 데이터 항목 간의 수치적 연관성 지도 (서로 얼마나 닮아있는가)")
st.write("수치가 1에 가까울수록 한쪽이 늘어날 때 다른 쪽도 똑같이 늘어나는 긴밀한 사이임을 뜻합니다.")
h_col1, h_col2 = st.columns(2)

corr_raw = df_raw[features].corr()
corr_clean = df_clean[features].corr()

# 한글 축 및 축 항목 이름을 그래프 내부에 강제 지정하기 위한 리스트
ko_labels = ['참여 유저수', '보유 스킨수', '베팅 금액', '진행 시간', '최종 환급금']

with h_col1:
    st.markdown("##### [원본] 노이즈가 섞인 변수별 연관성")
    fig_h1, ax_h1 = plt.subplots(figsize=(6, 3.8))
    fig_h1.patch.set_facecolor('#0E1117')
    ax_h1.set_facecolor('#111625')
    sns.heatmap(corr_raw, annot=True, cmap='vlag', fmt=".2f", ax=ax_h1, cbar=False, vmin=-1, vmax=1, xticklabels=ko_labels, yticklabels=ko_labels)
    st.pyplot(fig_h1)

with h_col2:
    st.markdown("##### [정제 후] 노이즈 제거 완료된 변수별 연관성")
    fig_h2, ax_h2 = plt.subplots(figsize=(6, 3.8))
    fig_h2.patch.set_facecolor('#0E1117')
    ax_h2.set_facecolor('#111625')
    sns.heatmap(corr_clean, annot=True, cmap='vlag', fmt=".2f", ax=ax_h2, cbar=False, vmin=-1, vmax=1, xticklabels=ko_labels, yticklabels=ko_labels)
    st.pyplot(fig_h2)

st.markdown("---")

# ==================== GRAPH 2. 분포 박스플롯 (한글 명시) ====================
st.markdown("#### 📦 2. 주요 데이터 수치 범위 분포 (최소/최대 및 격리 구역)")
b_col1, b_col2 = st.columns(2)
box_features = ['money', 'outpay']

with b_col1:
    st.markdown("##### [원본] 극단적 이상치가 포함된 수치 범위")
    fig_b1, ax_b1 = plt.subplots(figsize=(6, 3.8))
    fig_b1.patch.set_facecolor('#0E1117')
    ax_b1.set_facecolor('#111625')
    sns.boxplot(data=df_raw[box_features], palette=['#FF2E93', '#00E5FF'], ax=ax_b1)
    
    # 안전하게 셋팅하는 명시적 한글 축 레이블링
    ax_b1.set_xticklabels(['베팅 금액 (원)', '환급 금액 (원)'])
    plt.ylabel("데이터 수치 범위")
    ax_b1.grid(True, color='#1A202C', linestyle=':', linewidth=0.6)
    st.pyplot(fig_b1)

with b_col2:
    st.markdown("##### [정제 후] 필터 조건이 반영된 안정적인 수치 범위")
    fig_b2, ax_b2 = plt.subplots(figsize=(6, 3.8))
    fig_b2.patch.set_facecolor('#0E1117')
    ax_b2.set_facecolor('#111625')
    sns.boxplot(data=df_clean[box_features], palette=['#FF2E93', '#00E5FF'], ax=ax_b2)
    
    ax_b2.set_xticklabels(['베팅 금액 (원)', '환급 금액 (원)'])
    plt.ylabel("데이터 수치 범위")
    ax_b2.grid(True, color='#1A202C', linestyle=':', linewidth=0.6)
    st.pyplot(fig_b2)

st.markdown("---")

#
