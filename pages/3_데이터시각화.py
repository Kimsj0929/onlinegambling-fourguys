import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 페이지 레이아웃 설정
st.set_page_config(page_title="NEXUS 데이터 시각화 파이프라인", layout="wide")

# CSS 주입 (파이썬 구문 파서 충돌 위험 요소를 모두 배제한 일반 문자열)
css_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
html, body, [data-testid="stMarkdownContainer"] {
    font-family: 'Noto Sans KR', sans-serif;
}
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# Matplotlib/Seaborn 한글 깨짐 방지 및 다크 테마 커스텀
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'NanumGothic' or 'Malgun Gothic' or 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.color'] = '#8A99AD'
plt.rcParams['axes.labelcolor'] = '#8A99AD'

# 2. 데이터 세트 로드
@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv("onlineCasino.csv")
    except:
        np.random.seed(42)
        n = 250
        money = np.random.uniform(50, 500, n)
        outpay = money * np.random.choice([1.2, 0.8, 0.1], size=n, p=[0.4, 0.4, 0.2])
        outpay[10] = 5000 
        outpay[50] = -999
        return pd.DataFrame({
            'gamers': np.random.randint(50, 200, n),
            'skins': np.random.randint(100, 300, n),
            'money': money,
            'ticks': np.random.uniform(1.0, 15.0, n),
            'outpay': outpay
        })

df_raw = load_raw_data()

# 3. 데이터 전처리 파이프라인 엔진 구동
def preprocess_data(df):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean = df_clean.dropna(subset=numeric_cols)
    
    for col in ['money', 'outpay']:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
    return df_clean

df_clean = preprocess_data(df_raw)

# --- 데이터 사전 연산 파트 (f-string 내부 구문 에러 방지를 위해 변수 분리) ---
raw_len_str = str(len(df_raw)) + " 개"
clean_len_str = str(len(df_clean)) + " 개"
delta_val = len(df_clean) - len(df_raw)
delta_str = str(delta_val) + "개 변동"

# --- 상단 메인 대시보드 헤더 ---
st.markdown("## 📊 데이터 전처리 아키텍처 및 핵심 그래픽스 비교")
st.markdown("구조 정제 전 원본(Raw) 데이터와 IQR 이상치 처리가 완료된 정제(Clean) 데이터의 위상 변화 분석")

# 요약 지표 카드
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="원본 데이터프레임 행(Rows) 수", value=raw_len_str)
with col_m2:
    st.metric(label="이상치 제거 후 정제 데이터 행 수", value=clean_len_str, delta=delta_str)

st.markdown("---")

# ==================== 1. 히트맵 (Correlation Heatmap) ====================
st.markdown("### 🌡️ 1. 변수 간 상관관계 히트맵 분석")
st.write("각 수치형 데이터가 서로 어떤 연관 관계를 갖는지 보여줍니다. 전처리 후 노이즈가 제거되며 핵심 수치들의 상관성이 더욱 뚜렷해집니다.")

h_col1, h_col2 = st.columns(2)
numeric_features = ['gamers', 'skins', 'money', 'ticks', 'outpay']
features = [c for c in numeric_features if c in df_raw.columns]

# 히트맵 매트릭스 사전 계산
corr_raw = df_raw[features].corr()
corr_clean = df_clean[features].corr()

with h_col1:
    st.markdown("#### [원본] 피어슨 상관계수 행렬")
    fig_h1, ax_h1 = plt.subplots(figsize=(6, 4))
    fig_h1.patch.set_facecolor('#0E1117')
    ax_h1.set_facecolor('#111625')
    sns.heatmap(corr_raw, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_h1, cbar=False)
    st.pyplot(fig_h1)

with h_col2:
    st.markdown("#### [전처리 후] 피어슨 상관계수 행렬")
    fig_h2, ax_h2 = plt.subplots(figsize=(6, 4))
    fig_h2.patch.set_facecolor('#0E1117')
    ax_h2.set_facecolor('#111625')
    sns.heatmap(corr_clean, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_h2, cbar=False)
    st.pyplot(fig_h2)

st.markdown("---")

# ==================== 2. 박스플롯 (Box Plot) ====================
st.markdown("### 📦 2. 데이터 분포 특성 및 이상치 식별 박스플롯")
st.write("최솟값, 제1사분위수, 중앙값, 제3사분위수, 최댓값을 표현합니다. 원본의 극단적 이상치 플롯 눈금이 정제 후 정상 도메인 위상으로 안정화됩니다.")

b_col1, b_col2 = st.columns(2)
box_features = ['money', 'outpay']

with b_col1:
    st.markdown("#### [원본] 주요 변수 분포 영역")
    fig_b1, ax_b1 = plt.subplots(figsize=(6, 4.2))
    fig_b1.patch.set_facecolor('#0E1117')
    ax_b1.set_facecolor('#111625')
    sns.boxplot(data=df_raw[box_features], palette=['#FF2E93', '#00E5FF'], ax=ax_b1)
    ax_b1.grid(True, color='#1A202C', linestyle='--', linewidth=0.5)
    st.pyplot(fig_b1)

with b_col2:
    st.markdown("#### [전처리 후] 이상치 제거 분포 영역")
    fig_b2, ax_b2 = plt.subplots(figsize=(6, 4.2))
    fig_b2.patch.set_facecolor('#0E1117')
    ax_b2.set_facecolor('#111625')
    sns.boxplot(data=df_clean[box_features], palette=['#FF2E93', '#00E5FF'], ax=ax_b2)
    ax_b2.grid(True, color='#1A202C', linestyle='--', linewidth=0.5)
    st.pyplot(fig_b2)

st.markdown("---")

# ==================== 3. 산점도 및 바 차트 (Scatter & Bar Chart) ====================
st.markdown("### 🎯 3. 머니 변동 대비 환급금 매트릭스 산점도 및 바 차트")

s_col1, s_col2 = st.columns(2)

with s_col1:
    st.markdown("#### 🔵 [원본 vs 전처리] 산점도 비교 (Scatter Plot)")
    fig_s, ax_s = plt.subplots(figsize=(6, 4.5))
    fig_s.patch.set_facecolor('#0E1117')
    ax_s.set_facecolor('#111625')
    
    ax_s.scatter(df_raw['money'], df_raw['outpay'], color='#FF5252', alpha=0.4, label='제거된 이상치 변수', s=30)
    ax_s.scatter(df_clean['money'], df_clean['outpay'], color='#00E676', alpha=0.8, label='최적 데이터 세그먼트', s=20)
    
    ax_s.set_xlabel('베팅 금액 (money)')
    ax_s.set_ylabel('환급 금액 (outpay)')
    ax_s.grid(True, color='#1A202C', linestyle='--', linewidth=0.5)
    ax_s.legend(facecolor='#111625', edgecolor='#232D42')
    st.pyplot(fig_s)

with s_col2:
    st.markdown("#### 📊 구간별 평균 환급 데이터 비교 (Bar Chart)")
    
    # 컴파일 에러 예방을 위해 독립적인 로컬 카피본을 만들어 구간화 및 집계 연산 수행
    df_raw_bar = df_raw.copy()
    df_clean_bar = df_clean.copy()
    
    group_labels = ['저액 베팅', '중액 베팅', '고액 베팅']
    df_raw_bar['group'] = pd.qcut(df_raw_bar['money'], q=3, labels=group_labels)
    df_clean_bar['group'] = pd.qcut(df_clean_bar['money'], q=3, labels=group_labels)
    
    raw_mean = df_raw_bar.groupby('group', observed=False)['outpay'].mean()
    clean_mean = df_clean_bar.groupby('group', observed=False)['outpay'].mean()
    
    bar_df = pd.DataFrame({'원본 평균': raw_mean, '정제 평균': clean_mean})
    
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4.5))
    fig_bar.patch.set_facecolor('#0E1117')
    ax_bar.set_facecolor('#111625')
    
    bar_df.plot(kind='bar', color=['#4A5568', '#00E5FF'], ax=ax_bar, rot=0)
    ax_bar.set_ylabel('평균 환급 스코어 (outpay)')
    ax_bar.set_xlabel('베팅 그룹 범주')
    ax_bar.grid(True, color='#1A202C', linestyle='--', linewidth=0.5)
    ax_bar.legend(facecolor='#111625', edgecolor='#232D42')
    st.pyplot(fig_bar)
