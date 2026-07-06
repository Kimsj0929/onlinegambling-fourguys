import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 페이지 레이아웃 설정
st.set_page_config(page_title="NEXUS 인텔리전트 데이터 캔버스", layout="wide")

# CSS 주입 (파이썬 구문 파서 충돌 요소를 배제한 안전한 문자열)
css_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
html, body, [data-testid="stMarkdownContainer"] {
    font-family: 'Noto Sans KR', sans-serif;
}
.panel-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #00E5FF;
    margin-bottom: 0.5rem;
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
        # 인위적 극단 이상치 심화 주입
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

# --- 상단 디자인 헤der ---
st.markdown("## 🌌 NEXUS 다이내믹 데이터 분석 캔버스")
st.markdown("하이퍼파라미터를 조정하여 실시간 오버-필터링 트리거와 토폴로지 변화를 관측하십시오.")
st.markdown("---")

# ==================== 실시간 컨트롤러 컨솔 ====================
st.markdown("<div class='panel-title'>🎛️ 실시간 데이터 필터링 가중치 제어</div>", unsafe_allow_html=True)

# 슬라이더 값에 따라 이상치를 자르는 '엄격함'이 실시간 조절됨
iqr_weight = st.slider(
    "IQR(사분위수 범위) 이상치 차단 가중치 (낮을수록 더 엄격하게 데이터를 잘라냄)", 
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

# 슬라이더 수치를 실시간 반영한 정제 데이터셋 생성
df_clean = preprocess_dynamic(df_raw, iqr_weight)

# 대시보드 스코어보드 텍스트 사전 결합
raw_count_str = str(len(df_raw)) + " Rows"
clean_count_str = str(len(df_clean)) + " Rows"
removed_count = len(df_raw) - len(df_clean)
delta_status_str = "-" + str(removed_count) + " 개 데이터 격리됨"

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="입력 원본 엔터티 총량", value=raw_count_str)
with col_m2:
    st.metric(label="동적 정제 완료 엔터티 총량", value=clean_count_str, delta=delta_status_str)

st.markdown("---")

# 변수 리스트 구성
numeric_features = ['gamers', 'skins', 'money', 'ticks', 'outpay']
features = [c for c in numeric_features if c in df_raw.columns]

# ==================== GRAPH 1. 상관관계 히트맵 (Dynamic) ====================
st.markdown("<div class='panel-title'>🌡️ 1. 변수 간 상관관계 토폴로지 히트맵</div>", unsafe_allow_html=True)
h_col1, h_col2 = st.columns(2)

corr_raw = df_raw[features].corr()
corr_clean = df_clean[features].corr()

with h_col1:
    st.markdown("#### [Raw] 원본 상관 행렬")
    fig_h1, ax_h1 = plt.subplots(figsize=(6, 3.8))
    fig_h1.patch.set_facecolor('#0E1117')
    ax_h1.set_facecolor('#111625')
    sns.heatmap(corr_raw, annot=True, cmap='vlag', fmt=".2f", ax=ax_h1, cbar=False, vmin=-1, vmax=1)
    st.pyplot(fig_h1)

with h_col2:
    st.markdown("#### [Clean] 필터링 가중치 반영 상관 행렬")
    fig_h2, ax_h2 = plt.subplots(figsize=(6, 3.8))
    fig_h2.patch.set_facecolor('#0E1117')
    ax_h2.set_facecolor('#111625')
    sns.heatmap(corr_clean, annot=True, cmap='vlag', fmt=".2f", ax=ax_h2, cbar=False, vmin=-1, vmax=1)
    st.pyplot(fig_h2)

st.markdown("---")

# ==================== GRAPH 2. 분포 박스플롯 (Dynamic) ====================
st.markdown("<div class='panel-title'>📦 2. 변수 도메인 스케일 변화 제어 박스플롯</div>", unsafe_allow_html=True)
b_col1, b_col2 = st.columns(2)
box_features = ['money', 'outpay']

with b_col1:
    st.markdown("#### [Raw] 극단적 이상치 포함 상태")
    fig_b1, ax_b1 = plt.subplots(figsize=(6, 3.8))
    fig_b1.patch.set_facecolor('#0E1117')
    ax_b1.set_facecolor('#111625')
    sns.boxplot(data=df_raw[box_features], palette=['#FF2E93', '#00E5FF'], ax=ax_b1)
    ax_b1.grid(True, color='#1A202C', linestyle=':', linewidth=0.6)
    st.pyplot(fig_b1)

with b_col2:
    st.markdown("#### [Clean] 실시간 경계선 조절 상태")
    fig_b2, ax_b2 = plt.subplots(figsize=(6, 3.8))
    fig_b2.patch.set_facecolor('#0E1117')
    ax_b2.set_facecolor('#111625')
    sns.boxplot(data=df_clean[box_features], palette=['#FF2E93', '#00E5FF'], ax=ax_b2)
    ax_b2.grid(True, color='#1A202C', linestyle=':', linewidth=0.6)
    st.pyplot(fig_b2)

st.markdown("---")

# ==================== GRAPH 3. 산점도 및 바 차트 (Dynamic) ====================
st.markdown("<div class='panel-title'>🎯 3. 분포 구조 산점도 및 구간 집계 데이터 바 차트</div>", unsafe_allow_html=True)
s_col1, s_col2 = st.columns(2)

with s_col1:
    st.markdown("#### 🔵 데이터 컴포넌트 실시간 격리 산점도")
    fig_s, ax_s = plt.subplots(figsize=(6, 4.2))
    fig_s.patch.set_facecolor('#0E1117')
    ax_s.set_facecolor('#111625')
    
    # 두 데이터군을 겹쳐서 표현하되, 슬라이더에 의해 실시간으로 탈락하는 데이터가 무엇인지 빨간색으로 시각화
    ax_s.scatter(df_raw['money'], df_raw['outpay'], color='#FF5252', alpha=0.3, label='이상치 판단 영역', s=35)
    ax_s.scatter(df_clean['money'], df_clean['outpay'], color='#00E676', alpha=0.8, label='정상 정제 영역', s=20)
    
    ax_s.set_xlabel('베팅 금액 (money)')
    ax_s.set_ylabel('환급 금액 (outpay)')
    ax_s.grid(True, color='#1A202C', linestyle=':', linewidth=0.6)
    ax_s.legend(facecolor='#111625', edgecolor='#232D42')
    st.pyplot(fig_s)

with s_col2:
    st.markdown("#### 📊 필터 가중치 적용에 따른 구간별 평균 환급 바 차트")
    
    # 독립적 연산 데이터프레임 카피본 구성
    df_raw_bar = df_raw.copy()
    df_clean_bar = df_clean.copy()
    
    group_labels = ['저액 베팅', '중액 베팅', '고액 베팅']
    df_raw_bar['group'] = pd.qcut(df_raw_bar['money'], q=3, labels=group_labels)
    
    # 청소된 데이터가 너무 적어 qcut 에러가 발생할 것에 대비한 안전 예외 처리
    try:
        df_clean_bar['group'] = pd.qcut(df_clean_bar['money'], q=3, labels=group_labels)
    except:
        df_clean_bar['group'] = pd.cut(df_clean_bar['money'], bins=3, labels=group_labels)
    
    raw_mean = df_raw_bar.groupby('group', observed=False)['outpay'].mean()
    clean_mean = df_clean_bar.groupby('group', observed=False)['outpay'].mean()
    
    bar_df = pd.DataFrame({'원본 데이터 평균': raw_mean, '실시간 정제 평균': clean_mean})
    
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4.2))
    fig_bar.patch.set_facecolor('#0E1117')
    ax_bar.set_facecolor('#111625')
    
    bar_df.plot(kind='bar', color=['#4A5568', '#00E5FF'], ax=ax_bar, rot=0)
    ax_bar.set_ylabel('평균 환급 금액 (outpay)')
    ax_bar.set_xlabel('머니 구간 베팅 그룹')
    ax_bar.grid(True, color='#1A202C', linestyle=':', linewidth=0.6)
    ax_bar.legend(facecolor='#111625', edgecolor='#232D42')
    st.pyplot(fig_bar)
