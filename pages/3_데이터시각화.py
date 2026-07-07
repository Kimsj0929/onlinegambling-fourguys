import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 페이지 레이아웃 설정
st.set_page_config(page_title="NEXUS 인텔리전트 데이터 캔버스", layout="wide")

# CSS 주입 (안전한 기본 템플릿)
css_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
html, body, [data-testid="stMarkdownContainer"] {
    font-family: 'Noto Sans KR', sans-serif;
}
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# Matplotlib 테마 및 폰트 강제 고정 (한글 폰트 미설치 서버용 기본 고딕 설정)
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.color'] = '#8A99AD'
plt.rcParams['axes.labelcolor'] = '#8A99AD'

# 2. 데이터 세트 로드 (상관관계가 확실히 나타나도록 데이터 생성 로직 수정)
@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv("onlineCasino.csv")
    except:
        np.random.seed(42)
        n = 300
        
        # 기본 독립변수 (베팅 금액)
        money = np.random.uniform(50, 500, n)
        
        # 양의 상관관계 시나리오 (Money가 커질수록 Outpay도 커짐)
        # 배율을 높이고 노이즈를 줄여 대각선 모양을 더 명확하게 만듦
        base_outpay = money * 2.5 
        
        # 현실감을 위한 노이즈(변동성) 추가
        noise = np.random.normal(0, 100, n)
        outpay = base_outpay + noise
        
        # 이상치(Outliers) 강제 주입 - IQR 필터 슬라이더의 작동을 극적으로 보여주기 위함
        outpay[12] = 4000   # 거대한 양의 이상치
        outpay[85] = -1000   # 거대한 음의 이상치
        outpay[150] = 3500  # 추가 이상치
        
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
st.markdown("#### 🎛️ 실시간 데이터 필터링 강도 제어")

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

# ==================== GRAPH 1. 상관관계 히트맵 ====================
st.markdown("#### 🌡️ 1. 데이터 항목 간의 수치적 연관성 지도")
h_col1, h_col2 = st.columns(2)

corr_raw = df_raw[features].corr()
corr_clean = df_clean[features].corr()

clean_labels = ['Gamers', 'Skins', 'Money', 'Ticks', 'Outpay']

with h_col1:
    st.markdown("##### [Raw] 원본 상관성")
    fig_h1, ax_h1 = plt.subplots(figsize=(6, 3.8))
    fig_h1.patch.set_facecolor('#0E1117')
    ax_h1.set_facecolor('#111625')
    sns.heatmap(corr_raw, annot=True, cmap='vlag', fmt=".2f", ax=ax_h1, cbar=False, vmin=-1, vmax=1, xticklabels=clean_labels, yticklabels=clean_labels)
    st.pyplot(fig_h1)
    plt.close(fig_h1)  # 메모리 해제

with h_col2:
    st.markdown("##### [Clean] 정제 후 상관성")
    fig_h2, ax_h2 = plt.subplots(figsize=(6, 3.8))
    fig_h2.patch.set_facecolor('#0E1117')
    ax_h2.set_facecolor('#111625')
    sns.heatmap(corr_clean, annot=True, cmap='vlag', fmt=".2f", ax=ax_h2, cbar=False, vmin=-1, vmax=1, xticklabels=clean_labels, yticklabels=clean_labels)
    st.pyplot(fig_h2)
    plt.close(fig_h2)  # 메모리 해제

st.markdown("---")

# ==================== GRAPH 2. 분포 박스플롯 ====================
st.markdown("#### 📦 2. 주요 데이터 수치 범위 분포")
b_col1, b_col2 = st.columns(2)
box_features = ['money', 'outpay']

with b_col1:
    st.markdown("##### [Raw] 이상치 포함 영역")
    fig_b1, ax_b1 = plt.subplots(figsize=(6, 3.8))
    fig_b1.patch.set_facecolor('#0E1117')
    ax_b1.set_facecolor('#111625')
    
    sns.boxplot(data=df_raw[box_features], palette=['#FF2E93', '#00E5FF'], ax=ax_b1)
    
    ax_b1.set_xticklabels(['Money', 'Outpay'])
    ax_b1.set_ylabel("Value Range")
    ax_b1.grid(True, color='#1A202C', linestyle=':', linewidth=0.6)
    st.pyplot(fig_b1)
    plt.close(fig_b1)

with b_col2:
    st.markdown("##### [Clean] 경계선 실시간 조절")
    fig_b2, ax_b2 = plt.subplots(figsize=(6, 3.8))
    fig_b2.patch.set_facecolor('#0E1117')
    ax_b2.set_facecolor('#111625')
    
    sns.boxplot(data=df_clean[box_features], palette=['#FF2E93', '#00E5FF'], ax=ax_b2)
    
    ax_b2.set_xticklabels(['Money', 'Outpay'])
    ax_b2.set_ylabel("Value Range")
    ax_b2.grid(True, color='#1A202C', linestyle=':', linewidth=0.6)
    st.pyplot(fig_b2)
    plt.close(fig_b2)

st.markdown("---")

# ==================== GRAPH 3. 산점도 및 바 차트 (산점도 수정본 적용) ====================
st.markdown("#### 🎯 3. 데이터 분포 형태 및 등급별 평균 환급 구조")
s_col1, s_col2 = st.columns(2)

with s_col1:
    st.markdown("##### 🔵 데이터 분포 산점도")
    fig_s, ax_s = plt.subplots(figsize=(6, 4.2))
    fig_s.patch.set_facecolor('#0E1117')
    ax_s.set_facecolor('#111625')
    
    # 1. 원본 데이터 (이상치가 넓게 펼쳐진 투명한 붉은 점)
    ax_s.scatter(df_raw['money'], df_raw['outpay'], color='#FF5252', alpha=0.3, label='Outliers', s=35)
    
    # 2. 정제 데이터 (정상 구역에 모여있는 선명한 녹색 점)
    ax_s.scatter(df_clean['money'], df_clean['outpay'], color='#00E676', alpha=0.9, label='Cleaned', s=25)
    
    # [핵심] Y축 범위를 정제 완료 데이터 기준으로 자동 한계 제어하여 우상향 경향성을 극대화함
    if not df_clean.empty:
        # 데이터의 실제 범위에 맞춰 Y축 범위를 설정
        y_min = df_clean['outpay'].min() - 100
        y_max = df_clean['outpay'].max() + 100
        ax_s.set_ylim(y_min, y_max)
    
    ax_s.set_xlabel('Money')
    ax_s.set_ylabel('Outpay')
    ax_s.grid(True, color='#1A202C', linestyle=':', linewidth=0.6)
    ax_s.legend(facecolor='#111625', edgecolor='#232D42')
    st.pyplot(fig_s)
    plt.close(fig_s)

with s_col2:
    st.markdown("##### 📊 규모별 평균 환급 비교")
    
    df_raw_bar = df_raw.copy()
    df_clean_bar = df_clean.copy()
    
    group_labels = ['Low', 'Medium', 'High']
    df_raw_bar['group'] = pd.qcut(df_raw_bar['money'], q=3, labels=group_labels)
    
    try:
        df_clean_bar['group'] = pd.qcut(df_clean_bar['money'], q=3, labels=group_labels)
    except:
        df_clean_bar['group'] = pd.cut(df_clean_bar['money'], bins=3, labels=group_labels)
    
    raw_mean = df_raw_bar.groupby('group', observed=False)['outpay'].mean()
    clean_mean = df_clean_bar.groupby('group', observed=False)['outpay'].mean()
    
    bar_df = pd.DataFrame({'Raw Mean': raw_mean, 'Clean Mean': clean_mean})
    
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4.2))
    fig_bar.patch.set_facecolor('#0E1117')
    ax_bar.set_facecolor('#111625')
    
    bar_df.plot(kind='bar', color=['#4A5568', '#00E5FF'], ax=ax_bar, rot=0)
    
    ax_bar.set_xlabel('Betting Groups')
    ax_bar.set_ylabel('Average Outpay')
    ax_bar.grid(True, color='#1A202C', linestyle=':', linewidth=0.6)
    ax_bar.legend(facecolor='#111625', edgecolor='#232D42')
    st.pyplot(fig_bar)
    plt.close(fig_bar)
