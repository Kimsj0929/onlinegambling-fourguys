import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. 글로벌 레이아웃 설정
st.set_page_config(page_title="NEXUS Unified Intelligence Canvas", layout="wide")

# CSS 주입 (디자인)
css_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');
html, body, [data-testid="stMarkdownContainer"] {
    font-family: 'Noto Sans KR', sans-serif;
}
.main-title { font-size: 2.2rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem; }
.sub-title { font-size: 1rem; color: #8A99AD; margin-bottom: 2rem; }
.section-header { margin-top: 3rem; margin-bottom: 1.5rem; color: #00E5FF; font-weight: 700; border-left: 5px solid #FF2E93; padding-left: 1rem; }
.metric-card { background-color: #111625; border: 1px solid #232D42; border-radius: 8px; padding: 1.2rem; }
.metric-label { font-size: 0.8rem; color: #6C7D93; font-weight: 700; text-transform: uppercase; }
.metric-value { font-size: 1.8rem; color: #00E5FF; font-family: 'JetBrains Mono', monospace; }
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# 헤더 영역
st.markdown('<div class="main-title">🌌 NEXUS Intelligence Canvas</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time Data Filtering and Quantum AI Modeling Simulator</div>', unsafe_allow_html=True)

# 2. 데이터 생성 엔진 (파일 없어도 작동)
@st.cache_data
def load_unified_data():
    np.random.seed(42)
    n = 300
    money = np.random.uniform(50, 500, n)
    
    # 상관관계 부여 (Money -> Outpay)
    noise = np.random.normal(0, 40, n)
    outpay = (money * 0.85) + noise
    
    # 극단적 이상치 주입
    outpay[12], outpay[85], outpay[150] = 2200, -600, 1800
    
    # AI 모델용 이진 분류 타겟 생성 (Sigmoid 기반 확률)
    logit = (money - 275) / 40
    prob = 1 / (1 + np.exp(-logit))
    is_high_outpay = np.random.binomial(1, prob)
    
    return pd.DataFrame({
        'gamers': np.random.randint(40, 220, n),
        'skins': np.random.randint(90, 320, n),
        'money': money,
        'ticks': np.random.uniform(1.0, 18.0, n),
        'outpay': outpay,
        'target': is_high_outpay
    })

df_raw = load_unified_data()

# ==================== STEP 1. 데이터 정제 섹션 ====================
st.markdown('<div class="section-header">STEP 1. Dynamic Data Preprocessing (IQR Filter)</div>', unsafe_allow_html=True)

iqr_weight = st.slider("Outlier Detection Strength (Lower = Stricter)", 0.3, 3.0, 1.5, 0.1)

def preprocess_dynamic(df, weight):
    df_c = df.copy()
    for col in ['money', 'outpay']:
        Q1, Q3 = df_c[col].quantile(0.25), df_c[col].quantile(0.75)
        IQR = Q3 - Q1
        df_c = df_c[(df_c[col] >= Q1 - weight * IQR) & (df_c[col] <= Q3 + weight * IQR)]
    return df_c
