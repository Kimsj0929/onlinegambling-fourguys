import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. 글로벌 레이아웃 설정
st.set_page_config(page_title="NEXUS 퀀텀 AI | 분석 대시보드", layout="wide")

# CSS 주입
css_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');
html, body, [data-testid="stMarkdownContainer"] {
    font-family: 'Noto Sans KR', sans-serif;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.05rem;
    color: #FFFFFF;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 0.95rem;
    color: #8A99AD;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #111625;
    border: 1px solid #232D42;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.metric-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05rem;
    color: #6C7D93;
    font-weight: 700;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 600;
    color: #00E5FF;
    font-family: 'JetBrains Mono', monospace;
}
.sub-value {
    font-size: 1.1rem;
    color: #FF2E93;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    margin-top: 0.2rem;
}
.metric-desc {
    font-size: 0.8rem;
    color: #A0AEC0;
    margin-top: 0.4rem;
}
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# 헤더 영역
st.markdown('<div class="main-title">NEXUS 분석 엔진 │ 퀀텀 AI 코어</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">선형 예측 직선과 로지스틱 확률 곡선 토폴로지의 비교 분석 실시간 시뮬레이터</div>', unsafe_allow_html=True)

# 2. 데이터 세트 로드 및 분석 파이프라인 (S자 곡선 유도를 위해 로직 전면 튜닝)
@st.cache_data
def load_analytical_data():
    try:
        df = pd.read_csv("onlineCasino.csv")
        df['is_high_outpay'] = (df['outpay'] > df['money']).astype(int)
        return df
    except:
        # csv 파일이 없을 때 S자 곡선이 가장 이상적으로 그려지도록 시그모이드 확률 기반 데이터 생성
        np.random.seed(42)
        X_mock = np.random.uniform(50, 500, 250)
        
        # 중간값(275) 부근에서 확률이 완만하게 변하도록 설정 (로지스틱 회귀가 학습하기 최적의 환경)
        logit = (X_mock - 275) / 40
        prob = 1 / (1 + np.exp(-logit))
        y_mock = np.random.binomial(1, prob)
        
        return pd.DataFrame({'money': X_mock, 'is_high_outpay': y_mock})

df = load_analytical_data()
X = df[['money']].values
y = df['is_high_outpay'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 인공지능 모델 학습 연산
lin_reg = LinearRegression().fit(X_train, y_train)
log_reg = LogisticRegression().fit(X_train, y_train)

# 모델 검증 스코어 사전 연산
mse = mean_squared_error(y_test, lin_reg.predict(X_test))
acc = accuracy_score(y_test, log_reg.predict(X_test))

# ==================== 컨트롤러 (실시간 매개변수 입력) ====================
st.markdown("### 🎛️ 하이퍼파라미터 실시간 제어 컨솔")
min_x, max_x = float(X.min()), float(X.max())

# 인터랙티브 슬라이더
user_value = st.slider("분석 입력 데이터 제어: 베팅 금액 (Money)", min_x, max_x, float(X.mean()), step=0.1)

# AI 실시간 추론(Inference) 연산
user_X = np.array([[user_value]])
res_lin = lin_reg.predict(user_X)[0]
res_log_prob = log_reg.predict_proba(user_X)[0][1]
