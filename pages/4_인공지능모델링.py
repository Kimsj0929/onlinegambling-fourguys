import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. 글로벌 레이아웃 설정
st.set_page_config(page_title="NEXUS Quantum AI Modeling", layout="wide")

# CSS 주입 (디자인 커스텀)
css_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');
html, body, [data-testid="stMarkdownContainer"] {
    font-family: 'Noto Sans KR', sans-serif;
}
.main-title { font-size: 2.2rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem; }
.sub-title { font-size: 1rem; color: #8A99AD; margin-bottom: 2rem; }
.section-header { margin-top: 1rem; margin-bottom: 1.5rem; color: #00E5FF; font-weight: 700; border-left: 5px solid #FF2E93; padding-left: 1rem; }
.metric-card { background-color: #111625; border: 1px solid #232D42; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; }
.metric-label { font-size: 0.8rem; color: #6C7D93; font-weight: 700; text-transform: uppercase; }
.metric-value { font-size: 1.8rem; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; }
.sub-value { font-size: 1.1rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; margin-top: 0.2rem; }
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# 헤더 영역
st.markdown('<div class="main-title">🌌 NEXUS Quantum AI Modeling</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Comparative Analysis Simulator: Linear Regression Line vs. Logistic Sigmoid Curve Topology</div>', unsafe_allow_html=True)

# 2. 데이터 세트 자립형 생성 엔진
@st.cache_data
def load_analysis_data():
    np.random.seed(42)
    n = 250
    X_mock = np.random.uniform(50, 500, n)
    logit = (X_mock - 275) / 40
    prob = 1 / (1 + np.exp(-logit))
    y_mock = np.random.binomial(1, prob)
    return pd.DataFrame({'money': X_mock, 'target': y_mock})

df = load_analysis_data()
X = df[['money']].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 훈련
lin_model = LinearRegression().fit(X_train, y_train)
log_model = LogisticRegression().fit(X_train, y_train)

# 평가지표 산출 (선형 MSE / 로지스틱 Accuracy)
mse = mean_squared_error(y_test, lin_model.predict(X_test))
acc = accuracy_score(y_test, log_model.predict(X_test))

# ==================== 실시간 예측 컨트롤러 ====================
st.markdown('<div class="section-header">🕹️ 하이퍼파라미터 실시간 제어 콘솔</div>', unsafe_allow_html=True)
min_x, max_x = float(X.min()), float(X.max())

# 컨트롤 슬라이더 레이블 한글화 및 내부 변수 영어 유지
user_val = st.slider("입력 데이터 제어 범위 설정: Money (Betting Amount)", min_x, max_x, float(X.mean()), step=0.1)

# 실시간 분석 추론 연산
pred_lin = lin_model.predict([[user_val]])[0]
pred_log_prob = log_model.predict_proba([[user_val]])[0][1]

st.markdown("<br>", unsafe_allow_html=True)

# ==================== 두 파트로 분리된 스코어 및 평가지표 메트릭 ====================
p_col1, p_col2 = st.columns(2)

# 구문 분석 충돌 에러 방지를 위해 메트릭 데이터 가공 사전 처리
val_lin = f"{pred_lin:.4f}"
val_mse = f"{mse:.4f}"
val_prob = f"{pred_log_prob * 10
