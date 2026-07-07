import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. 글로벌 레이아웃 설정
st.set_page_config(page_title="NEXUS Quantum AI | Dashboard", layout="wide")

# CSS 주입 (디자인 템플릿)
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
st.markdown('<div class="main-title">NEXUS Engine │ Quantum AI Core</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Comparative Analysis Simulator: Linear Regression Line vs. Logistic Sigmoid Curve Topology</div>', unsafe_allow_html=True)

# 2. 외부 파일 의존성 제거: 즉시 완벽한 S곡선 데이터 생성
@st.cache_data
def generate_perfect_data():
    np.random.seed(42)
    X_mock = np.random.uniform(50, 500, 250)
    logit = (X_mock - 275) / 40
    prob = 1 / (1 + np.exp(-logit))
    y_mock = np.random.binomial(1, prob)
    return pd.DataFrame({'money': X_mock, 'is_high_outpay': y_mock})

df = generate_perfect_data()
X = df[['money']].values
y = df['is_high_outpay'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 에러 원천 차단: scikit-learn 모델 선언 및 학습 명확화
lin_reg = LinearRegression().fit(X_train, y_train)
log_reg = LogisticRegression().fit(X_train, y_train)

# 모델 검증 스코어 사전 연산
mse = mean_squared_error(y_test, lin_reg.predict(X_test))
acc = accuracy_score(y_test, log_reg.predict(X_test))

# ==================== 컨트롤러 (실시간 매개변수 입력) ====================
st.markdown("### 🎛️ Hyperparameter Real-time Control Console")
min_x, max_x = float(X.min()), float(X.max())

# 인터랙티브 슬라이더
user_value = st.slider("Input Parameter Control: Money (Betting Amount)", min_x, max_x, float(X.mean()), step=0.1)

# AI 실시간 추론(Inference) 연산
user_X = np.array([[user_value]])
res_lin = lin_reg.predict(user_X)[0]
res_log_prob = log_reg.predict_proba(user_X)[0][1]
res_log_class = 1 if res_log_prob >= 0.5 else 0

st.markdown("<br>", unsafe_allow_html=True)

# ==================== 실시간 분석 메트릭 시각화 ====================
p_col1, p_col2 = st.columns(2)

with p_col1:
    if res_lin < 0 or res_lin > 1:
        desc_text = "Out of Bounds: Linear model limitation produced an impossible score."
    else:
        desc_text = "Within Bounds: Continuous point score calculated via linear equation."
        
    val_lin = f"{res_lin:.4f}"
    val_mse = f"{mse:.4f}"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Linear Regression Prediction & Error</div>
        <div class="metric-value">{val_lin}</div>
        <div class="sub-value">Model MSE: {val_mse}</div>
        <div class="metric-desc">{desc_text}<br><span style='color:#6C7D93;'>* Predicts continuous values; evaluated by Mean Squared Error (closer to 0 is better).</span></div>
    </div>
    """, unsafe_allow_html=True)

with p_col2:
    if res_log_class == 1:
        status_color = "#00E676"
        status_text = "Final Decision: High Outpay (Success)"
    else:
        status_color = "#FF5252"
        status_text = "Final Decision: Low Outpay (Failure)"
        
    val_prob = f"{res_log_prob * 100:.2f}"
    val_acc = f"{acc * 100:.1f}"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Logistic Regression Probability & Accuracy</div>
        <div class="metric-value">{val_prob}%</div>
        <div class="sub-value" style="color:#00E5FF;">Model Accuracy: {val_acc}%</div>
        <div class="metric-desc" style="color:{status_color}; font-weight:600;">{status_text}</div>
        <div class="metric-desc" style="margin-top:0px;"><span style='color:#6C7D93;'>* Classifies success/failure; evaluated by Accuracy Score (closer to 100% is better).</span></div>
    </div>
    """, unsafe_allow_html=True)

# ==================== 시각화 분석 그래픽스 엔진 ====================
col1, col2 = st.columns(2)

X_range = np
