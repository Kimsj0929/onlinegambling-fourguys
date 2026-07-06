import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. 딥 테크 기업 스타일의 글로벌 레이아웃 설정
st.set_page_config(page_title="NEXUS QUANTUM AI | ANALYTICS", layout="wide")

# [⚠️ 수정 완료] unsafe_allow_html=True 로 옵션명 변경 및 스타일 튜닝
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    html, body, [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', sans-serif;
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
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        color: #6C7D93;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #00E5FF;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-desc {
        font-size: 0.8rem;
        color: #A0AEC0;
        margin-top: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# 헤더 영역
st.markdown('<div class="main-title">NEXUS ANALYTICS │ QUANTUM AI ENGINE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Comparative Analysis of Linear Projection vs. Logistic Probability Topologies</div>', unsafe_allow_html=True)

# 2. 데이터 시뮬레이션 및 파이프라인 빌드
@st.cache_data
def load_analytical_data():
    try:
        df = pd.read_csv("onlineCasino.csv")
        df['is_high_outpay'] = (df['outpay'] > df['money']).astype(int)
        return df
    except:
        np.random.seed(42)
        X_mock = np.random.uniform(50, 500, 250)
        y_mock = (X_mock > 260).astype(int)
        noise = np.random.choice([0, 1], size=250, p=[0.92, 0.08])
        y_mock = np.abs(y_mock - noise)
        return pd.DataFrame({'money': X_mock, 'is_high_outpay': y_mock})

df = load_analytical_data()
X = df[['money']].values
y = df['is_high_outpay'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 백엔드 연산
lin_reg = LinearRegression().fit(X_train, y_train)
log_reg = LogisticRegression().fit(X_train, y_train)

# ==================== CONTROLLER (PARAMETER INPUT) ====================
st.markdown("### 🎛️ PARAMETER REAL-TIME CONFIGURATOR")
min_x, max_x = float(X.min()), float(X.max())

# 인터랙티브 컨트롤 슬라이더
user_value = st.slider("HYPER-PARAMETER INPUT: BETA VALUE (MONEY)", min_x, max_x, float(X.mean()), step=0.1)

# 실시간 인프런스(Inference) 연산
user_X = np.array([[user_value]])
res_lin = lin_reg.predict(user_X)[0]
res_log_prob = log_reg.predict_proba(user_X)[0][1]
res_log_class = 1 if res_log_prob >= 0.5 else 0

st.markdown("<br>", unsafe_allow_html=True)

# ==================== METRIC SHOWCASE ====================
p_col1, p_col2 = st.columns(2)

with p_col1:
    desc_text = "<span style='color:#FF5252;'>⚠️ Out of bounds ([0, 1] Range Violated)</span>" if (res_lin < 0 or res_lin > 1) else "✅ Within theoretical bounds"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Linear Projection Output (Continuous)</div>
        <div class="metric-value">{res_lin:.4f}</div>
        <div class="metric-desc">{desc_text}</div>
    </div>
    """, unsafe_allow_html=True)

with p_col2:
    status_color = "#00E676" if res_log_class == 1 else "#FF5252"
    status_text = "STATE: ACTIVE (1)" if res_log_class == 1 else "STATE: INACTIVE (0)"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Logistic Sigmoid Probability</div>
        <div class="metric-value">{res_
