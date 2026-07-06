import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

# 1. 딥 테크 기업 스타일의 글로벌 레이아웃 설정
st.set_page_config(page_title="NEXUS QUANTUM AI | ANALYTICS", layout="wide")

# 폰트 및 다크 테마 커스텀 CSS 주입 (고급스러운 마크업)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    html, body, [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', sans-serif;
    }
    .mono-text {
        font-family: 'JetBrains Mono', monospace;
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
""", unsafe_index=True)

# 헤더 영역
st.markdown('<div class="main-title">NEXUS ANALYTICS │ QUANTUM AI ENGINE</div>', unsafe_index=True)
st.markdown('<div class="sub-title">Comparative Analysis of Linear Projection vs. Logistic Probability Topologies</div>', unsafe_index=True)

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

# 인터랙티브 컨트롤 휠 대용 슬라이더
user_value = st.slider("HYPER-PARAMETER INPUT: BETA VALUE (MONEY)", min_x, max_x, float(X.mean()), step=0.1)

# 실시간 인프런스(Inference) 연산
user_X = np.array([[user_value]])
res_lin = lin_reg.predict(user_X)[0]
res_log_prob = log_reg.predict_proba(user_X)[0][1]
res_log_class = 1 if res_log_prob >= 0.5 else 0

st.markdown("<br>", unsafe_index=True)

# ==================== METRIC SHOWCASE ====================
p_col1, p_col2 = st.columns(2)

with p_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Linear Projection Output (Continuous)</div>
        <div class="metric-value">{res_lin:.4f}</div>
        <div class="metric-desc">{"<span style='color:#FF5252;'>⚠️ Out of bounds ([0, 1] Range Violated)</span>" if (res_lin < 0 or res_lin > 1) else "✅ Within theoretical bounds"}</div>
    </div>
    """, unsafe_index=True)

with p_col2:
    status_color = "#00E676" if res_log_class == 1 else "#FF5252"
    status_text = "STATE: ACTIVE (1)" if res_log_class == 1 else "STATE: INACTIVE (0)"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Logistic Sigmoid Probability</div>
        <div class="metric-value">{res_log_prob * 100:.2f}%</div>
        <div class="metric-desc" style="color:{status_color}; font-weight:600;">{status_text} (Threshold: 0.5)</div>
    </div>
    """, unsafe_index=True)

# ==================== VISUALIZATION ENGINE ====================
col1, col2 = st.columns(2)

# 고해상도 S-곡선 매핑을 위한 촘촘한 도메인 생성
X_range = np.linspace(min_x, max_x, 500).reshape(-1, 1)
lin_line = lin_reg.predict(X_range)
log_curve = log_reg.predict_proba(X_range)[:, 1]

# Matplotlib 글로벌 다크 스타일 정의
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#8A99AD'
plt.rcParams['axes.labelcolor'] = '#8A99AD'
plt.rcParams['xtick.color'] = '#4A5568'
plt.rcParams['ytick.color'] = '#4A5568'

# --- Left Graph: Linear ---
with col1:
    st.markdown("#### 📉 LINEAR REGRESSION PROJECTION")
    fig1, ax1 = plt.subplots(figsize=(7, 4.2))
    fig1.patch.set_facecolor('#0E1117')
    ax1.set_facecolor('#111625')
    
    ax1.scatter(X_test, y_test, color='#2D3748', alpha=0.6, s=25, label='Empirical Data', zorder=2)
    ax1.plot(X_range, lin_line, color='#FF2E93', linewidth=2.5, label='Linear State', zorder=3)
    
    # 실시간 포인트 인디케이터 (주황 크로스헤어 & 대형 타겟 도트)
    ax1.scatter(user_value, res_lin, color='#FFD700', edgecolor='#FFFFFF', s=160, marker='o', zorder=5, label='Current State')
    ax1.axvline(user_value, color='#4A5568', linestyle=':', alpha=0.5, zorder=1)
    
    ax1.set_ylim(-0.3, 1.3)
    ax1.grid(True, color='#1A202C', linestyle='--', linewidth=0.8)
    ax1.legend(facecolor='#111625', edgecolor='#232D42', loc='upper left')
    st.pyplot(fig1)

# --- Right Graph: Logistic ---
with col2:
    st.markdown("#### 📈 LOGISTIC CLASSIFICATION CURVE")
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    fig2.patch.set_facecolor('#0E1117')
    ax2.set_facecolor('#111625')
    
    ax2.scatter(X_test, y_test, color='#2D3748', alpha=0.6, s=25, label='Empirical Data', zorder=2)
    ax2.plot(X_range, log_curve, color='#00E5FF', linewidth=2.5, label='Sigmoid Function', zorder=3)
    ax2.axhline(0.5, color='#718096', linestyle='--', linewidth=1, alpha=0.6, label='Decision Boundary (0.5)')
    
    # 실시간 포인트 인디케이터
    ax2.scatter(user_value, res_log_prob, color='#FFD700', edgecolor='#FFFFFF', s=160, marker='o', zorder=5, label='Current State')
    ax2.axvline(user_value, color='#4A5568', linestyle=':', alpha=0.5, zorder=1)
    
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, color='#1A202C', linestyle='--', linewidth=0.8)
    ax2.legend(facecolor='#111625', edgecolor='#232D42', loc='upper left')
    st.pyplot(fig2)

# ==================== SYNOPSIS & PERFORMANCE REPORT ====================
st.markdown("<br>", unsafe_index=True)
st.markdown("### ⚖️ MODEL SYNOPSIS & EFFICIENCY REPORT")

# 오차 및 검증 행렬 연산
mse = mean_squared_error(y_test, lin_reg.predict(X_test))
acc = accuracy_score(y_test, log_reg.predict(X_test))

# 고성능 매트릭스 표 테이블 시각화
summary_matrix = pd.DataFrame({
    "MATRICES": ["OBJECTIVE FUNCTION", "OUTPUT TOPOLOGY", "CLASSIFICATION APTITUDE", "VALIDATION SCORE"],
    "MODEL 01: LINEAR REGRESSION": [
        "Continuous Value Optimization",
        "Unbounded (-∞ to +∞)",
        "Not Recommended (Mathematical Distortion)",
        f"MSE: {mse:.4f}"
    ],
    "MODEL 02: LOGISTIC REGRESSION": [
        "Binary Class Probability Optimization",
        "Bounded Sigmoid Space ([0.0, 1.0])",
        "Highly Optimal (Definitive Categorization)",
        f"ACCURACY: {acc*100:.1f}%"
    ]
})

st.table(summary_matrix.set_index("MATRICES"))

# 인텔리전스 브리프 서머리
st.markdown("""
> **INTELLIGENCE BRIEF:** > 본 분석 엔진 검증 결과, 종속 변수가 이진 범주형($\{0, 1\}$) 구조를 가질 때 **Linear Model**은 도메인 경계를 초과하는 수치적 왜곡을 초래합니다. 
> 반면, **Logistic Model**은 출력을 0과 1 사이의 기하학적 확률 곡선 공간 내에 완전 구속함으로써 예리한 임계 기준 예측을 가능하게 합니다. 
""")
