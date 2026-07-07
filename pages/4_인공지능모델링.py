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

# CSS 주입 (매직 파서 우회를 위해 함수화 처리 및 특수 기호 마스킹)
def inject_custom_css():
    css_content = (
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');"
        "html, body, [data-testid='stMarkdownContainer'] { font-family: 'Noto Sans KR', sans-serif; }"
        ".main-title { font-size: 2.2rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem; }"
        ".sub-title { font-size: 1rem; color: #8A99AD; margin-bottom: 2rem; }"
        ".section-header { margin-top: 1rem; margin-bottom: 1.5rem; color: #00E5FF; font-weight: 700; border-left: 5px solid #FF2E93; padding-left: 1rem; }"
        ".metric-card { background-color: #111625; border: 1px solid #232D42; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; }"
        ".metric-label { font-size: 0.8rem; color: #6C7D93; font-weight: 700; text-transform: uppercase; }"
        ".metric-value { font-size: 1.8rem; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; }"
        ".sub-value { font-size: 1.1rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; margin-top: 0.2rem; }"
        "</style>"
    )
    st.markdown(css_content, unsafe_allow_html=True)

inject_custom_css()

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
min_x = float(X.min())
max_x = float(X.max())
mean_x = float(X.mean())

# 컨트롤 슬라이더
user_val = st.slider("입력 데이터 제어 범위 설정: Money (Betting Amount)", min_x, max_x, mean_x, step=0.1)

# 실시간 분석 추론 연산
pred_lin = lin_model.predict([[user_val]])[0]
pred_log_prob = log_model.predict_proba([[user_val]])[0][1]

st.markdown("<br>", unsafe_allow_html=True)

# ==================== 두 파트로 분리된 스코어 및 평가지표 메트릭 ====================
p_col1, p_col2 = st.columns(2)

val_lin = f"{pred_lin:.4f}"
val_mse = f"{mse:.4f}"
val_prob = f"{pred_log_prob * 100:.2f}"
val_acc = f"{acc * 100:.1f}"

with p_col1:
    html_lin = (
        '<div class="metric-card">'
        '<div class="metric-label">Linear Regression Output</div>'
        f'<div class="metric-value" style="color: #FF2E93;">{val_lin}</div>'
        f'<div class="sub-value" style="color: #FF8DA1;">Model Error (MSE): {val_mse}</div>'
        '<div style="color:#6C7D93; font-size:0.75rem; margin-top:0.4rem;">* Predicts continuous numerical points; evaluated via Mean Squared Error (closer to 0 is better).</div>'
        '</div>'
    )
    st.markdown(html_lin, unsafe_allow_html=True)

with p_col2:
    html_log = (
        '<div class="metric-card">'
        '<div class="metric-label">Logistic Regression Probability</div>'
        f'<div class="metric-value" style="color: #00E5FF;">{val_prob}%</div>'
        f'<div class="sub-value" style="color: #80F2FF;">Model Accuracy: {val_acc}%</div>'
        '<div style="color:#6C7D93; font-size:0.75rem; margin-top:0.4rem;">* Classifies success probability; evaluated via Accuracy Score (closer to 100% is better).</div>'
        '</div>'
    )
    st.markdown(html_log, unsafe_allow_html=True)

# ==================== AI 시각화 그래픽스 엔진 ====================
X_range = np.linspace(min_x, max_x, 500).reshape(-1, 1)

# 서버 환경 폰트 깨짐 예방 고딕체 강제 지정
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'text.color': '#8A99AD', 
    'axes.labelcolor': '#8A99AD',
    'xtick.color': '#4A5568',
    'ytick.color': '#4A5568'
})

# --- 첫 번째 그래프 행 (기존 회귀분석 차트) ---
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("#### 📉 선형 회귀 추세선 모델 예측 결과")
    fig1, ax1 = plt.subplots(figsize=(7, 3.8))
    fig1.patch.set_facecolor('#0E1117')
    ax1.set_facecolor('#111625')
    
    ax1.scatter(X_test, y_test, color='#2D3748', alpha=0.6, s=25, label='Validation Data', zorder=2)
    ax1.plot(X_range, lin_model.predict(X_range), color='#FF2E93', linewidth=2.5, label='Linear Trend Line', zorder=3)
    ax1.scatter(user_val, pred_lin, color='#FFD700', edgecolor='#FFFFFF', s=160, marker='o', zorder=5, label='Real-time Input')
    ax1.axvline(user_val, color='#4A5568', linestyle=':', alpha=0.5, zorder=1)
    
    ax1.set_xlabel('Money')
    ax1.set_ylabel('Predicted Value')
    ax1.set_ylim(-0.3, 1.3)
    ax1.grid(True, color='#1A202C', linestyle='--', linewidth=0.8)
    ax1.legend(facecolor='#111625', edgecolor='#232D42', loc='upper left')
    st.pyplot(fig1)
    plt.close(fig1)

with col_g2:
    st.markdown("#### 📈 로지스틱 시그모이드 곡선 모델 예측 결과")
    fig2, ax2 = plt.subplots(figsize=(7, 3.8))
    fig2.patch.set_facecolor('#0E1117')
    ax2.set_facecolor('#111625')
    
    ax2.scatter(X_test, y_test, color='#2D3748', alpha=0.6, s=25, label='Validation Data', zorder=2)
    ax2.plot(X_range, log_model.predict_proba(X_range)[:, 1], color='#00E5FF', linewidth=2.5, label='Sigmoid Curve', zorder=3)
    ax2.axhline(0.5, color='#718096', linestyle='--', linewidth=1, alpha=0.6, label='Decision Boundary (0.5)')
    ax2.scatter(user_val, pred_log_prob, color='#FFD700', edgecolor='#FFFFFF', s=160, marker='o', zorder=5, label='Real-time Input')
    ax2.axvline(user_val, color='#4A5568', linestyle=':', alpha=0.5, zorder=1)
    
    ax2.set_xlabel('Money')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, color='#1A202C', linestyle='--', linewidth=0.8)
    ax2.legend(facecolor='#111625', edgecolor='#232D42', loc='upper left')
    st.pyplot(fig2)
    plt.close(fig2)

# --- 두 번째 그래프 행 (새롭게 추가된 데이터 심층 분석 차트) ---
col_g3, col_g4 = st.columns(2)

with col_g3:
    st.markdown("#### 🔍 선형 예측 오차 분석 (Residuals Scatter Plot)")
    fig3, ax3 = plt.subplots(figsize=(7, 3.8))
    fig3.patch.set_facecolor('#0E1117')
    ax3.set_facecolor('#111625')
    
    y_res_pred = lin_model.predict(X_test)
    residuals = y_test - y_res_pred
    
    ax3.scatter(X_test, residuals, color='#FF2E93', alpha=0.5, s=25, label='Residual Points')
    ax3.axhline(0, color='#FFFFFF', linestyle='-', linewidth=1, alpha=0.4)
    ax3.axvline(user_val, color='#4A5568', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Money')
    ax3.set_ylabel('Residual Error')
    ax3.grid(True, color='#1A202C', linestyle='--', linewidth=0.8)
    ax3.legend(facecolor='#111625', edgecolor='#232D42', loc='upper left')
    st.pyplot(fig3)
    plt.close(fig3)

with col_g4:
    st.markdown("#### 📊 타겟 클래스별 데이터 분포 밀도 (Density Topology Plot)")
    fig4, ax4 = plt.subplots(figsize=(7, 3.8))
    fig4.patch.set_facecolor('#0E1117')
    ax4.set_facecolor('#111625')
    
    df_t0 = df[df['target'] == 0]
    df_t1 = df[df['target'] == 1]
    
    sns.kdeplot(data=df_t0, x='money', fill=True, color='#FF2E93', alpha=0.3, label='Target 0', ax=ax4)
    sns.kdeplot(data=df_t1, x='money', fill=True, color='#00E5FF', alpha=0.3, label='Target 1', ax=ax4)
    ax4.axvline(user_val, color='#FFD700', linestyle='-', linewidth=1.5, label='User Input Point')
    
    ax4.set_xlabel('Money')
    ax4.set_ylabel('Density')
    ax4.grid(True, color='#1A202C', linestyle='--', linewidth=0.8)
    ax4.legend(facecolor='#111625', edgecolor='#232D42', loc='upper left')
    st.pyplot(fig4)
    plt.close(fig4)

# ==================== 요약 성능 리포트 ====================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### ⚖️ 모델 종합 성능 리포트 요약")

summary_matrix = pd.DataFrame({
    "Evaluation Metric": ["Optimization Objective", "Output Data Space", "Binary Classification Fit", "Model Performance Score"],
    "Model 01: Linear Regression": [
        "Minimize Continuous Error",
        "Unbounded (-inf to +inf)",
        "Unsuitable (Mathematical Distortion)",
        f"MSE: {val_mse}"
    ],
    "Model 02: Logistic Regression": [
        "Maximize Log-Likelihood",
        "Bounded (Sigmoid Space: 0 to 1)",
        "Highly Optimal",
        f"Accuracy: {val_acc}%"
    ]
})

st.table(summary_matrix.set_index("Evaluation Metric"))
