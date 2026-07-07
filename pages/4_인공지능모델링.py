import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. 글로벌 레이아웃 설정
st.set_page_config(page_title="NEXUS Quantum AI Modeling", layout="wide")

# CSS 주입 (메인 타이틀 색상을 어두운 블랙/차콜 계열로 설정)
def inject_custom_css():
    css_content = (
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');"
        "html, body, [data-testid='stMarkdownContainer'] { font-family: 'Noto Sans KR', sans-serif; }"
        ".main-title { font-size: 2.2rem; font-weight: 700; color: #111625; margin-bottom: 0.5rem; }"
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
st.markdown('<div class="main-title">⚠️ 우리가 도박의 늪에 빠지면 안 되는 이유</div>', unsafe_allow_html=True)
desc_text = (
    "본 프로젝트는 단순한 게임을 넘어 일상을 파괴하는 청소년 도박의 치명적인 심각성을 인지하고, "
    "실태조사 데이터를 바탕으로 도박 위험군을 조기에 예측 및 분류하기 위해 구축된 데이터 분석 웹 대시보드입니다.<br><br>"
    "<b>'한 번쯤은 괜찮겠지'</b>라는 호기심이 어떻게 헤어 나올 수 없는 중독과 위험으로 이어지는지 "
    "데이터가 증명하는 진실을 마주해 보세요. 좌측 사이드바의 메뉴를 통해 데이터 정제부터 모델링까지, "
    "도박이 우리 삶을 잠식해 가는 전 과정을 과학적으로 확인할 수 있습니다."
)
st.markdown('<div class="sub-title">' + desc_text + '</div>', unsafe_allow_html=True)

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
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)

# 평가지표 산출
mse = mean_squared_error(y_test, lin_model.predict(X_test))
acc_log = accuracy_score(y_test, log_model.predict(X_test))
acc_rf = accuracy_score(y_test, rf_model.predict(X_test))

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
pred_rf_prob = rf_model.predict_proba([[user_val]])[0][1]

st.markdown("<br>", unsafe_allow_html=True)

# ==================== 세 파트로 분리된 스코어 및 평가지표 메트릭 ====================
p_col1, p_col2, p_col3 = st.columns(3)

val_lin = f"{pred_lin:.4f}"
val_mse = f"{mse:.4f}"
val_log_prob = f"{pred_log_prob * 100:.2f}"
val_acc_log = f"{acc_log * 100:.1f}"
val_rf_prob = f"{pred_rf_prob * 100:.2f}"
val_acc_rf = f"{acc_rf * 100:.1f}"

# 매직 파서 우회를 위해 미리 문자열 결합 처리
str_lin_mse = "Model Error (MSE): " + val_mse
str_log_acc = "Model Accuracy: " + val_acc_log + "%"
str_log_prob = val_log_prob + "%"
str_rf_acc = "Model Accuracy: " + val_acc_rf + "%"
str_rf_prob = val_rf_prob + "%"

with p_col1:
    html_lin = (
        '<div class="metric-card">'
        '<div class="metric-label">Linear Regression Output</div>'
        '<div class="metric-value" style="color: #FF2E93;">' + val_lin + '</div>'
        '<div class="sub-value" style="color: #FF8DA1;">' + str_lin_mse + '</div>'
        '<div style="color:#6C7D93; font-size:0.75rem; margin-top:0.4rem;">* 연속 수치 예측 결과값; 상하한선 경계가 없음.</div>'
        '</div>'
    )
    st.markdown(html_lin, unsafe_allow_html=True)

with p_col2:
    html_log = (
        '<div class="metric-card">'
        '<div class="metric-label">Logistic Regression Probability</div>'
        '<div class="metric-value" style="color: #00E5FF;">' + str_log_prob + '</div>'
        '<div class="sub-value" style="color: #80F2FF;">' + str_log_acc + '</div>'
        '<div style="color:#6C7D93; font-size:0.75rem; margin-top:0.4rem;">* 통계적 시그모이드 기반 위험 확률 추정값.</div>'
        '</div>'
    )
    st.markdown(html_log, unsafe_allow_html=True)

with p_col3:
    html_rf = (
        '<div class="metric-card">'
        '<div class="metric-label">Random Forest Probability</div>'
        '<div class="metric-value" style="color: #FFD700;">' + str_rf_prob + '</div>'
        '<div class="sub-value" style="color: #FFE680;">' + str_rf_acc + '</div>'
        '<div style="color:#6C7D93; font-size:0.75rem; margin-top:0.4rem;">* 머신러닝 의사결정나무 앙상블 기반 위험 확률 추정값.</div>'
        '</div>'
    )
    st.markdown(html_rf, unsafe_allow_html=True)

# ==================== AI 시각화 그래픽스 엔진 ====================
X_range = np.linspace(min_x, max_x, 500).reshape(-1, 1)

plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'text.color': '#8A99AD', 
    'axes.labelcolor': '#8A99AD',
    'xtick.color': '#4A5568',
    'ytick.color': '#4A5568'
})

# --- 첫 번째 그래프 행 (선형 회귀 그래프 단독 배치) ---
st.markdown("#### 📉 선형 회귀 추세선 모델 예측 결과")
fig1, ax1 = plt.subplots(figsize=(14, 3.5))
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

# --- 두 번째 그래프 행 (새로 추가된 로지스틱 회귀 시그모이드 확률 곡선 그래프) ---
st.markdown("#### 📈 로지스틱 회귀 시그모이드(Sigmoid) 위험 확률 곡선")
fig2, ax2 = plt.subplots(figsize=(14, 3.5))
fig2.patch.set_facecolor('#0E1117')
ax2.set_facecolor('#111625')

ax2.scatter(X_test, y_test, color='#2D3748', alpha=0.6, s=25, label='Validation Data', zorder=2)
ax2.plot(X_range, log_model.predict_proba(X_range)[:, 1], color='#00E5FF', linewidth=2.5, label='Logistic Probability Curve', zorder=3)
ax2.axhline(0.5, color='#718096', linestyle='--', linewidth=1, alpha=0.6, label='Decision Boundary (0.5)')
ax2.scatter(user_val, pred_log_prob, color='#FFD700', edgecolor='#FFFFFF', s=160, marker='o', zorder=5, label='Real-time Probability')
ax2.axvline(user_val, color='#4A5568', linestyle=':', alpha=0.5, zorder=1)

ax2.set_xlabel('Money')
ax2.set_ylabel('Gambling Risk Probability')
ax2.set_ylim(-0.1, 1.1)
ax2.grid(True, color='#1A202C', linestyle='--', linewidth=0.8)
ax2.legend(facecolor='#111625', edgecolor='#232D42', loc='upper left')
st.pyplot(fig2)
plt.close(fig2)

# --- 세 번째 그래프 행 (심층 오차 분석 및 타겟 분포 밀도) ---
col_g3, col_g4 = st.columns(2)

with col_g3:
    st.markdown("#### 🔍 선형 예측 오차 분석 (Residuals Scatter Plot)")
    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
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
    fig4, ax4 = plt.subplots(figsize=(7, 3.5))
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

score_lin = "MSE: " + val_mse
score_log = "Accuracy: " + val_acc_log + "%"
score_rf = "Accuracy: " + val_acc_rf + "%"

summary_matrix = pd.DataFrame({
    "Evaluation Metric": ["Optimization Objective", "Output Data Space", "Binary Classification Fit", "Model Performance Score"],
    "Model 01: Linear Regression": [
        "Minimize Continuous Error",
        "Unbounded (-inf to +inf)",
        "Unsuitable (Mathematical Distortion)",
        score_lin
    ],
    "Model 02: Logistic Regression": [
        "Maximize Log-Likelihood",
        "Bounded (Sigmoid Space: 0 to 1)",
        "Highly Optimal",
        score_log
    ],
    "Model 03: Random Forest": [
        "Maximize Information Gain (Gini)",
        "Discrete Ensemble Probability (0 to 1)",
        "Excellent (Captures Complex Non-linear Rules)",
        score_rf
    ]
})

st.table(summary_matrix.set_index("Evaluation Metric"))
