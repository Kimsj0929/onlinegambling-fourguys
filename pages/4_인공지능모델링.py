import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. 글로벌 레이아웃 설정
st.set_page_config(page_title="NEXUS 퀀텀 AI | 분석 대시보드", layout="wide")

# CSS 주입 (안정적인 커스텀 스타일)
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

# 2. 데이터 세트 로드 및 분석 파이프라인
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
res_log_class = 1 if res_log_prob >= 0.5 else 0

st.markdown("<br>", unsafe_allow_html=True)

# ==================== 실시간 분석 메트릭 시각화 (요청사항 반영 세션) ====================
p_col1, p_col2 = st.columns(2)

with p_col1:
    if res_lin < 0 or res_lin > 1:
        desc_text = "<span style='color:#FF5252;'>⚠️ <b>범위 초과 오류</b>: 선형 모델의 한계로 인해 불가능한 스코어가 출력되었습니다.</span>"
    else:
        desc_text = "✅ <b>수치 해석</b>: 단순 정비례 직선 수식으로만 계산된 연속형 위치 점수입니다."
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📉 선형 회귀 예측 출력값 & 모델 오차</div>
        <div class="metric-value">{res_lin:.4f}</div>
        <div class="sub-value">📊 모델 전체 오차 (MSE): {mse:.4f}</div>
        <div class="metric-desc">{desc_text}<br><span style='color:#6C7D93;'>* 연속적인 수치를 예측하므로 정확도 대신 오차 지표(MSE)를 사용하며, 0에 가까울수록 정밀합니다.</span></div>
    </div>
    """, unsafe_allow_html=True)

with p_col2:
    if res_log_class == 1:
        status_color = "#00E676"
        status_text = "🎯 최종 판정: 환급 성공 예측 (1)"
    else:
        status_color = "#FF5252"
        status_text = "💀 최종 판정: 환급 실패 예측 (0)"
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📈 로지스틱 연산 확률 & 분류 정확도</div>
        <div class="metric-value">{res_log_prob * 100:.2f}%</div>
        <div class="sub-value" style="color:#00E5FF;">🎯 모델 최종 정확도 (Accuracy): {acc*100:.1f}%</div>
        <div class="metric-desc" style="color:{status_color}; font-weight:600;">{status_text}</div>
        <div class="metric-desc" style="margin-top:0px;"><span style='color:#6C7D93;'>* 성공/실패 맞춤 여부를 판정하므로 정확도(Accuracy)를 지표로 사용하며, 100%에 가까울수록 완벽합니다.</span></div>
    </div>
    """, unsafe_allow_html=True)

# ==================== 시각화 분석 그래픽스 엔진 ====================
col1, col2 = st.columns(2)

X_range = np.linspace(min_x, max_x, 500).reshape(-1, 1)
lin_line = lin_reg.predict(X_range)
log_curve = log_reg.predict_proba(X_range)[:, 1]

plt.style.use('dark_background')
plt.rcParams['text.color'] = '#8A99AD'
plt.rcParams['axes.labelcolor'] = '#8A99AD'
plt.rcParams['xtick.color'] = '#4A5568'
plt.rcParams['ytick.color'] = '#4A5568'

# --- 왼쪽 그래프: 선형 회귀 ---
with col1:
    st.markdown("#### 📉 선형 회귀 모델 예측 직선")
    fig1, ax1 = plt.subplots(figsize=(7, 4.2))
    fig1.patch.set_facecolor('#0E1117')
    ax1.set_facecolor('#111625')
    
    ax1.scatter(X_test, y_test, color='#2D3748', alpha=0.6, s=25, label='실제 검증 데이터(Empirical)', zorder=2)
    ax1.plot(X_range, lin_line, color='#FF2E93', linewidth=2.5, label='선형 예측 추세선', zorder=3)
    
    ax1.scatter(user_value, res_lin, color='#FFD700', edgecolor='#FFFFFF', s=160, marker='o', zorder=5, label='실시간 입력 위치')
    ax1.axvline(user_value, color='#4A5568', linestyle=':', alpha=0.5, zorder=1)
    
    ax1.set_ylim(-0.3, 1.3)
    ax1.grid(True, color='#1A202C', linestyle='--', linewidth=0.8)
    ax1.legend(facecolor='#111625', edgecolor='#232D42', loc='upper left')
    st.pyplot(fig1)

# --- 오른쪽 그래프: 로지스틱 회귀 ---
with col2:
    st.markdown("#### 📈 로지스틱 분류 확률 시그모이드 곡선")
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    fig2.patch.set_facecolor('#0E1117')
    ax2.set_facecolor('#111625')
    
    ax2.scatter(X_test, y_test, color='#2D3748', alpha=0.6, s=25, label='실제 검증 데이터(Empirical)', zorder=2)
    ax2.plot(X_range, log_curve, color='#00E5FF', linewidth=2.5, label='시그모이드 최적 곡선', zorder=3)
    ax2.axhline(0.5, color='#718096', linestyle='--', linewidth=1, alpha=0.6, label='분류 결정 임계선 (0.5)')
    
    ax2.scatter(user_value, res_log_prob, color='#FFD700', edgecolor='#FFFFFF', s=160, marker='
