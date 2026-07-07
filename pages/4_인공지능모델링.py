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
    outpay[12] = 2200
    outpay[85] = -600
    outpay[150] = 1800
    
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
        Q1 = df_c[col].quantile(0.25)
        Q3 = df_c[col].quantile(0.75)
        IQR = Q3 - Q1
        df_c = df_c[(df_c[col] >= Q1 - weight * IQR) & (df_c[col] <= Q3 + weight * IQR)]
    return df_c

df_clean = preprocess_dynamic(df_raw, iqr_weight)

# 메트릭 표시
m1, m2, m3 = st.columns(3)
m1.metric("Raw Rows", f"{len(df_raw)}")
m2.metric("Cleaned Rows", f"{len(df_clean)}")
m3.metric("Removed", f"{len(df_raw) - len(df_clean)}", delta_color="inverse")

# 분석 그래프 (히트맵 & 박스플롯)
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 10, 'text.color': '#8A99AD', 'axes.labelcolor': '#8A99AD'})

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("##### Correlation Heatmap (Cleaned)")
    fig_h, ax_h = plt.subplots(figsize=(6, 4))
    fig_h.patch.set_facecolor('#0E1117')
    ax_h.set_facecolor('#111625')
    features = ['gamers', 'skins', 'money', 'ticks', 'outpay']
    sns.heatmap(df_clean[features].corr(), annot=True, cmap='vlag', fmt=".2f", ax=ax_h, cbar=False)
    ax_h.set_xticklabels(['Gamers', 'Skins', 'Money', 'Ticks', 'Outpay'])
    ax_h.set_yticklabels(['Gamers', 'Skins', 'Money', 'Ticks', 'Outpay'])
    st.pyplot(fig_h)
    plt.close()

with col_b:
    st.markdown("##### Outlier Distribution (Raw vs Clean)")
    fig_bx, ax_bx = plt.subplots(figsize=(6, 4))
    fig_bx.patch.set_facecolor('#0E1117')
    ax_bx.set_facecolor('#111625')
    sns.boxplot(data=df_raw[['money', 'outpay']], palette='husl', ax=ax_bx)
    ax_bx.set_xticklabels(['Money', 'Outpay'])
    ax_bx.set_title("Raw Data Range")
    st.pyplot(fig_bx)
    plt.close()

# 산점도 섹션 (문법 오류 유발 세미콜론 라인 전면 수정)
st.markdown("##### Data Distribution Scatter Plot")
fig_sc, ax_sc = plt.subplots(figsize=(12, 4))
fig_sc.patch.set_facecolor('#0E1117')
ax_sc.set_facecolor('#111625')
ax_sc.scatter(df_raw['money'], df_raw['outpay'], color='#FF5252', alpha=0.3, label='Outliers', s=40)
ax_sc.scatter(df_clean['money'], df_clean['outpay'], color='#00E676', alpha=0.7, label='Cleaned', s=20)
ax_sc.set_xlabel('Money')
ax_sc.set_ylabel('Outpay')
ax_sc.legend()
ax_sc.grid(alpha=0.1)
st.pyplot(fig_sc)
plt.close()

# ==================== STEP 2. AI 모델링 섹션 ====================
st.markdown('<div class="section-header">STEP 2. Quantum AI Modeling (Regression vs Classification)</div>', unsafe_allow_html=True)

# 모델 학습 데이터 준비 (정제된 데이터 기반)
X = df_clean[['money']].values
y = df_clean['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
lin_model = LinearRegression().fit(X_train, y_train)
log_model = LogisticRegression().fit(X_train, y_train)

# 실시간 예측 컨트롤러
user_val = st.slider("Predict for Money Value", float(X.min()), float(X.max()), float(X.mean()))
pred_lin = lin_model.predict([[user_val]])[0]
pred_log_prob = log_model.predict_proba([[user_val]])[0][1]

# 메트릭 카드
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Linear Regression Prediction</div>
    <div class="metric-value">{pred_lin:.4f}</div><div style='color:#6C7D93; font-size:0.7rem;'>* Based on Linear Trend</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Logistic Probability</div>
    <div class="metric-value">{pred_log_prob*100:.2f}%</div><div style='color:#00E676; font-size:0.7rem;'>* Class: {"High" if pred_log_prob >= 0.5 else "Low"}</div></div>""", unsafe_allow_html=True)

# 모델 그래프 (Linear vs Logistic)
col_m1, col_m2 = st.columns(2)
X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

with col_m1:
    st.markdown("##### Linear Regression Line")
    fig_l1, ax_l1 = plt.subplots(figsize=(6, 4))
    fig_l1.patch.set_facecolor('#0E1117')
    ax_l1.set_facecolor('#111625')
    ax_l1.scatter(X_test, y_test, color='#2D3748', alpha=0.5, label='Actual')
    ax_l1.plot(X_range, lin_model.predict(X_range), color='#FF2E93', lw=3, label='Linear Line')
    ax_l1.scatter(user_val, pred_lin, color='gold', s=150, edgecolors='white', zorder=5)
    ax_l1.set_xlabel('Money')
    ax_l1.set_ylim(-0.2, 1.2)
    ax_l1.legend()
    st.pyplot(fig_l1)
    plt.close()

with col_m2:
    st.markdown("##### Logistic Sigmoid Curve")
    fig_l2, ax_l2 = plt.subplots(figsize=(6, 4))
    fig_l2.patch.set_facecolor('#0E1117')
    ax_l2.set_facecolor('#111625')
    ax_l2.scatter(X_test, y_test, color='#2D3748', alpha=0.5, label='Actual')
    ax_l2.plot(X_range, log_model.predict_proba(X_range)[:, 1], color='#00E5FF', lw=3, label='Sigmoid Curve')
    ax_l2.scatter(user_val, pred_log_prob, color='gold', s=150, edgecolors='white', zorder=5)
    ax_l2.axhline(0.5, color='white', ls='--', alpha=0.3)
    ax_l2.set_xlabel('Money')
    ax_l2.set_ylim(-0.1, 1.1)
    ax_l2.legend()
    st.pyplot(fig_l2)
    plt.close()

# 성능 요약 테이블
st.markdown("##### AI Model Comparison Summary")
summary_data = pd.DataFrame({
    "Metric": ["Evaluation Objective", "Mathematical Shape", "Performance Score"],
    "Linear Regression": ["Error Minimization", "Straight Line", f"MSE: {mean_squared_error(y_test, lin_model.predict(X_test)):.4f}"],
    "Logistic Regression": ["Probability Maximization", "S-Curve (Sigmoid)", f"Accuracy: {accuracy_score(y_test, log_model.predict(X_test))*100:.1f}%"]
})
st.table(summary_data.set_index("Metric"))
