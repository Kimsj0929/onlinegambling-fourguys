import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

# 1. 페이지 설정
st.set_page_config(page_title="인공지능 실시간 예측", layout="wide")

st.title("🤖 인공지능 모델링: 실시간 예측 및 성능 비교")
st.markdown("금액(Money)에 따른 환급 여부(0 또는 1)를 예측하는 모델입니다. 직접 값을 조절하며 두 모델의 차이를 확인하세요.")

# 2. 데이터 로드 및 전처리
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("onlineCasino.csv")
        df['is_high_outpay'] = (df['outpay'] > df['money']).astype(int)
        return df
    except:
        # 데이터가 없을 경우를 대비한 학습용 가상 데이터 생성
        np.random.seed(42)
        X_mock = np.random.uniform(50, 500, 200)
        y_mock = (X_mock > 250).astype(int)
        # 약간의 오차 추가
        noise = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
        y_mock = np.abs(y_mock - noise)
        return pd.DataFrame({'money': X_mock, 'is_high_outpay': y_mock})

df = load_data()
X = df[['money']].values
y = df['is_high_outpay'].values

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습 (한 번만 실행)
lin_reg = LinearRegression().fit(X_train, y_train)
log_reg = LogisticRegression().fit(X_train, y_train)

# ==================== 🎮 4. 실시간 사용자 입력 세션 ====================
st.markdown("---")
st.header("🎮 직접 값을 조절해보세요!")
# 사용자가 값을 조정할 수 있는 슬라이더 (중앙에 배치)
min_x, max_x = float(X.min()), float(X.max())
user_value = st.slider("베팅 금액(Money)을 조절하세요:", min_x, max_x, float(X.mean()))

# 사용자 입력값에 대한 실시간 예측 계산
user_X = np.array([[user_value]])
res_lin = lin_reg.predict(user_X)[0]
res_log_prob = log_reg.predict_proba(user_X)[0][1]
res_log_class = 1 if res_log_prob >= 0.5 else 0

# 예측 결과 요약 배너
c1, c2 = st.columns(2)
with c1:
    st.metric("📉 선형 회귀 예측값", f"{res_lin:.2f}")
    if res_lin > 1 or res_lin < 0:
        st.warning("⚠️ 예측값이 0과 1의 범위를 벗어났습니다! (선형 회귀의 한계)")
with c2:
    status = "성공(1)" if res_log_class == 1 else "실패(0)"
    st.metric("📈 로지스틱 확률", f"{res_log_prob*100:.1f}%", f"결과: {status}")
st.markdown("---")

# ==================== 📊 5. 그래프 시각화 세션 ====================
col1, col2 = st.columns(2)

# 시각화를 위한 부드러운 곡선용 데이터 생성 (S자 곡선을 제대로 그리기 위함)
X_range = np.linspace(min_x, max_x, 300).reshape(-1, 1)
lin_line = lin_reg.predict(X_range)
log_curve = log_reg.predict_proba(X_range)[:, 1]

# --- 왼쪽: 선형 회귀 그래프 ---
with col1:
    st.subheader("📉 선형 회귀 결과")
    fig1, ax1 = plt.subplots()
    ax1.scatter(X_test, y_test, color='gray', alpha=0.3, label='Actual')
    ax1.plot(X_range, lin_line, color='red', label='Linear Model') # 예측 직선
    # 사용자 입력 위치 표시 (노란 별)
    ax1.scatter(user_value, res_lin, color='orange', marker='*', s=300, zorder=5, label='Your Input')
    ax1.set_ylim(-0.2, 1.2) # 범위 고정
    ax1.legend()
    st.pyplot(fig1)

# --- 오른쪽: 로지스틱 회귀 그래프 ---
with col2:
    st.subheader("📈 로지스틱 회귀 결과")
    fig2, ax2 = plt.subplots()
    ax2.scatter(X_test, y_test, color='gray', alpha=0.3, label='Actual')
    ax2.plot(X_range, log_curve, color='blue', label='Logistic Model (S-Curve)') # 예측 S곡선
    ax2.axhline(0.5, color='green', linestyle='--', alpha=0.5) # 기준선
    # 사용자 입력 위치 표시 (노란 별)
    ax2.scatter(user_value, res_log_prob, color='orange', marker='*', s=300, zorder=5, label='Your Input')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    st.pyplot(fig2)

# ==================== 🏆 6. 모델 성능 비교 세션 ====================
st.header("🏆 어떤 모델이 더 똑똑할까요?")

# 성능 지표 계산
mse = mean_squared_error(y_test, lin_reg.predict(X_test))
acc = accuracy_score(y_test, log_reg.predict(X_test))

m_col1, m_col2, m_col3 = st.columns(3)
m_col1.info(f"**선형 회귀 MSE (오차)**\n\n {mse:.4f}")
m_col2.success(f"**로지스틱 정확도**\n\n {acc*100:.1f}%")
m_col3.write("**💡 모델링 가이드**\n이진 분류(0/1) 문제에서는 **로지스틱 회귀**가 확률을 0~1 사이로 제한해주기 때문에 훨씬 정확하고 안정적입니다.")

# 비교표
comparison = {
    "특징": ["예측 형태", "값의 범위", "분류 방식", "적합도"],
    "선형 회귀": ["곧은 직선", "제한 없음 (범위 이탈)", "단순 수치 계산", "낮음"],
    "로지스틱 회귀": ["부드러운 S자 곡선", "0 ~ 1 사이 고정", "확률 기반 분류", "매우 높음"]
}
st.table(pd.DataFrame(comparison))
