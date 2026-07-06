import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, brier_score_loss

# 1. 페이지 설정
st.set_page_config(page_title="인공지능 실시간 모델링", layout="wide")

st.title("4. 인공지능 모델링 및 실시간 예측 (선형 vs 로지스틱)")
st.markdown("사용자가 입력한 값에 따른 모델의 예측 결과를 실시간으로 확인하고, 두 모델의 성능을 비교합니다.")

# 2. 데이터 로드 함수
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("onlineCasino.csv")
        if 'money' not in df.columns or 'outpay' not in df.columns:
            raise KeyError("필수 컬럼이 부족합니다.")
        df['is_high_outpay'] = (df['outpay'] > df['money']).astype(int)
        return df
    except Exception as e:
        # 파일이 없을 경우 안전하게 가상 데이터 생성
        np.random.seed(42)
        mock_X = np.random.uniform(10, 500, 300)
        mock_y = (mock_X > 250).astype(int)
        noise = np.random.choice([0, 1], size=300, p=[0.9, 0.1])
        mock_y = np.abs(mock_y - noise) 
        return pd.DataFrame({'money': mock_X, 'is_high_outpay': mock_y})

df = load_data()
X = df[['money']].values
y = df['is_high_outpay'].values

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 모델 학습
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# ================= 🌟 [추가] 4. 사용자 입력 및 실시간 예측 =================
st.sidebar.header("📥 실시간 예측 입력")
st.sidebar.markdown("아래 슬라이더를 조절하여 모델에게 예측을 시켜보세요!")

# 데이터의 최소/최대값 범위를 기준으로 슬라이더 생성
min_val = int(df['money'].min())
max_val = int(df['money'].max())
user_input = st.sidebar.slider("베팅 금액 (Money) 입력", min_value=min_val, max_value=max_val, value=int((min_val+max_val)/2))

# 사용자 입력값에 대한 예측 수행
user_X = np.array([[user_input]])
user_lin_pred = lin_reg.predict(user_X)[0]
user_log_pred = log_reg.predict(user_X)[0]
user_log_proba = log_reg.predict_proba(user_X)[0][1]

# 상단에 실시간 예측 결과 배너 출력
st.subheader("🔮 실시간 예측 결과")
p_col1, p_col2 = st.columns(2)

with p_col1:
    st.info(f"**📉 선형 회귀의 예측값:** `{user_lin_pred:.2f}`\n\n(0과 1 사이를 벗어날 수 있습니다)")

with p_col2:
    pred_text = "환급금 높음 (1)" if user_log_pred == 1 else "환급금 낮음 (0)"
    st.success(f"**📈 로지스틱 회귀의 예측:** `{pred_text}` (확률: `{user_log_proba*100:.1f}%`) ")
st.markdown("---")
# =========================================================================

# 5. 화면 레이아웃 구성 (2단 컬럼 구성)
col1, col2 = st.columns(2)

# --- 왼쪽: 선형 회귀 컬럼 ---
with col1:
    st.header("📉 선형 회귀 (Linear Regression)")
    st.markdown("연속형 숫자를 예측합니다. 이진 분류에 쓰면 예측값이 범위를 벗어납니다.")
    
    lin_mse = mean_squared_error(y_test, lin_pred)
    lin_r2 = r2_score(y_test, lin_pred)
    
    m1, m2 = st.columns(2)
    m1.metric("MSE (평균제곱오차)", f"{lin_mse:.4f}")
    m2.metric("R² Score (결정계수)", f"{lin_r2:.4f}")
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(X_test, y_test, color='gray', alpha=0.4, label='Actual Data')
    
    sort_idx = np.argsort(X_test.flatten())
    ax1.plot(X_test[sort_idx], lin_pred[sort_idx], color='red', linewidth=2, label='Linear Line')
    
    # [추가] 사용자가 입력한 포인트 그래프에 표시
    ax1.scatter(user_input, user_lin_pred, color='gold', edgecolor='black', s=250, marker='*', zorder=5, label=f'Your Input ({user_input})')
    
    ax1.set_xlabel('Money')
    ax1.set_ylabel('Predicted Value')
    ax1.legend()
    st.pyplot(fig1)

# --- 오른쪽: 로지스틱 회귀 컬럼 ---
with col2:
    st.header("📈 로지스틱 회귀 (Logistic Regression)")
    st.markdown("특정 클래스에 속할 확률을 예측하며, 아름다운 S자 곡선(Sigmoid)을 그립니다.")
    
    log_acc = accuracy_score(y_test, log_pred)
    # 분류 모델을 위한 연속형 지표인 Brier Score도 추가 계산 (비교용)
    log_brier = brier_score_loss(y_test, log_pred_proba)
    
    m3, m4 = st.columns(2)
    m3.metric("Accuracy (정확도)", f"{log_acc * 100:.1f}%")
    m4.metric("Brier Score (예측오차)", f"{log_brier:.4f}")
