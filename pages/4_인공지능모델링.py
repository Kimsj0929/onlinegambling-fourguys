import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

# 페이지 설정
st.set_page_config(page_title="인공지능 모델링 비교", layout="wide")

st.title("4. 인공지능 모델링 (회귀 vs 분류)")
st.markdown("""
이 페이지는 **선형 회귀(Linear Regression)**와 **로지스틱 회귀(Logistic Regression)**의 작동 방식과 성능 차이를 직관적으로 비교하는 페이지입니다.
""")

# 1. 데이터 로드 (업로드된 데이터 가정)
@st.cache_data
def load_data():
    # 실제 환경에 맞게 경로 수정 가능 (여기서는 예시 데이터 생성 또는 로드)
    try:
        df = pd.read_csv("onlineCasino.csv")
        # 데모를 위해 돈(money)에 따른 특정 조건(예: outpay가 money보다 큰지 여부)을 이진 타겟으로 설정
        df['is_high_outpay'] = (df['outpay'] > df['money']).astype(int)
        return df
    except:
        # 파일이 없을 경우를 대비한 가상 데이터
        np.random.seed(42)
        X = np.random.uniform(10, 500, 200)
        y = (X > 250).astype(int)
        return pd.DataFrame({'money': X, 'is_high_outpay': y})

df = load_data()

# 2. 모델 학습 준비 (독립변수: money, 종속변수: is_high_outpay)
X = df[['money']].values
y = df['is_high_outpay'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 생성 및 학습
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 예측
lin_pred = lin_reg.predict(X_test)
log_pred = log_reg.predict(X_test)
log_pred_proba = log_reg.predict_proba(X_test)[:, 1] # 확률 값

# --- 화면 레이아웃 구성 ---

# 사이드바 또는 상단에서 target 설명
st.sidebar.header("설정 및 안내")
st.sidebar.write("📌 **타겟 변수**: 고액 환급 여부 (0 또는 1)")
st.sidebar.write("선형 회귀는 이진 데이터도 직선으로 예측하는 반면, 로지스틱 회귀는 S자 곡선(Sigmoid)으로 확률을 예측합니다.")

# 메인 화면: 두 모델 비교 레이아웃
col1, col2 = st.columns(2)

# 왼쪽 컬럼: 선형 회귀 (Linear Regression)
with col1:
    st.header("📉 선형 회귀 (Linear Regression)")
    st.markdown("연속적인 값을 예측하는 모델로, 이진 분류에 적용 시 0 미만이거나 1을 초과하는 예측 값이 나올 수 있습니다.")
    
    # 평가 지표
    lin_mse = mean_squared_error(y_test, lin_pred)
    lin_r2 = r2_score(y_test, lin_pred)
    
    m1, m2 = st.columns(2)
    m1.metric("MSE (평균제곱오차)", f"{lin_mse:.4f}")
    m2.metric("R² Score (결정계수)", f"{lin_r2:.4f}")
    
    # 시각화
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual')
    # 정렬해서 그리기
    sort_idx = np.argsort(X_test.flatten())
    ax.plot(X_test[sort_idx], lin_pred[sort_idx], color='red', linewidth=2, label='Linear Trend')
    ax.set_xlabel('Money')
    ax.set_ylabel('Prediction')
    ax.legend()
    st.pyplot(fig)

# 오른쪽 컬럼: 로지스틱 회귀 (Logistic Regression)
with col2:
    st.header("📈 로지스틱 회귀 (Logistic Regression)")
    st.markdown("데이터가 특정 클래스(0 또는 1)에 속할 **확률**을 0과 1 사이의 S자 곡선(Sigmoid)으로 예측합니다.")
    
    # 평가 지표 (분류 모델이므로 Accuracy 측정)
    log_acc = accuracy_score(y_test, log_pred)
    
    m1, m2 = st.columns(2)
    m1.metric("Accuracy (정확도)", f"{log_acc * 100:.1f}%")
    m2.metric("Target Type", "Binary (0 or 1)")
    
    # 시각화
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual')
    # S자 곡선 그리기 위해 정렬
    sort_idx = np.argsort(X_test.flatten())
    ax.plot(X_test[sort_idx], log_pred_proba[sort_idx], color='blue', linewidth=2, label='Logistic Sigmoid')
    ax.axhline(0.5, color='green', linestyle='--', label='Threshold (0.5)')
    ax.set_xlabel('Money')
    ax.set_ylabel('Probability')
    ax.legend()
    st.pyplot(fig)

---
st.subheader("💡 결과 해석 가이드")
st.info("""
* **선형 회귀** 그래프를 보면 데이터가 0과 1 사이에만 존재함에도 불구하고 직선이 범위를 벗어나 뻗어나가는 것을 볼 수 있습니다. 회귀 분석에는 좋지만 분류에는 부적합하다는 것을 직관적으로 알 수 있습니다.
* **로지스틱 회귀**는 그래프가 딱 0과 1 사이에서 아름다운 S자 곡선을 그리며, 0.5(Threshold)를 기준으로 데이터를 어떻게 분류하는지 명확히 보여줍니다.
""")
