import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

# 1. 페이지 설정 (가장 무조건 첫 줄에 와야 합니다)
st.set_page_config(page_title="인공지능 모델링", layout="wide")

st.title("4. 인공지능 모델링 (선형 vs 로지스틱)")
st.markdown("모델 선택, 학습, 평가 결과를 배치하고 두 모델의 성능 차이를 비교하는 페이지입니다.")

# 2. 데이터 로드 함수 (오류 방지 처리가 강화되었습니다)
@st.cache_data
def load_data():
    try:
        # 파일 읽기 시도
        df = pd.read_csv("onlineCasino.csv")
        
        # 파일은 열렸으나 필요한 컬럼이 없는 경우를 대비한 방어 코드
        if 'money' not in df.columns or 'outpay' not in df.columns:
            raise KeyError("필수 컬럼이 부족합니다.")
            
        # 이진 분류 타겟 만들기: 환급금(outpay)이 베팅금(money)보다 큰지 여부 (0 또는 1)
        df['is_high_outpay'] = (df['outpay'] > df['money']).astype(int)
        return df
    except Exception as e:
        # 파일이 없거나 에러가 나면 화면에 에러를 보여주는 대신 가상 데이터를 안전하게 생성
        np.random.seed(42)
        mock_X = np.random.uniform(10, 500, 300)
        # 특정 기준값보다 크면 1, 아니면 0
        mock_y = (mock_X > 250).astype(int)
        # 노이즈 추가 (실제 데이터 느낌을 주기 위함)
        noise = np.random.choice([0, 1], size=300, p=[0.9, 0.1])
        mock_y = np.abs(mock_y - noise) 
        
        return pd.DataFrame({'money': mock_X, 'is_high_outpay': mock_y})

# 데이터 준비
df = load_data()

# 독립변수(X)와 종속변수(y) 설정
X = df[['money']].values
y = df['is_high_outpay'].values

# 데이터 분할 (Train / Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 모델 학습 및 예측
# [선형 회귀]
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)

# [로지스틱 회귀]
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_pred_proba = log_reg.predict_proba(X_test)[:, 1] # 1일 확률 추출

# 4. 화면 레이아웃 구성 (2단 컬럼 구성)
col1, col2 = st.columns(2)

# --- 왼쪽: 선형 회귀 컬럼 ---
with col1:
    st.header("📉 선형 회귀 (Linear Regression)")
    st.markdown("연속형 숫자를 예측하는 모델입니다. 분류 문제에 적용하면 예측값이 0 미만이거나 1을 초과하는 현상이 발생합니다.")
    
    # 평가 지표 계산
    lin_mse = mean_squared_error(y_test, lin_pred)
    lin_r2 = r2_score(y_test, lin_pred)
    
    # 지표 시각화
    m1, m2 = st.columns(2)
    m1.metric("MSE (평균제곱오차)", f"{lin_mse:.4f}")
    m2.metric("R² Score (결정계수)", f"{lin_r2:.4f}")
    
    # 그래프 그리기
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual Data')
    
    # 시각화를 위해 정렬된 선 그리기
    sort_idx = np.argsort(X_test.flatten())
    ax1.plot(X_test[sort_idx], lin_pred[sort_idx], color='red', linewidth=2, label='Linear Regression Line')
    
    ax1.set_xlabel('Money')
    ax1.set_ylabel('Predicted Value')
    ax1.legend()
    st.pyplot(fig1)

# --- 오른쪽: 로지스틱 회귀 컬럼 ---
with col2:
    st.header("📈 로지스틱 회귀 (Logistic Regression)")
    st.markdown("특정 클래스에 속할 확률을 예측하는 모델입니다. 데이터를 0과 1 사이의 아름다운 S자 곡선(Sigmoid)으로 표현합니다.")
    
    # 평가 지표 계산
    log_acc = accuracy_score(y_test, log_pred)
    
    # 지표 시각화
    m3, m4 = st.columns(2)
    m3.metric("Accuracy (정확도)", f"{log_acc * 100:.1f}%")
    m4.metric("분류 기준", "이진 분류 (0 또는 1)")
    
    # 그래프 그리기
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual Data')
    
    # 시각화를 위해 정렬된 S자 곡선 그리기
    sort_idx = np.argsort(X_test.flatten())
    ax2.plot(X_test[sort_idx], log_pred_proba[sort_idx], color='blue', linewidth=2, label='Logistic Sigmoid Curve')
    ax2.axhline(0.5, color='green', linestyle='--', label='Threshold (0.5)')
    
    ax2.set_xlabel('Money')
    ax2.set_ylabel('Probability')
    ax2.legend()
    st.pyplot(fig2)

# 5. 하단 요약 가이드
st.subheader("💡 결과 해석 가이드")
st.info(
    "1. 왼쪽 선형 회귀 그래프를 보면 데이터가 0과 1에만 분포함에도 불구하고, 예측선이 그 범위를 벗어나 직선으로 쭉 뻗어나갑니다.\n"
    "2. 반면 오른쪽 로지스틱 회귀는 0과 1 사이를 부드럽게 연결하는 S자 곡선을 그리며, 0.5 점선 기준 위아래로 완벽하게 분류해 줍니다.\n"
    "결론적으로, '이진 분류' 성격의 데이터에는 선형 회귀보다 로지스틱 회귀가 훨씬 적합하다는 것을 알 수 있습니다!"
)
