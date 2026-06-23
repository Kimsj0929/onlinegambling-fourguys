import streamlit as st

st.set_page_config(page_title="인공지능 모델링", layout="wide")

st.title("4. 인공지능 모델링")
st.markdown("모델 선택, 학습, 평가 결과를 배치하는 페이지입니다.")

model = st.selectbox(
    "모델 선택",
    ["Linear Regression", "Random Forest", "XGBoost", "Neural Network"]
)
st.write(f"선택한 모델: {model}")

st.subheader("평가 결과")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "0.91")
col2.metric("F1 Score", "0.88")
col3.metric("Loss", "0.12")
