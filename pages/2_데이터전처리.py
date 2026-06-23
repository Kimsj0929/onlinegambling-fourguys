import streamlit as st

st.set_page_config(page_title="데이터 전처리", layout="wide")

st.title("2. 데이터 전처리")
st.markdown("결측치 처리, 이상치 처리, 인코딩, 스케일링 등을 구성하는 페이지입니다.")

st.subheader("전처리 체크리스트")
items = [
    "결측치 확인 및 처리",
    "중복 데이터 제거",
    "이상치 탐지",
    "범주형 변수 인코딩",
    "수치형 변수 스케일링",
]

for item in items:
    st.checkbox(item)
