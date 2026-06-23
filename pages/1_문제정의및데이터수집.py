import streamlit as st

st.set_page_config(page_title="문제정의, 데이터 수집", layout="wide")

st.title("1. 문제정의, 데이터 수집")
st.markdown("프로젝트의 목적과 데이터를 어떻게 모을지 정리하는 페이지입니다.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("문제 정의")
    st.write("- 해결하려는 문제를 한 문장으로 정의합니다.")
    st.write("- 대상 사용자와 기대 효과를 적습니다.")

with col2:
    st.subheader("데이터 수집")
    st.write("- 사용 데이터의 출처를 정리합니다.")
    st.write("- 크롤링, API, 파일 업로드 등 수집 방법을 기록합니다.")
