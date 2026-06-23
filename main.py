import streamlit as st

st.set_page_config(
    page_title="Streamlit Website Template",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Navigation")
st.sidebar.caption("프로젝트 페이지를 선택하세요")
st.sidebar.radio(
    "Menu",
    ["Home", "Problem", "Preprocess", "Visualize", "Modeling"],
    label_visibility="collapsed",
)

st.title("Streamlit 웹사이트 기본 템플릿")
st.markdown(
    """
    GitHub에 바로 올릴 수 있는 기본 구조입니다.
    왼쪽 사이드바에서 페이지를 선택하고, 각 단계별 내용을 채워 넣으면 됩니다.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("문제 정의")
    st.write("프로젝트 목표, 배경, 핵심 질문을 정리합니다.")

with col2:
    st.subheader("데이터 파이프라인")
    st.write("수집, 전처리, 탐색, 학습 순서로 구성합니다.")

with col3:
    st.subheader("모델링")
    st.write("기본 모델, 평가 지표, 결과 해석을 배치합니다.")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("프로젝트 개요")
    st.write("이 화면은 Streamlit 앱의 메인 랜딩 페이지 역할을 합니다.")
    st.write("여기에 소개 문구, 버튼, 이미지, 표, 차트 등을 추가할 수 있습니다.")

with right:
    st.subheader("Quick Start")
    st.code("pip install -r requirements.txt")
    st.code("streamlit run main.py")
