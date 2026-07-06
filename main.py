import streamlit as st

st.set_page_config(
    page_title="청소년 도박 데이터 분석 프로젝트",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 사이드바 내비게이션 설정
st.sidebar.title("🧭 내비게이션")
st.sidebar.caption("분석 단계를 선택하세요")
st.sidebar.radio(
    "Menu",
    ["홈 (Home)", "문제 정의 (Problem)", "데이터 전처리 (Preprocess)", "데이터 시각화 (Visualize)", "모델링 및 예측 (Modeling)"],
    label_visibility="collapsed",
)

# 메인 타이틀
st.title("🚨 청소년 도박 실태 및 위험군 예측 프로젝트")
st.markdown(
    """
    본 프로젝트는 청소년 도박의 심각성을 인지하고, 실태조사 데이터를 바탕으로 **도박 위험군을 조기에 예측 및 분류**하기 위한 데이터 분석 웹 대시보드입니다.  
    좌측 사이드바의 메뉴를 통해 데이터 정제부터 모델링까지의 전 과정을 확인할 수 있습니다.
    """
)

st.divider()

# 3단 컬럼 (핵심 단계 소개)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 문제 정의")
    st.write("**청소년 불법 도박의 확산과 위험성**")
    st.caption("""
    - 돈 따는 짜릿함과 또래 문화로 인한 중독성 분석
    - 사설 토토, 온라인 카지노 등 불법 도박 접근 경로 파악
    - 핵심 질문: '어떤 청소년이 도박 중독 위험에 가장 취약한가?'
    """)

with col2:
    st.subheader("⚙️ 데이터 파이프라인")
    st.write("**실태조사 데이터 수집부터 정제까지**")
    st.caption("""
    - **수집:** 청소년 도박 실태조사 설문 및 상담 데이터 연계
    - **정제:** 불성실 응답(이상치) 제거 및 학년 무응답 처리
    - **탐색:** 용돈 대비 도박 지출 비율 등 핵심 파생변수 생성
    - **준비:** 모델 학습을 위한 인코딩 및 데이터 분할
    """)

with col3:
    st.subheader("🤖 모델링 및 예측")
    st.write("**도박 위험군 분류 모델 구현**")
    st.caption("""
    - 머신러닝 기반의 위험군(정상/위험/고위험) 분류
    - 정밀도(Precision)와 재현율(Recall) 중심의 모델 평가
    - 주요 변수(용돈, 스마트폰 사용 시간 등)의 영향도 분석
    """)

st.divider()

# 하단 상세 구역
left, right = st.columns([2, 1])

with left:
    st.subheader("📋 프로젝트 개요 및 배경")
    st.write("""
    최근 스마트폰을 통한 불법 사설 도박 접근성이 높아지면서, 청소년 도박 중독이 심각한 사회적 문제로 대두되고 있습니다. 
    단순한 호기심으로 시작한 도박이 자금 마련을 위한 2차 범죄(절도, 중고거래 사기 등)로 이어지는 악순환을 끊기 위해서는 **조기 예측과 맞춤형 개입**이 필수적입니다.
    
    이 대시보드는 실제 통계와 머신러닝 모델을 활용하여 청소년 도박 문제 해결을 위한 데이터 기반의 인사이트를 제공합니다.
    """)

with right:
    st.subheader("🚀 실행 방법 (Quick Start)")
    st.text("필수 라이브러리 설치:")
    st.code("pip install -r requirements.txt")
    st.text("대시보드 앱 실행:")
    st.code("streamlit run main.py")
