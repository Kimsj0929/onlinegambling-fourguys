import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="청소년 도박 데이터 전처리", layout="wide")

st.title("2. 데이터 전처리 (청소년 도박 데이터)")
st.markdown("""
청소년 도박 실태조사 및 상담 데이터의 신뢰성을 높이기 위한 전처리 페이지입니다.  
결측치 처리, 불성실 응답(이상치) 제거, 도박 위험군 분류를 위한 인코딩 및 스케일링을 수행합니다.
""")

# --- 1. 가상의 청소년 도박 데이터 생성 (예시용) ---
@st.cache_data
def load_data():
    # 전처리 전 데이터 (결측치와 말도 안 되는 이상치 포함)
    raw_data = pd.DataFrame({
        "학생ID": [1, 2, 3, 4, 5, 6, 6], # 6번 중복
        "학년": ["중2", "고1", None, "고3", "중3", "고2", "고2"], # 결측치 포함
        "월평균_용돈": [50000, 100000, 30000, 80000, 50000, 60000, 60000],
        "월_도박_지출": [5000, 20000, 0, 99999999, 15000, 4000, 4000] # 4번 학생 용돈 대비 과도한 이상치(장난)
    })
    
    # 전처리 후 데이터 (중복 제거, 결측치 대체, 이상치 제거)
    clean_data = raw_data.drop_duplicates().copy()
    clean_data["학년"] = clean_data["학년"].fillna("미기재")
    clean_data = clean_data[clean_data["월_도박_지출"] < 1000000] # 100만 원 이상 장난 응답 제거
    
    return raw_data, clean_data

raw_df, clean_df = load_data()


# --- 2. 체크리스트 및 팁 구역 ---
st.subheader("📋 청소년 데이터 맞춤 전처리 체크리스트")
col1, col2 = st.columns([1, 1])

with col1:
    items = [
        "결측치 확인 (무응답 '미기재' 처리)",
        "중복 데이터 제거 (동일ID 중복 제출 필터링)",
        "이상치 탐지 및 제거 (용돈 대비 터무니없는 도박 지출액)",
        "범주형 변수 인코딩 및 스케일링 준비"
    ]
    for item in items:
        st.checkbox(item, value=True) # 예시를 위해 기본 체크

with col2:
    st.info("""
    💡 **청소년 데이터 전처리 핵심 포인트:**
    - 청소년 설문조사 특성상 **용돈 범위를 벗어난 도박 금액(예: 9,999만 원)** 같은 불성실 응답(이상치)을 반드시 필터링해야 분석 결과가 왜곡되지 않습니다.
    """)

st.markdown("---")


# --- 3. ⭐️ 전처리 전/후 비교 섹션 ⭐️ ---
st.subheader("📊 전처리 전 vs 후 데이터 비교")

# 탭을 나누어 표와 요약 통계를 비교할 수 있도록 구성
tab1, tab2 = st.tabs(["📄 데이터프레임 직접 비교", "📈 주요 통계치 변화"])

with tab1:
    st.markdown("##### 원본 데이터와 전처리(정제)가 완료된 데이터를 나란히 비교합니다.")
    col_raw, col_clean = st.columns(2)
    
    with col_raw:
        st.error(f"❌ 전처리 전 원본 데이터 (총 {len(raw_df)}건)")
        st.dataframe(raw_df, use_container_width=True)
        st.caption("⚠️ 빨간색 강조: 3번 학년 결측치(None), 4번 도박 지출 이상치, 6-7번 중복 데이터")
        
    with col_clean:
        st.success(f"⭕ 전처리 후 정제된 데이터 (총 {len(clean_df)}건)")
        st.dataframe(clean_df, use_container_width=True)
        st.caption("✨ 조치사항: 중복 제거 완료, 결측치 '미기재' 대체 완료, 불성실 이상치 제거 완료")

with tab2:
    st.markdown("##### 전처리를 통해 왜곡되었던 데이터의 평균과 통계가 어떻게 정상화되었는지 확인합니다.")
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.metric(
            label="원본 데이터 월평균 도박 지출액", 
            value=f"{int(raw_df['월_도박_지출'].mean()):,}원",
            delta="장난 데이터로 인해 평균이 극도로 왜곡됨",
            delta_color="inverse"
        )
        
    with col_stat2:
        st.metric(
            label="정제 후 월평균 도박 지출액", 
            value=f"{int(clean_df['월_도박_지출'].mean()):,}원",
            delta=f"현실적인 수치로 조정됨 (감소폭: {int(raw_df['월_도박_지출'].mean() - clean_df['월_to_박_지출'].mean()):,}원)",
            delta_color="normal"
        )
        
    # 간단한 가로 막대 그래프로 데이터 건수 변화 시각화
    st.markdown("##### 데이터 건수 변화 시각화")
    count_df = pd.DataFrame({
        "상태": ["전처리 전", "전처리 후"],
        "데이터 건수": [len(raw_df), len(clean_df)]
    })
    st.bar_chart(data=count_df, x="상태", y="데이터 건수", color="#3b82f6", use_container_width=True)
