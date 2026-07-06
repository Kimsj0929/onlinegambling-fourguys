import streamlit as st
import pandas as pd
import numpy as np

# 1. 페이지 기본 설정
st.set_page_config(page_title="청소년 도박 데이터 전처리", layout="wide")

st.title("2. 데이터 전처리")
st.markdown("""
청소년 도박 실태조사 및 상담 데이터의 신뢰성을 높이기 위한 전처리 페이지입니다.  
결측치 처리, 불성실 응답(이상치) 제거, 도박 위험군 분류를 위한 인코딩 및 스케일링을 수행합니다.
""")

# --- 2. 가상의 청소년 도박 데이터 생성 (비교 데모용) ---
@st.cache_data
def load_data():
    # 전처리 전 데이터 (결측치, 말도 안 되는 이상치, 중복 데이터 포함)
    raw_data = pd.DataFrame({
        "학생ID": [1, 2, 3, 4, 5, 6, 6], # 6번 중복
        "학년": ["중2", "고1", None, "고3", "중3", "고2", "고2"], # 결측치 포함
        "월평균_용돈": [50000, 100000, 30000, 80000, 50000, 60000, 60000],
        "월_도박_지출": [5000, 20000, 0, 99999999, 15000, 4000, 4000] # 4번 학생의 불성실 응답(이상치)
    })
    
    # 전처리 후 데이터 (중복 제거, 결측치 대체, 이상치 제거)
    clean_data = raw_data.drop_duplicates().copy()
    clean_data["학년"] = clean_data["학년"].fillna("미기재")
    clean_data = clean_data[clean_data["월_도박_지출"] < 1000000] # 100만 원 이상 장난 응답 제거
    
    return raw_data, clean_data

raw_df, clean_df = load_data()


# --- 3. 전처리 체크리스트 및 팁 ---
st.subheader("📋 청소년 데이터 맞춤 전처리 체크리스트")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### **진행 상황 체크**")
    items = [
        "결측치 확인 및 처리 (무응답 '미기재' 처리)",
        "중복 데이터 제거 (동일 인물의 중복 제출 필터링)",
        "이상치 탐지 및 제거 (용돈 대비 터무니없는 도박 지출액)",
        "범주형 변수 인코딩 (학년, 성별, 도박 종류 등)",
        "수치형 변수 스케일링 (도박 빈도, 용돈 대비 지출 비율 등)"
    ]
    for item in items:
        st.checkbox(item, value=True) # 예시 시각화를 위해 기본 체크 처리

with col2:
    st.markdown("### 💡 전처리 핵심 팁")
    st.info("""
    - **불성실 응답 필터링:** 청소년 설문조사는 '돈을 하루에 10억 썼다' 같은 장난식 답변이 있을 수 있으므로 **이상치 제거**가 매우 중요합니다.
    - **도박 종류 인코딩:** 불법 도박(온라인 카지노, 사설 토토 등)과 합법 도박(복권, 인형 뽑기 등)을 나누어 인코딩하는 것을 권장합니다.
    - **파생 변수 생성 추천:** 단순히 '도박 금액'을 보기보다 **'용돈 대비 도박 지출 비율'** 같은 변수를 만들면 분석 효과가 극대화됩니다.
    """)

st.markdown("---")


# --- 4. 전처리 전 vs 후 데이터 시각적 비교 ---
st.subheader("📊 전처리 전 vs 후 데이터 비교")

tab1, tab2 = st.tabs(["📄 데이터프레임 직접 비교", "📈 주요 통계치 변화"])

with tab1:
    st.markdown("##### 원본 데이터와 전처리(정제)가 완료된 데이터를 나란히 비교합니다.")
    col_raw, col_clean = st.columns(2)
    
    with col_raw:
        st.error(f"❌ 전처리 전 원본 데이터 (총 {len(raw_df)}건)")
        st.dataframe(raw_df, use_container_width=True)
        st.caption("⚠️ 결함 지점: 3번 학년 결측치(None), 4번 도박 지출 이상치, 6-7번 중복 데이터")
        
    with col_clean:
        st.success(f"⭕ 전처리 후 정제된 데이터 (총 {len(clean_df)}건)")
        st.dataframe(clean_df, use_container_width=True)
        st.caption("✨ 정제 완료: 중복 제거 완료, 결측치 '미기재' 대체 완료, 불성실 이상치 행 삭제 완료")

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
        # 네임에러(월_to_박_지출 오타)를 '월_도박_지출'로 완벽히 수정함
        st.metric(
            label="정제 후 월평균 도박 지출액", 
            value=f"{int(clean_df['월_도박_지출'].mean()):,}원",
            delta=f"현실적인 수치로 조정됨 (감소폭: {int(raw_df['월_도박_지출'].mean() - clean_df['월_도박_지출'].mean()):,}원)",
            delta_color="normal"
        )
        
    st.markdown("##### 데이터 건수 변화 시각화")
    count_df = pd.DataFrame({
        "상태": ["전처리 전", "전처리 후"],
        "데이터 건수": [len(raw_df), len(clean_df)]
    })
    st.bar_chart(data=count_df, x="상태", y="데이터 건수", color="#3b82f6", use_container_width=True)


# --- 5. 전처리 세부 조치 리포트 (어떤 부분이 잘못되어 어떻게 고쳤는가) ---
st.markdown("---")
st.subheader("📝 전처리 세부 조치 리포트")
st.markdown("데이터 수집 과정에서 발생한 결함과 이를 해결한 구체적인 전처리 내역입니다.")

bad_col, arrow_col, good_col = st.columns([1.2, 0.2, 1.2])

with bad_col:
    st.markdown("#### ❌ 발견된 데이터 결함 (Before)")
    
    st.error("""
    **1. 불성실 응답 (이상치 발견)**
    - **문제점:** 4번 학생의 `월_도박_지출`이 **99,999,999원**으로 기록됨.
    - **이유:** 청소년 설문조사 특성상 장난으로 과장되게 적은 답변(불성실 응답)으로 판단됨. 이로 인해 전체 평균치가 심각하게 왜곡됨.
    """)
    
    st.error("""
    **2. 무응답 데이터 (결측치 발생)**
    - **문제점:** 3번 학생의 `학년` 정보가 **None(공백)**으로 누락됨.
    - **이유:** 도박 관련 민감한 문항에 답변하는 과정에서 인적 사항을 기피했거나 설문 시스템 오류로 추정됨.
    """)
    
    st.error("""
    **3. 중복 제출 데이터 (중복값 존재)**
    - **문제점:** 6번과 7번 데이터의 학생ID, 학년, 용돈, 지출액이 **100% 일치**.
    - **이유:** 설문 제출 버튼을 여러 번 클릭했거나, 시스템상 동일 인물의 데이터가 중복 수집됨.
    """)

with arrow_col:
    st.markdown("<h2 style='text-align: center; margin-top: 50px; color: #94a3b8;'>➡️</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: 130px; color: #94a3b8;'>➡️</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: 130px; color: #94a3b8;'>➡️</h2>", unsafe_allow_html=True)

with good_col:
    st.markdown("#### ⭕ 조치 및 고친 내용 (After)")
    
    st.success("""
    **1. 임계치(Threshold) 기준 이상치 제거**
    - **해결책:** 청소년 평균 용돈 범위를 고려하여 `월_도박_지출`이 **100만 원 이상인 데이터는 삭제** 조치함.
    - **결과:** 평균 도박 지출액이 정상 범위로 복구되어 분석의 신뢰도 확보.
    """)
    
    st.success("""
    **2. 결측치 문자열 대체**
    - **해결책:** 해당 행을 무작정 지우면 다른 유의미한 데이터(용돈 등)까지 손실되므로, 학년을 **'미기재'라는 새로운 범주로 대체**함.
    - **결과:** 데이터 손실을 최소화하고 학년 미기재 그룹의 도박 성향도 추적 가능해짐.
    """)
    
    st.success("""
    **3. 중복 행(Row) 제거**
    - **해결책:** `drop_duplicates()` 함수를 사용하여 완벽히 일치하는 **중복 데이터 1건을 삭제**하고 고유한 1건만 남김.
    - **결과:** 통계치 계산 시 데이터가 부풀려지는 현상(과적합 위험)을 방지함.
    """)
