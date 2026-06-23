import streamlit as st

st.set_page_config(page_title="문제정의 및 데이터 수집", layout="wide")

st.title("🎰 1. 문제 정의 및 데이터 수집 (기획/AI 설계)")
st.markdown("본 페이지는 청소년 도박 중독 예방을 위한 프로젝트 목적, 캐글 데이터셋 구조, 그리고 AI 모델 선정 논리를 정의하는 공간입니다.")

st.divider()

# 1. 문제 정의 섹션
st.header("💡 1. 문제 정의 (Problem Definition)")
col_prob1, col_prob2 = st.columns([2, 1])

with col_prob1:
    st.markdown("""
    * **🚩 해결하려는 사회적 문제**
        * 2022~2024년 청소년 도박 중독 환자 수가 **4.8% 증가**했으며, 이 중 **19.8%**는 지속적으로 도박 중독에서 벗어나지 못하는 심각한 상황입니다.
        * 핵심 원인은 청소년들이 온라인 도박의 불공정한 승률 구조와 중독성을 정량적으로 인지하지 못하고 있다는 점입니다.
    * **🎯 프로젝트 목표**
        * 청소년들에게 **'온라인 도박이 구조적으로 얼마나 큰 경제적 손해를 가져오는지'**를 데이터로 명확히 시각화하여 보여줍니다.
        * 이를 통해 선제적인 경각심을 일깨우고 건전한 인터넷 사용 습관 형성을 돕는 시스템을 구축합니다.
    """)
with col_prob2:
    st.metric(label="청소년 도박 환자 증가율 (22~24)", value="4.8% ▲")
    st.metric(label="지속적 도박 중독 비율", value="19.8%")

st.divider()

# 2. 데이터 수집 배경 섹션
st.header("📊 2. 데이터 수집 배경 (Data Collection)")
st.markdown("""
* **🗂️ 사용 데이터셋:** Kaggle 제공 **'Online Unfair Casino'** 데이터셋
* **🧠 수집 배경:** 온라인 카지노의 오픈 소켓에서 수집된 실제 데이터로, 게임 내 가상 아이템(Skins)이 외부 사이트에서 현금화된다는 점에서 **청소년들이 쉽게 빠지는 실제 도박과 매우 유사한 위험성**을 내포하고 있습니다. 도박 사이트의 불공정성과 구조적 손실을 증명하기에 최적의 데이터입니다.
""")

# 데이터셋 컬럼 설명 표
st.markdown("### 📋 핵심 데이터셋 컬럼 명세 (Data Schema)")
data_schema = {
    "컬럼명": ["ID", "gamers", "skins", "money", "ticks", "peopleWin", "peopleLost", "outpay", "time", "moderator"],
    "설명": [
        "각 게임의 고유 ID",
        "이 게임에 참여한 플레이어 수",
        "플레이어들이 이 게임에 걸었던 아이템(스킨) 수",
        "아이템의 실제 현금 환산 금액 (달러 $)",
        "이 게임에 대한 사이트의 계수 (배당률과 유사)",
        "이 게임에서 사람들이 획득한 총 금액 (달러 $)",
        "이 게임에서 사람들이 잃은 총 금액 (달러 $)",
        "총 지급액 (money - peopleLost + peopleWin)",
        "게임 진행 시간 (YY-MM-DD hh-mm)",
        "운영자(봇)의 게임 참여 여부 (True/False)"
    ]
}
st.table(data_schema)

st.divider()

# 3. AI 모델 선정 논리 섹션
st.header("🤖 3. 분류/회귀 모델 선정 논리 (Model Selection)")
col_mod1, col_mod2 = st.columns(2)

with col_mod1:
    st.subheader("📈 회귀(Regression) 모델: 손실액 예측")
    st.warning("**🎯 목표: 특정 게임 조건에서 발생할 유저의 총 손실액(peopleLost) 예측**")
    st.markdown("""
    * **사용 알고리즘:** 선형 회귀(Linear Regression), 랜덤 포레스트 회귀, XGBoost, LightGBM
    * **선정 논리:** * 배당률(`ticks`)과 판돈(`money`) 등의 변수가 손실액에 미치는 인과관계를 정량적으로 밝혀냅니다.
        * 청소년들에게 *"이 방(조건)에서 도박을 하면 너는 평균적으로 **XX달러를 잃게 된다**"*는 수치적 경고를 직접적으로 제시하기 위함입니다.
    """)

with col_mod2:
    st.subheader("🎯 분류(Classification) 모델: 중독 위험군 예측")
    st.info("**🎯 목표: 유저의 게임 참여 패턴을 기반으로 '도박 중독 위험군' 분류**")
    st.markdown("""
    * **사용 알고리즘:** 로지스틱 회귀, 서포트 벡터 머신(SVM), 랜덤 포레스트 분류기
    * **선정 논리:** * 데이터 자체에 중독 여부 레이블이 없으므로, 파악되는 장기적·반복적 참여 패턴과 손실액 크기를 기준으로 '중독 위험(위험/정상)'을 정의하는 전처리를 선행합니다.
        * 규칙 기반 모델(Decision Tree 등)을 통해 **어떤 행동 양식이 중독 위험을 유발하는지 설명력(XAI)**을 갖추기 위해 선정했습니다.
    """)

st.divider()
st.success("📢 [기획/AI 설계자 완료] 본 페이지의 기획 데이터 및 AI 파이프라인 설계 기준이 확정되었습니다. 개발팀은 위 명세를 기준으로 모델링을 전개해 주세요.")
