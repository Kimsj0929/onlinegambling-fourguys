import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 페이지 레이아웃 및 한글 폰트 설정
st.set_page_config(page_title="NEXUS 데이터 시각화 파이프라인", layout="wide")

# Matplotlib/Seaborn 한글 깨짐 방지 및 다크 테마 커스텀
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'NanumGothic' or 'Malgun Gothic' or 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.color'] = '#8A99AD'
plt.rcParams['axes.labelcolor'] = '#8A99AD'

# 2. 데이터 데이터셋 로드
@st.cache_data
def load_raw_data():
    try:
        # 업로드된 실제 파일 로드
        return pd.read_csv("onlineCasino.csv")
    except:
        # 백업용 가상 데이터 생성 (구조 싱크 최적화)
        np.random.seed(42)
        n = 250
        money = np.random.uniform(50, 500, n)
        outpay = money * np.random.choice([1.2, 0.8, 0.1], size=n, p=[0.4, 0.4, 0.2])
        # 인위적인 극단적 이상치 주입
        outpay[10] = 5000 
        outpay[50] = -999
        return pd.DataFrame({
            'gamers': np.random.randint(50, 200, n),
            'skins': np.random.randint(100, 300, n),
            'money': money,
            'ticks': np.random.uniform(1.0, 15.0, n),
            'outpay': outpay
        })

df_raw = load_raw_data()

# 3. 데이터 전처리 파이프라인 엔진 구동
def preprocess_data(df):
    df_clean = df.copy()
    
    # [단계 1] 수치형 컬럼 추출
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # [단계 2] 결측치 제거
    df_clean = df_clean.dropna(subset=numeric_cols)
    
    # [단계 3] 사분위수(IQR) 기준 이상치(Outlier) 제거 필터링
    for col in ['money', 'outpay']:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # 정상 범위 데이터만 필터링링
            df_clean = df_clean[(df_clean[col] >= lower
