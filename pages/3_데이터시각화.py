import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="데이터 시각화", layout="wide")

st.title("3. 데이터 시각화")
st.markdown("기본 샘플 차트를 보여주는 페이지입니다.")

df = pd.DataFrame({
    "x": np.arange(1, 11),
    "y": np.random.randint(10, 100, 10),
})

chart = alt.Chart(df).mark_line(point=True).encode(
    x="x",
    y="y"
)

st.altair_chart(chart, use_container_width=True)
