import streamlit as st
import pandas as pd
import random
st.set_page_config(page_title="Understand the Data", layout="wide")

st.markdown("# Understand the Data")
dataset = pd.read_csv("test_sample.csv")

txt = dataset.iat[0, 0]
original_summary = dataset.iat[0, 1]
if st.button('Randomly generate a Bill Example'):
    my_num = random.randrange(len(dataset))
    txt = dataset.iat[my_num, 0]
    original_summary = dataset.iat[my_num, 1]
else:
    pass
col1, col2 = st.columns(2)
with col1:
    st.header("Original `Billsum` dataset:")
    st.write(dataset.head(10))
with col2:
    txt = st.text_area('Example text to analyze', txt, height = 250)
    original_summary = st.text_area('Corresponding summary', original_summary, height = 250)