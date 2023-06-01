import streamlit as st
import pandas as pd
import random
st.set_page_config(page_title="Understand the Data", layout="wide")

st.markdown("# Understand the Data")
dataset = pd.read_csv("test_sample.csv")

txt = dataset.iat[0, 0]
original_summary = dataset.iat[0, 1]

col1, col2 = st.columns(2)
with col1:
    st.header("Original `Billsum` test set:")
    st.write(dataset.head(10))
    avg_len_text = dataset['text'].str.len().mean()
    avg_len_summary = dataset['summary'].str.len().mean()
    avg_len_title = dataset['title'].str.len().mean()
    st.write("Average length of a Bill:", avg_len_text)
    st.write("Average length of a Summary:", avg_len_summary)
    st.write("Average length of a Title:", avg_len_title)
with col2:
    st.header("Example:")
    if st.button('Randomly generate a Bill Example'):
        my_num = random.randrange(len(dataset))
        txt = dataset.iat[my_num, 0]
        original_summary = dataset.iat[my_num, 1]
    else:
        pass
    txt = st.text_area('Text', txt, height = 250)
    original_summary = st.text_area('Corresponding summary', original_summary, height = 250)