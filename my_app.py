import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.markdown("# Welcome to Text Summarization Demo! 👋")

st.sidebar.success("Select a demo above.")

st.markdown("By Aaron Tae, Kevin Hamakawa, Emily Gong, Emily Huang, Tony Lei, Ved Phadke, Vivian Lee, Victor Shi")

st.markdown("## Introduction")
st.markdown('''
For our project, we looked into text summarization. Based on our research, there are 2 main categories of text summarization techniques: *extractive* and *abstractive*. As the names suggest,
the *extractive summarization* method directly extracts information from the orignal text whereas the *abstractive summarization* 
method employs abstraction to produce a high level summary similar to humans'.
''')
st.markdown("## Feel free to visit the tabs on the left to learn more")
st.markdown('Find us on [GitHub](https://github.com/the-data-science-union/DSU-S2023-Text-Summarization) or [Medium](https://www.google.com)')

st.balloons()
