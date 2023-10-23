from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

load_dotenv()

API_KEY = os.environ['OPENAI_API_KEY']

with st.sidebar:
    st.title("Chat-EDA")
    uploaded_file = st.file_uploader("Insert yor dataset", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    nrows = st.number_input("Show number of rows", min_value=1, value=5, step=1)
    st.dataframe(df.head(nrows), use_container_width=True)

    with st.form("prompt_area"):
        prompt = st.text_input("Ask here")
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.write("This is the answer")