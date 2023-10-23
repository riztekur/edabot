from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import matplotlib

st.set_page_config(layout="wide", page_title="Chat-EDA")

matplotlib.use('TkAgg')
load_dotenv()

API_KEY = os.environ['OPENAI_API_KEY']
llm = OpenAI(api_token=API_KEY)

with st.sidebar:
    st.title("Chat-EDA")
    uploaded_file = st.file_uploader("Insert yor dataset", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = SmartDataframe(df, config={'llm':llm})
    with st.sidebar:
        st.subheader('Dataset information')
        info = pd.DataFrame(
            {
                'Rows' : [df.shape[0]],
                'Columns': [df.shape[1]]
            }
        )
        st.dataframe(info, use_container_width=True, hide_index=True)

        missing_values = pd.DataFrame(
            df.isnull().sum().to_frame('Nulls').reset_index()
        ).rename(columns={'index':'Column'})

        st.dataframe(missing_values, use_container_width=True, hide_index=True)

    nrows = st.number_input("Show number of rows", min_value=1, value=5, step=1)
    st.dataframe(df.head(nrows), use_container_width=True)

    with st.form("prompt_area"):
        prompt = st.text_input("Ask here")
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        if prompt:
            with st.spinner("Generating answer, please wait..."):
                st.write(df.chat(prompt))
        else:
            st.write("Please enter a request.")