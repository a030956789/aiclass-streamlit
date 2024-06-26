import streamlit as st
from openai import AzureOpenAI


api_key = st.text_input('api_key')
endpoint = st.tex_input('endpoint')


#azureにアクセスできる状態にする
Client=AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version='2024-02-01'
)

st.title('初めてのstreamlit')

#返答のランダム性を変えるため
temperature = st.slider('Temperature',0.0,1.0,step=0.1)

pronpt = st.text_area("プロンプトの入力")

output_button = st.button('GPTの出力')

if output_button:

    #GPT3.5にアクセス
    response = Client.chat.completions.create(
        model="gpt-35-turbo-0613",
        messages=[
            {"role":"user",
            "content":pronpt}]
)

    st.write(response.choices[0].message.content)
