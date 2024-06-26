import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



api_key = "4fdd3135c3a94d9e920354bec1fa3f7f"
endpoint =  "https://tech-ai-ml-class-shima.openai.azure.com/"


st.title('LLMによるタスク処理')

llm = AzureChatOpenAI(
    deployment_name = 'gpt-35-turbo-0613',
    openai_api_version='2023-05-15',
    azure_endpoint=endpoint,
    openai_api_key=api_key,
    temperature=1
)

user_imput = st.text_area('プロンプトを入力')

#タスクの選択
template_option = st.selectbox('タスクの選択',('Aタスク', 'Bタスク', 'ヒヤリハット報告要約'))

#
templates = {
    '要約タスク' : '''以下の文章を要約してください。
    
    {input}
    
    ''' ,
    'ヒヤリハット報告要約' : '''以下の文章はヒヤリハット報告です。
    
    {input}
    
    上記を基に、必ず以下の形式で出力してください。
    情報が含まれない場合、「不明」と出力してください。
    ・いつ
    （ここに出力）

    ・どこで
    （ここに出力）

    ・誰が
    （ここに出力）

    ・なにをしていた
    （ここに出力）

    ・なにが起きた
    （ここに出力）

    ・原因
    （ここに出力）
    '''
}

st.write(templates[template_option])

prompt_template = PromptTemplate(
    input_variables=['input'],
    template=templates[template_option] #ユーザーが選んだテンプレート
)

output_button = st.button('GPTの出力')

if output_button :
    chain = LLMChain (llm=llm, prompt=prompt_template)
    reponse = chain.run(user_imput) #LLMから出力

    st.write(reponse)