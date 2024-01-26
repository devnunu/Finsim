from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.faiss import FAISS
from langchain_openai import AzureOpenAI, AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

import plotly.express as px

import matplotlib.pyplot as plt
import streamlit as st
import os
import pandas as pd

load_dotenv()

openai_api_type = os.getenv("OPENAI_API_TYPE")
openai_api_version = os.getenv("OPENAI_API_VERSION")
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_API_BASE")

st.set_page_config(
    page_title="Finsim",
    page_icon="⚡️",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(show_spinner="파일 임베딩중...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    df = pd.read_csv(file_path)
    return df

def filter_csv(df):
    data = {
        '뉴스 식별자': df['뉴스 식별자'], 
        '일자': df['일자'],
        '언론사': df['언론사'],
        '기고자': df['기고자'],
        '제목': df['제목'],
        '본문': df['본문'],
        'URL': df['URL'],
    }
    filteredDf = pd.DataFrame(data)
    st.write(filteredDf.head())
    return pd.DataFrame(data)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            will be passed a keyword. In the given context, the evaluation of the received keyword should be expressed as "positive", "negative", or "neutral" only.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

map_data_frame_chain = (
    {
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

df = pd.DataFrame()

def map_data_frame(inputs):
    question = inputs["question"]
    results = {'감정': [], '뉴스 식별자':[]}
    for idx, row in df.iterrows():
        content = row['본문']
        id = row['뉴스 식별자'] 
        response = map_data_frame_chain.invoke(
            {"context": content, "question": question}
        ).content
        results['뉴스 식별자'].append(id)
        results['감정'].append(response)
    return results



##
# UI 섹션
##
st.title("핀심(Finsim)")

st.markdown(
"""
안녕하세요!

핀다 민심 판독기 - [핀심]입니다!

좌측에 CSV 파일을 업로드 하고 분석 데이터를 확인해보세요!
            
"""
)

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("start_date")

with col2:
    end_date = st.date_input("end_date")

keyword = st.text_input("키워드를 입력해주세요")

with st.sidebar:
    file = st.file_uploader(
        "Upload a .csv file",
        type=["csv"],
    )

if file:
    df = filter_csv(embed_file(file))
    send_message("파일 업로드가 완료되었습니다! 무엇이든 물어보세요", "ai", save=False)
    paint_history()
    message = st.chat_input("업로드된 파일 대한 질문을 해보세요.")
    if message:
        send_message(message, "human")
        map_chain = {
            "question": RunnablePassthrough()
        } | RunnableLambda(map_data_frame)


        chain_results = map_chain.invoke(message)

        newDataFrame = pd.DataFrame(chain_results)
        finalDf = pd.merge(df, right = newDataFrame, on = '뉴스 식별자')

        groupby_emotion = finalDf.groupby('감정')

        df_emotion_count = pd.DataFrame({'count':groupby_emotion.size()}).reset_index()

        st.dataframe(df_emotion_count.head())

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(df_emotion_count, values='count', names='감정')
            fig.update_traces(hole=.3)
            st.plotly_chart(fig)

        with col2:
            fig = px.pie(df_emotion_count, values='count', names='감정')
            fig.update_traces(hole=.3)
            st.plotly_chart(fig)

        

else:
    st.session_state["messages"] = []


st.empty()
st.button("분석하기")