import streamlit as st
from pdf_qa import PdfQA

# from pathlib import Path
# from tempfile import NamedTemporaryFile
import time
# import shutil
from constants import *

import os

os.environ["OPENAI_API_KEY"] = "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"

# Streamlit app code
st.set_page_config(
    page_title='AIT',
    page_icon='./chatbot.png',
    layout='wide',
    initial_sidebar_state='auto',
)

st.image('./chatbot.png',width=100)

if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"]:PdfQA = PdfQA() ## Intialisation

## To cache resource across multiple session 
@st.cache_resource
def load_llm(llm,load_in_8bit):
    if llm == LLM_OPENAI_GPT35:
        return PdfQA.create_openai_35(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")

## To cache resource across multiple session
@st.cache_resource
def load_emb(emb):
    if emb == EMB_OPENAI_ADA:
        return PdfQA.create_openai_embaddings()
    else:
        raise ValueError("Invalid embedding setting")




st.title("AIT Chat⛵")

with st.sidebar:
    emb = st.radio("**Select Embaddings**", [EMB_OPENAI_ADA],index=0)
    llm = st.radio("**Select LLM Model**", [LLM_OPENAI_GPT35],index=0)
    load_in_8bit = False
    # pdf_file = st.file_uploader("**Upload PDF**", type="pdf")

    
    if st.button("Submit"):
        with st.spinner(text="Uploading PDF and Generating Embeddings.."):
            
            st.session_state["pdf_qa_model"].config = {
                "pdf_path": str('B1'),
                "embedding": emb,
                "llm": llm,
                "load_in_8bit": load_in_8bit
            }
            st.session_state["pdf_qa_model"].embedding = load_emb(emb)
            st.session_state["pdf_qa_model"].llm = load_llm(llm,load_in_8bit)        
            st.session_state["pdf_qa_model"].init_embeddings()
            st.session_state["pdf_qa_model"].init_models()
            st.session_state["pdf_qa_model"].vector_db_pdf()
            st.sidebar.success("Model Trained successfully")

question = st.text_input('Ask a question', 'Hey, how are you?')

if st.button("Answer"):
    
    try:
        with st.spinner(text="Generating Response!!!"):
            st.session_state["pdf_qa_model"].retreival_qa_chain()
            # question = question + " Provide 5-7 key points on it with explaination and example."
            answer = st.session_state["pdf_qa_model"].answer_query(question)
        t = st.empty()
        for i in range(len(answer) + 1):
            t.markdown("%s.." % answer[0:i])
            time.sleep(0.001)
        #st.write("#### Sources")
        # for i in range(len(sources)):
        #     st.write(f'##### {i+1}.')
        #     st.write(f'{sources[i]}')
        
    except Exception as e:
        st.error(f"Error answering the question: {str(e)}")
        
