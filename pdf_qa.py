from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
#from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
#from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from constants import *
from transformers import AutoTokenizer
import torch
import os
import re
import OpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"

class PdfQA:
    def __init__(self,config:dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None
        self.chat_history = []

    
    @classmethod
    def create_openai_embaddings(cls):
        embaddings = OpenAIEmbeddings()
        return embaddings
    
    @classmethod
    def create_openai_35(cls,load_in_8bit=False):
        llm = ChatOpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.7,max_tokens=350)
        return llm
    
    
        
    def init_embeddings(self) -> None:
        # OpenAI ada embeddings API
        if self.config["embedding"] == EMB_OPENAI_ADA:
            self.embedding = OpenAIEmbeddings()
            
        else:
            self.embedding = None ## DuckDb uses sbert embeddings
           
    def init_models(self) -> None:
        """ Initialize LLM models based on config """
        load_in_8bit = self.config.get("load_in_8bit",False)
        # OpenAI GPT 3.5 API
        if self.config["llm"] == LLM_OPENAI_GPT35:
            if self.llm is None:
                self.llm = PdfQA.create_openai_35(load_in_8bit=load_in_8bit)
        
        
        else:
            raise ValueError("Invalid config")        
    
    
    def vector_db_pdf(self) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        persist_directory = self.config.get("persist_directory",None)

        path = "./data/"
        dir_list = os.listdir(path)
        documents=[]
        if len(dir_list) != 0:
            for file in dir_list:
                file = path + file
                loader = PDFPlumberLoader(file)
                doc = loader.load()
                documents+=doc
            
            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
            texts = text_splitter.split_documents(documents)

            self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=persist_directory)
        else:
            raise ValueError("NO PDF found")

    def retreival_qa_chain(self):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k":3})
        
        if self.config["llm"] == LLM_OPENAI_GPT35:
          
          self.qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.8,max_tokens=1024),
                                      self.vectordb.as_retriever(search_kwargs={"k":3}))
          
          
          #self.qa.return_source_documents = True
            
    def answer_query(self,question:str) ->str:

        answer_dict = self.qa({"question":question,"chat_history":self.chat_history})
        print(answer_dict)
        answer = answer_dict["answer"]
        self.chat_history.append((question,answer)) # add query and response to the chat history
        
        
        return answer
    
    

