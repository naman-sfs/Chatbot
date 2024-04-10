from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pdf_qa import PdfQA
from constants import *
import os


app = FastAPI()

bot = PdfQA()
emb = EMB_OPENAI_ADA
llm = LLM_OPENAI_GPT35


class Question(BaseModel):
    question: str 
    
    
def load_llm(llm,load_in_8bit):
    if llm == LLM_OPENAI_GPT35:
        return bot.create_openai_35(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")

def load_emb(emb):
    if emb == EMB_OPENAI_ADA:
        return bot.create_openai_embaddings()
    else:
        raise ValueError("Invalid embedding setting")
    
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post('/api/train_model')
async def trainModel():
    try:

        bot.config = {
                "pdf_path": str('B1'),
                "embedding": emb,
                "llm": llm,
                "load_in_8bit": False
        }
        bot.embedding = load_emb(emb)
        bot.llm = load_llm(llm,False)
        bot.init_embeddings()
        bot.init_models()

        bot.vector_db_pdf()
        return {"msg":"Model Trained Successfully!!!"}
    
    except:
        return {"msg":"Internal Server Error!!!"}
        
@app.get('/api/ask')
async def askQuery(question:Question):
    try:
        bot.retreival_qa_chain()
        answer = bot.answer_query(question.question)
        return {"msg":"Answer Generated Successfully!!!","question":question.question,"answer":answer}
    except:
        return {"msg":"Internal Server Error!!!"}




