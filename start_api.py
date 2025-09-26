from fastapi import FastAPI, HTTPException, Depends
import uvicorn
import os
from chatbot_functions import generate_answer, get_ground, stream_answer, get_ground_rag, generate_answer_rag, get_ground_highlight, stream_answer_rag
import chatbot_functions_mock as mock
# from auth import app as auth_app, get_current_active_user, User
import time
import argparse
import json 
from typing import List, Optional, Dict
from pydantic import BaseModel
from aixparag import RAGmain
from aixparag.data_preparation import extract_metadata
from aixparag.Retriever import Retriever
from aixparag.global_cache import _GLOBAL_RERANKERS 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import digitalhub as dh

parser = argparse.ArgumentParser()
parser.add_argument('--host', default="0.0.0.0")
parser.add_argument('--port', default=8018, type=int)
parser.add_argument('--openai_base_url', default='http://localhost:1235/v1')
parser.add_argument('--openai_key', default='ignore')
parser.add_argument('--openai_model', default='aixpa')
parser.add_argument('--openai_base_model', default='llama31')
parser.add_argument('--mock', action='store_true')
parser.add_argument('--data_artifact', default=None)
args = parser.parse_args()

start_api_openai_base_url = args.openai_base_url
start_api_openai_key = args.openai_key
start_api_openai_model = args.openai_model
start_api_openai_base_model = args.openai_base_model
start_api_mock = args.mock or os.environ.get("MOCK", "False").lower() == "true"
hf_token = os.environ.get("HF_TOKEN", "")

print("start_api_openai_base_url", start_api_openai_base_url)
print("start_api_mock", start_api_mock)

# aixpa-new-ground


_GLOBAL_RERANKERS["reranker_hf_model"] = 'nickprock/cross-encoder-italian-bert-stsb'


# instantiate FastApi application
app = FastAPI(version="0.0.1")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=['access-control-allow-origin'],
)

# add authentication
# app.mount("/auth", auth_app)

# add endpoint status
# create_status_endpoint(app)

# RAGmain.create_vector_store()

def read_txt_files(folder_path):
    contents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            print(filename)
            with open(file_path, "r", encoding="utf-8") as f:
                contents.append(f.read())
    return contents

start_time = time.time()

if not start_api_mock and args.data_artifact is not None:
    project = dh.get_or_create_project(os.environ.get("PROJECT_NAME"))
    project.get_artifact(args.data_artifact).download("RAG_documents", overwrite=True)

print(start_time, "Data Creation RAG")
if not start_api_mock:
    documents_list = read_txt_files("RAG_documents")
    print("Loaded " + str(len(documents_list)) + " documents")
    extract_metadata(documents_list)
    my_vector_store = RAGmain.create_vector_store()

    Retriever(vector_store=my_vector_store, reranker_model_name=_GLOBAL_RERANKERS["reranker_hf_model"])
    print(_GLOBAL_RERANKERS["reranker_hf_model"])




@app.get('/')
async def version():
    return {"version": app.version}

class TurnGenerationRequest(BaseModel):
    documents_list: List[str]
    dialogue_list: List[dict]
    user: str
    tone: str
    chatbot_is_first: bool
    
class TurnGroundRequest(BaseModel):
    documents_list: List[str]
    query: str
    options_number: int

class TurnGroundRequestRAG(BaseModel):
    documents_list: List[str]
    dialogue_list: List[dict]
    options_number: int
    chatbot_is_first: bool

class DataCreationRAG(BaseModel):
    documents_list: List[str]


# API METHODS

# @app.post('/turn_generation_old')
# def dialogue_generation_dynamic(request: TurnGenerationRequest):
#     start_time = time.time()
#     print(start_time, "Request turn generation")
#     return generate_answer(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first)


@app.post('/turn_generation')
def dialogue_generation_dynamic(request: TurnGenerationRequest):
    start_time = time.time()
    print(start_time, "Request turn generation")
    if start_api_mock:
        return mock.generate_answer(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first)
    return generate_answer_rag(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first, hf_token)

@app.post('/turn_stream')
def dialogue_generation_dynamic(request: TurnGenerationRequest):
    start_time = time.time()
    print(start_time, "Request turn Stream")
    if start_api_mock:
        return mock.stream_answer(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first)
    return stream_answer_rag(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first, hf_token)


@app.post('/turn_ground')
def dialogue_generation_dynamic(request: TurnGroundRequest):
    start_time = time.time()
    print(start_time, "Request ground")
    if start_api_mock:
        return mock.get_ground(request.documents_list, request.query, request.options_number)
    return get_ground_highlight(request.documents_list, request.query, request.options_number, hf_token)


@app.post('/turn_ground_rag')
def dialogue_generation_dynamic(request: TurnGroundRequestRAG):
    start_time = time.time()
    print(start_time, "Request ground RAG")
    return get_ground_rag(request.documents_list, request.dialogue_list, request.options_number, hf_token, request.chatbot_is_first)


if __name__ == '__main__':

    uvicorn.run("start_api:app", host=args.host, port=args.port, workers=1)
