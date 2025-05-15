from fastapi import FastAPI, HTTPException, Depends
import uvicorn
import os
from chatbot_functions import generate_answer, get_ground, stream_answer
import chatbot_functions_mock as mock
# from auth import app as auth_app, get_current_active_user, User
import time
import argparse
import json 
from typing import List, Optional, Dict
from pydantic import BaseModel

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

parser = argparse.ArgumentParser()
parser.add_argument('--host', default="0.0.0.0")
parser.add_argument('--port', default=8018, type=int)
parser.add_argument('--openai_base_url', default='http://localhost:1235/v1')
parser.add_argument('--openai_key', default='ignore')
parser.add_argument('--openai_model', default='aixpa')
parser.add_argument('--mock', action='store_true')
args = parser.parse_args()

start_api_openai_base_url = args.openai_base_url
start_api_openai_key = args.openai_key
start_api_openai_model = args.openai_model
start_api_mock = args.mock

print("start_api_openai_base_url", start_api_openai_base_url)
print("start_api_mock", start_api_mock)

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

# API METHODS

@app.post('/turn_generation')
def dialogue_generation_dynamic(request: TurnGenerationRequest):
    start_time = time.time()
    print(start_time, "Request turn generation")
    if start_api_mock:
        return mock.generate_answer(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first)
    return generate_answer(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first)

@app.post('/turn_stream')
def dialogue_generation_dynamic(request: TurnGenerationRequest):
    start_time = time.time()
    print(start_time, "Request turn Stream")
    if start_api_mock:
        return mock.stream_answer(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first)
    return stream_answer(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first)


@app.post('/turn_ground')
def dialogue_generation_dynamic(request: TurnGroundRequest):
    start_time = time.time()
    print(start_time, "Request ground")
    if start_api_mock:
        return mock.get_ground(request.documents_list, request.query, request.options_number)
    return get_ground(request.documents_list, request.query, request.options_number)


if __name__ == '__main__':

    uvicorn.run("start_api:app", host=args.host, port=args.port, workers=3)
