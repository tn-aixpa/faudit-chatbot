from fastapi import FastAPI, HTTPException, Depends
import uvicorn
import os
from chatbot_functions import generate_answer, get_ground, stream_answer
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
parser.add_argument('--kubeai_host', default='http://localhost:1235/v1')
args = parser.parse_args()

start_api_kubeai_host = args.kubeai_host


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
    return generate_answer(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first)

@app.post('/turn_stream')
def dialogue_generation_dynamic(request: TurnGenerationRequest):
    start_time = time.time()
    print(start_time, "Request turn Stream")
    return stream_answer(request.documents_list, request.dialogue_list, request.user, request.tone, request.chatbot_is_first)


@app.post('/turn_ground')
def dialogue_generation_dynamic(request: TurnGroundRequest):
    start_time = time.time()
    print(start_time, "Request ground")
    return get_ground(request.documents_list, request.query, request.options_number)


if __name__ == '__main__':

    uvicorn.run("start_api:app", host=args.host, port=args.port, workers=3)
