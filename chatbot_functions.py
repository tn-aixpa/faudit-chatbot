import os
from tools import chunker, dialogue, retrieval, span
import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from typing import Callable, Dict, List, Union
import random
# from lorax import Client
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

def create_chat_prompt (documents_list, dialogue_list, user, tone, chatbot_is_first):
    
    dial= dialogue.Dialogue(turns = dialogue_list)

    # Make Prompt
    prompt = '''You are an helpful assistant from the public administration, use a <<TONE>> tone.
    The user is a <<ROLE>>.
    Your task is to provide a relevant answer to the user using the provided evidence and past dialogue history.
    The evidence is contained in <document> tags.Be proactive asking for the information needed to help the user.
    When you receive a question, answer by referring exclusively to the content of the document.
    Answer in Italian.
    <document><<DOCUMENTS>></document>'''


    if user == "cittadino":
        prompt = prompt.replace("<<ROLE>>", "Citizen")

    if user == "operatore":
        prompt = prompt.replace("<<ROLE>>", "Public Operator")

    if tone == "informale":
        prompt = prompt.replace("<<TONE>>", "informal")

    if tone == "formale":
        prompt = prompt.replace("<<TONE>>", "formal")
    
    
    prompt = prompt.replace("<<DOCUMENTS>>", "\n".join(documents_list))
    
    # Make input
    chatbot_prompt_list = []

    sys_prompt = {"role": "system", "content": prompt}
    chatbot_prompt_list.append(sys_prompt)

    if dial.turn_numbers > 0:
        chat_list = []
        for j, msg in enumerate(dialogue_list):
            if chatbot_is_first:
                if j % 2 == 0:
                    role = "assistant"
                else:
                    role = "user"
            else:
                if j % 2 == 0:
                    role = "user"
                else:
                    role = "assistant"

            chatbot_prompt_list.append({"role": role, "content": msg['turn_text']})
    
    return chatbot_prompt_list

def stream_answer(documents_list, dialogue_list, user, tone, chatbot_is_first):
    from start_api import start_api_openai_base_url, start_api_openai_key, start_api_openai_model
    
    chatbot_prompt_list = create_chat_prompt (documents_list, dialogue_list, user, tone, chatbot_is_first)

   
    client = OpenAI(
        base_url = start_api_openai_base_url,
        api_key=start_api_openai_key
    )

    
    # Generate next turn
    stream = client.chat.completions.create(
        model=start_api_openai_model,
        messages=chatbot_prompt_list,
        temperature=0.6,
        stream=True
    )  
    
    async def event_generator():
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return StreamingResponse(event_generator(), media_type="application/json")


def generate_answer(documents_list, dialogue_list, user, tone, chatbot_is_first):
    from start_api import start_api_openai_base_url, start_api_openai_key, start_api_openai_model
    
    chatbot_prompt_list = create_chat_prompt (documents_list, dialogue_list, user, tone, chatbot_is_first)
    
    client = OpenAI(
        base_url = start_api_openai_base_url,
        api_key=start_api_openai_key
    )

    # Generate next turn
    message = client.chat.completions.create(
        model=start_api_openai_model,
        messages=chatbot_prompt_list,
        temperature=0.6,
        # max_completion_tokens=1000
    ).choices[0].message.content

    
    next_turn = {
            "turn_text": message,                
        }        
    
    return next_turn




def get_ground(documents_list, query, options_number):

    chunks = chunker.Chunker_llama_index(
            documents_list  = documents_list, 
            chunk_size    = 200,
            chunk_overlap = 0
            )   

    retr = retrieval.Retriever_bm25(
        knowledge_base=chunks,
        name="BM25",
        top_k=options_number
        )

    retrieved_chunks = retr.retrieve(query)

    grounds_list = []
    for chunk in retrieved_chunks:

        ground_info = dict()
        
        g_text =   chunk.text
        g_doc = chunk.metadata["document_id"]
        index_start, index_end = span.find_indexes(documents_list[g_doc],g_text)
        
        ground_info["text"] = g_text
        ground_info["file_index"] = g_doc
        ground_info["offset_start"] = index_start
        ground_info["offset_end"] = index_end
        grounds_list.append(ground_info)
        
    return grounds_list


