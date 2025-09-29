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
from aixparag.RAGmain import rag_answer,rag_answer_highlight
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_chat_prompt (documents_list, dialogue_list, user, tone, chatbot_is_first):
    
    dial= dialogue.Dialogue(turns = dialogue_list)

    # Make Prompt
    # prompt = '''You are an helpful assistant from the public administration, use a <<TONE>> tone.
    # The user is a <<ROLE>>.
    # Your task is to provide a relevant answer to the user using the provided evidence and past dialogue history.
    # The evidence is contained in <document> tags.Be proactive asking for the information needed to help the user.
    # When you receive a question, answer by referring exclusively to the content of the document.
    # Answer in Italian.
    # <document><<DOCUMENTS>></document>'''

    prompt = '''You are an helpful assistant from the public administration, use a <<TONE>> tone.
    The user is a <<ROLE>>.
    Your task is to provide a relevant answer to the user using the provided evidence and past dialogue history.
    The evidence is contained in <document> tags.Be proactive asking for the information needed to help the user.
    When you receive a question, answer by referring exclusively to the content of the document. 
    If you need more information to fullfill the user request, ask the user a specific question to clarify what they need brfore asking.
    Ask for every information you need to write an action or a plan when the user request you to do it
    Avoid repetitions and repeating the same information in different ways.
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
        temperature=0.5,
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
        temperature=0.5,
        # max_completion_tokens=1000
    ).choices[0].message.content

    
    next_turn = {
            "turn_text": message,                
        }        
    
    return next_turn


def stream_answer_rag(documents_list, dialogue_list, user, tone, chatbot_is_first, hf_token):
    from start_api import start_api_openai_base_url, start_api_openai_key, start_api_openai_model
    
    output_rag = get_ground_rag(documents_list, dialogue_list, 5, hf_token, chatbot_is_first) #the number of item (5) do nothing
    ground_rag = []
    for g in output_rag:
        ground_rag.append(g["text"])
    # print("---------------------------------")
    # print("---------------------------------")
    # print("GROUND RAG", ground_rag)
    # print("---------------------------------")
    # print("---------------------------------")

    chatbot_prompt_list = create_chat_prompt(ground_rag, dialogue_list, user, tone, chatbot_is_first)
   
    client = OpenAI(
        base_url = start_api_openai_base_url,
        api_key=start_api_openai_key
    )

    
    # Generate next turn
    stream = client.chat.completions.create(
        # model="c320",
        # model="aixpa-new-ground",
        model = start_api_openai_model,
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


def generate_answer_rag(documents_list, dialogue_list, user, tone, chatbot_is_first, hf_token):
    from start_api import start_api_openai_base_url, start_api_openai_key, start_api_openai_model
    
    output_rag = get_ground_rag(documents_list, dialogue_list, 5, hf_token, chatbot_is_first) #the number of item (5) do nothing
    # logger.info("RAG output:")
    # logger.info(output_rag)
    ground_rag = []
    for g in output_rag:
        ground_rag.append(g["text"])
    # logger.info("RAG ground:")
    # logger.info(ground_rag)
    chatbot_prompt_list = create_chat_prompt(ground_rag, dialogue_list, user, tone, chatbot_is_first)
    logger.info("Chatbot prompt list:")
    logger.info(chatbot_prompt_list)
    # start = datetime.datetime.now().timestamp()
    client = OpenAI(
        base_url = start_api_openai_base_url,
        api_key=start_api_openai_key
    )
    # Generate next turn
    message = client.chat.completions.create(
        # model="c320",
        # model="aixpa-new-ground",
        model = start_api_openai_model,
        # model="mask048",
        # model="aixpa",
        messages=chatbot_prompt_list,
        temperature=0.6,
        # max_completion_tokens=1000
    ).choices[0].message.content
    # end = datetime.datetime.now().timestamp()
    # print("GENERATE", (end - start))

    
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


def get_ground_highlight(documents_list, query, options_number, hf_token):

    retrieved_chunks = rag_answer_highlight(documents_list,query, options_number, hf_token)

    grounds_list = [] 

    # print("retrieved_chunks LIST")
    # print(retrieved_chunks)

    for chunk in retrieved_chunks:
        chunk = chunk.strip()
        if chunk.startswith("COMUNE"):
            chunk_clean =  chunk.split("\n", 1)[1] if "\n" in chunk else ""
        else:
            chunk_clean = chunk

        found = False
        for i, doc in enumerate(documents_list):
            index_start, index_end = span.find_indexes(documents_list[i], chunk_clean.strip())
            # logger.info("INDEXES", index_start, index_end, "DOC INDEX", i)
            # logger.info("DOC INDEX", i,"First 100 chars of doc:", documents_list[i][:100])
            # Only add if indexes are valid
            if index_start is not None and index_end is not None:
                ground_info = {
                    "text": chunk_clean,
                    "file_index": i,
                    "offset_start": index_start,
                    "offset_end": index_end
                }
                grounds_list.append(ground_info)
                found = True
        
        if not found:
            grounds_list.append({
                "text": chunk_clean,
                "file_index": 0,  # or -1 if you prefer
                "offset_start": 0,
                "offset_end": 1
            })

    return grounds_list


def get_ground_rag(documents_list, dialogue_list, options_number, hf_token, chatbot_is_first):
    
    query = dialogue_list[-1]['turn_text']
    
    retrieved_chunks = rag_answer(documents_list, dialogue_list, query, options_number, hf_token, chatbot_is_first)
    # logger.info("Retrieved chunks (GROUND RAG):")
    # logger.info(retrieved_chunks)
    grounds_list = [] 

    for chunk in retrieved_chunks:
        
        if chunk.startswith("COMUNE"):
            chunk_clean =  chunk.split("\n", 1)[1] if "\n" in chunk else ""
        else:
            chunk_clean = chunk
        
        found = False
        for i, doc in enumerate(documents_list):
            index_start, index_end = span.find_indexes(documents_list[i], chunk_clean)

            # Only add if indexes are valid
            if index_start is not None and index_end is not None:
                ground_info = {
                    "text": chunk,
                    "file_index": i,
                    "offset_start": index_start,
                    "offset_end": index_end
                }
                grounds_list.append(ground_info)
                found = True

        if not found:
            grounds_list.append({
                "text": chunk,
                "file_index": 0,  # or -1 if you prefer
                "offset_start": 0,
                "offset_end": 1
            })
    return grounds_list


