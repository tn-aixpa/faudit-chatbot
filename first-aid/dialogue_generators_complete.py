import os
from fastapi import HTTPException, status
import math
from tools import chunker, dialogue, retrieval, openai_gpt, span
import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

with open('keys.json') as keys_file:
    keys = json.load(keys_file)

with open('methods_full_dialogue/prompts.json') as keys_file:
    prompts = json.load(keys_file)

def get_complete_generation_options():
    generation_options = [
        {
            "generation_method": "hatedemics_en",
            "description": "Generates a dialogue in English between a Hater and an Operator. Ground text is only provided on Operator's turns.",
            "roles": [
                {
                    "label": "speaker_1",
                    "name": "Hater",
                    "ground": False
                },
                {
                    "label": "speaker_2",
                    "name": "Operator",
                    "ground": True
                }
            ]
        },
        {
            "generation_method": "hatedemics_it",
            "description": "Generates a dialogue in Italian between a Hater and an Operator. Ground text is only provided on Operator's turns.",
            "roles": [
                {
                    "label": "speaker_1",
                    "name": "Hater",
                    "ground": False
                },
                {
                    "label": "speaker_2",
                    "name": "Operator",
                    "ground": True
                }
            ]       
        },
        {
            "generation_method": "hatedemics_es",
            "description": "Generates a dialogue in Spanish between a Hater and an Operator. Ground text is only provided on Operator's turns.",
            "roles": [
                {
                    "label": "speaker_1",
                    "name": "Hater",
                    "ground": False
                },
                {
                    "label": "speaker_2",
                    "name": "Operator",
                    "ground": True
                }
            ]      
        },
        {
            "generation_method": "hatedemics_mt",
            "description": "Generates a dialogue in Maltese between a Hater and an Operator. Ground text is only provided on Operator's turns.",
            "roles": [
                {
                    "label": "speaker_1",
                    "name": "Hater",
                    "ground": False
                },
                {
                    "label": "speaker_2",
                    "name": "Operator",
                    "ground": True
                }
            ]     
        },
        {
            "generation_method": "hatedemics_pl",
            "description": "Generates a dialogue in Polish between a Hater and an Operator. Ground text is only provided on Operator's turns.",
            "roles": [
                {
                    "label": "speaker_1",
                    "name": "Hater",
                    "ground": False
                },
                {
                    "label": "speaker_2",
                    "name": "Operator",
                    "ground": True
                }
            ]      
        },
        {
            "generation_method": "aixpa_it",
            "description": "Generates a dialogue aixpa",
            "roles": [
                {
                    "label": "speaker_1",
                    "name": "Impiegato",
                    "ground": False
                },
                {
                    "label": "speaker_2",
                    "name": "Chatbot",
                    "ground": True
                }
            ]      
        }           

    ]
    return generation_options



def generate_dialogue_complete(generation_opton, *args, **kwargs):
    try:
        # Get the function object from the global namespace
        func = globals()[generation_opton]
        # Call the function with the provided arguments and return the result
        return func(*args, **kwargs)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Generation option {generation_opton} not found.")
        
    except TypeError as e:
        return str(e)

def hatedemics_en(documents_list, num_turns):
    return(hatedemics(documents_list, num_turns, "English"))

def hatedemics_it(documents_list, num_turns):
    return(hatedemics(documents_list, num_turns, "Italian"))

def hatedemics_pl(documents_list, num_turns):
    return(hatedemics(documents_list, num_turns, "Polish"))

def hatedemics_mt(documents_list, num_turns):
    return(hatedemics(documents_list, num_turns, "Maltese"))

def hatedemics_es(documents_list, num_turns):
    return(hatedemics(documents_list, num_turns, "Spanish"))


def hatedemics (documents_list, num_turns, language):
    roles = {
        "speaker_1": "Hater",
        "speaker_2": "Operator",
        "Hater": "speaker_1",
        "Operator": "speaker_2"
        } 
    ground_required_dict =  {
        "speaker_1": False,
        "speaker_2": True
        } 
    if not num_turns:
        num_turns = 6
    
    # documents_list_with_target has documents with the structure "Article: article content\nTarget: targeted minority"
    documents_list_with_target = documents_list.copy()

    # while in document_list I only keep the articles content
    documents_list = []
    # I assume all documents contain the same target
    targets = []
    for doc in documents_list_with_target:
        if 'Article:' in doc and 'Target:' in doc:
            documents_list.append(doc.split('Article:')[1].split('Target:')[0].strip())
            
            target = doc.split('Target:')[1].strip()
            targets.append(target)
        else:
            documents_list.append(doc)
    if len(set(targets)) > 1:
        raise Exception("More than one target present")
    
    elif len(set(targets)) == 0:
        raise Exception("No target present")
    
    else:
        target = targets[0].strip()
        if target == 'POC':
            target = 'Black people'

    chunks = chunker.Chunker_SaT(
            documents_list  = documents_list, 
            language = language,
            window_size = 1
            )    

    generator = openai_gpt.GenerationModelChat(
        model_name = "gpt-4o-mini-2024-07-18",  
        access_token = keys['OPENAI_API_HATEDEMICS']
        )
 
    retr = retrieval.Retriever_bm25(
        knowledge_base=chunks,
        name="BM25",
        top_k=3
        )
    
    prompt = prompts["HATEDEMICS"]
    prompt = prompt.replace("<LANGUAGE>", language)
    prompt = prompt.replace("<TARGET>", target)
    prompt = prompt.replace("<TURNS_NUMBER>", str(num_turns))
    prompt = prompt.replace("<SPEAKER_1>", roles["speaker_1"])
    prompt = prompt.replace("<SPEAKER_2>", roles["speaker_2"])
    prompt = "\n\n".join(documents_list) + prompt
    
    question_prompt = []
    sys_prompt = {'role': 'system',
                    'content': prompt}
    question_prompt.append(sys_prompt)
    generated_text = generator.generate_text(question_prompt)
    generated_text = generated_text.replace("```json", "")
    generated_text = generated_text.replace("```", "")
    data = json.loads(generated_text)

    turns_list = []

    for turn in data["dialogue"]:
        turn_dict = dict()
        turn_dict["ground"] =  []
        speaker_id = roles[turn["speaker"]]
        if ground_required_dict[speaker_id]: 

            if language == 'Maltese':
                # TO DO: make it deterministic
                mt_query = turn["text"]
                translation_prompt = {'role': 'system', 'content': "Translate the following text in English: {}. Answer only with the English translation".format(mt_query)}
                query = generator.generate_text([translation_prompt])

            else:
                query = turn["text"]

            grounds = retr.retrieve(query)
            for ground in grounds:
                ground_info = dict()
                # print(documents_list_with_target)
                index_start, index_end = span.find_indexes(documents_list_with_target[ground.metadata["document_id"]],ground.text)
                # print("\n________________________\nhighlighted ground: ", documents_list_with_target[ground.metadata["document_id"]][index_start:index_end])
                ground_info["text"] =  ground.text
                ground_info["file_index"] = ground.metadata["document_id"]
                ground_info["offset_start"] = index_start
                ground_info["offset_end"] = index_end
                turn_dict["ground"].append(ground_info)
             
        turn_dict["speaker"] =  speaker_id
        turn_dict["turn_text"] = turn["text"]
        turns_list.append(turn_dict)
    
    return(turns_list)


def aixpa_it(documents_list, num_turns):
    return(aixpa(documents_list, num_turns, "Italian"))


def aixpa(documents_list, num_turns, language):
    roles = {
        "speaker_1": "Impiegato",
        "speaker_2": "Chatbot",
        "Impiegato": "speaker_1",
        "Chatbot": "speaker_2"
        } 
    ground_required_dict =  {
        "speaker_1": False,
        "speaker_2": True
        } 
    if not num_turns:
        num_turns = 6
    
    # for the prompt only I keep the structure "Article: article content\nTarget: targeted minority"
    documents_list_with_target = documents_list.copy()
    # while in document_list I only keep the articles content
    
    documents_list = []
    for doc in documents_list_with_target:
        if 'Article:' in doc and 'Target:' in doc:
            documents_list.append(doc.split('Article:')[1].split('Target:')[0].strip())
        else:
            documents_list.append(doc)

    # chunks = chunker.Chunker_SaT(
    #         documents_list  = documents_list, 
    #         language = language,
    #         window_size = 1
    #         )  
    chunks = chunker.Chunker_llama_index(
        documents_list  = documents_list, 
        chunk_size    = 70,
        chunk_overlap = 25
        )  
    generator = openai_gpt.GenerationModelChat(
        model_name = "gpt-4o-mini-2024-07-18",  
        access_token = keys['OPENAI_API_HATEDEMICS']
        )
 
    # retr = retrieval.Retriever_bm25(
    #     knowledge_base=chunks,
    #     name="BM25",
    #     top_k=3
    #     )
    retr = retrieval.Retriever_llamaindex_bm25(
        knowledge_base=chunks,
        name="BM25",
        top_k=3
        )
    prompt = "given the following text, make a dialogue in <LANGUAGE> between a worker from the public administration who is trying to fill the form and wants to know more informations and an agent who speaks formally and is capable to help and to give clarification when something is unclear to the worker. You will be provided with an article (delimited with XML tag). Use only information from the article to write the dialogue. The dialogue must include at most <TURNS_NUMBER> exchanges. Return the response in JSON format where each turn is clearly marked by the speaker. Use the following format: { \"dialogue\": [ {\"speaker\": \"<SPEAKER_1>\", \"text\": \"[Dialogue]\"}, {\"speaker\": \"<SPEAKER_2>\", \"text\": \"[Dialogue]\"}, ... ] } Ensure the structure remains consistent throughout.\". The citizen may ask question whose answers are negative"
    # prompt = prompts["AIXPA"]
    prompt = prompt.replace("<LANGUAGE>", language)
    prompt = prompt.replace("<TURNS_NUMBER>", str(num_turns))
    prompt = prompt.replace("<SPEAKER_1>", roles["speaker_1"])
    prompt = prompt.replace("<SPEAKER_2>", roles["speaker_2"])
    prompt = "\n\n".join(documents_list_with_target) + prompt
    
    question_prompt = []
    sys_prompt = {'role': 'system',
                    'content': prompt}
    question_prompt.append(sys_prompt)
    generated_text = generator.generate_text(question_prompt)
    generated_text = generated_text.replace("```json", "")
    generated_text = generated_text.replace("```", "")
    data = json.loads(generated_text)

    turns_list = []

    for turn in data["dialogue"]:
        turn_dict = dict()
        turn_dict["ground"] =  []
        speaker_id = roles[turn["speaker"]]
        if ground_required_dict[speaker_id]: 

            if language == 'Maltese':
                # TO DO: make it deterministic
                mt_query = turn["text"]
                translation_prompt = {'role': 'system', 'content': "Translate the following text in English: {}. Answer only with the English translation".format(mt_query)}
                query = generator.generate_text([translation_prompt])

            else:
                query = turn["text"]

            grounds = retr.retrieve(query)
            
            for ground in grounds:
                ground_info = dict()
                index_start, index_end = span.find_indexes(documents_list_with_target[ground.metadata["document_id"]],ground.text)
                ground_info["text"] =  ground.text
                ground_info["file_index"] = ground.metadata["document_id"]
                ground_info["offset_start"] = index_start
                ground_info["offset_end"] = index_end
                turn_dict["ground"].append(ground_info)
             
        turn_dict["speaker"] =  speaker_id
        turn_dict["turn_text"] = turn["text"]
        turns_list.append(turn_dict)
    
    return(turns_list)