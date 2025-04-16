import os
from tools import chunker, dialogue, retrieval, openai_gpt, span, prompt_composer
import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from typing import Callable, Dict, List, Union
import random
# from lorax import Client
from openai import OpenAI


# with open('keys.json') as keys_file:
#     keys = json.load(keys_file)

# with open('methods_dynamic_dialogue/prompts.json') as keys_file:
#     prompts = json.load(keys_file)


def get_dynamic_generation_options():
    generation_options = [
        {
            "generation_method": "aixpa_dynamic",
            "description": "genera dialoghi per supportare operatori comunali",
            "endpoint": "/dialogue_generation_dynamic/",
            "roles": [
                {
                    "label": "speaker_1",
                    "name": "Public operator",
                    "ground": False
                },
                {
                    "label": "speaker_2",
                    "name": "ChatBot",
                    "ground": True
                }
            ]
        },
    ]
    return generation_options


def generate_dialogue_dynamic(generation_opton, *args, **kwargs):
    # print(generation_opton)
    # print(*args)
    # print(*kwargs)
    # print(globals())
    # try:
        # Get the function object from the global namespace
    func = globals()[generation_opton]
    
    # Call the function with the provided arguments and return the result
    return func(*args, **kwargs)
    # except KeyError:
    #     return f"Generation option {generation_opton} not found."
    # except TypeError as e:
    #     return str(e)


def aixpa_dynamic(documents_list, dialogue_list, speaker, options_number, manual_selected_grounds):
    from start_api import start_api_kubeai_host
    
    user = "operatore"
    style = "informal"
    
    adapter = "aixpa"
    
    speaker = "speaker_2"


    roles = {
        "speaker_1": "user",
        "speaker_2": "assistant",
        "user": "speaker_1",
        "assistant": "speaker_2"
        } 
    ground_required_dict =  {
        "speaker_1": False,
        "speaker_2": True
        } 
    if not options_number:
        options_number = 1

    # options_number = 3

    chunks = chunker.Chunker_llama_index(
            documents_list  = documents_list, 
            chunk_size    = 200,
            chunk_overlap = 0
            )   

    client = OpenAI(
        base_url = start_api_kubeai_host,
        api_key='ollama', # required, but unused
    )

    # print('KUBEAI HOST IS ', start_api_kubeai_host)

    dial= dialogue.Dialogue(turns = dialogue_list)

    # retr = retrieval.Embedder_BGE_m3_kubeai(
    #     knowledge_base=chunks,
    #     name="bge-m3",
    #     top_k=options_number
    #     )
    retr = retrieval.Retriever_bm25(
        knowledge_base=chunks,
        name="BM25",
        top_k=options_number
        )

    # PROMPT MANAGEMENT
    prompt = '''You are an helpful assistant from the public administration, use a <<STYLE>> tone.
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

    if style == "informale":
        prompt = prompt.replace("<<STYLE>>", "informal")

    if style == "formale":
        prompt = prompt.replace("<<STYLE>>", "formal")
    
    
    


    # Chatbot turn
    if speaker == "speaker_2":
        speaker2_prompt_list = []
        
        if len(manual_selected_grounds) > 0:
            # chunks = []
            # chunks.append(" ".join(manual_selected_grounds))
            prompt = prompt.replace("<<DOCUMENTS>>", "\n".join(manual_selected_grounds))   
            
        else:
            ###### genera con un chunk
        #     query = dial.get_last_turn()
        #     chunks = retr.retrieve(query)
        #     chunks = random.sample(chunks, options_number)

        # for ground in chunks:
        #     if len(manual_selected_grounds) > 0:
        #         prompt = prompt.replace("<<DOCUMENTS>>", ground)
        #     else:
        #         prompt = prompt.replace("<<DOCUMENTS>>", ground.text.strip())          
            
            ###### genera con tutto il documento
            prompt = prompt.replace("<<DOCUMENTS>>", "\n".join(documents_list))   
        
        sys_prompt = [{"role": "system", "content": prompt}]
        speaker2_prompt_list.append(sys_prompt)

    chat = []

    if dial.turn_numbers > 0:
        for j, msg in enumerate(dialogue_list):
            if speaker == 'speaker_1':
                role = "user"
            
            elif speaker == 'speaker_2':
                role = 'assistant'

            chat.append({"role": role, "content": msg['turn_text']})


    # GENERATE
    generation_outputs = []


        

    if speaker == "speaker_2":    
        for p in speaker2_prompt_list:
            input_chat = p + chat
            print('------------------------------INPUT CHAT------------------------------')
            print(input_chat)

            # TODO: change model paths

            
            message = client.chat.completions.create(
                model="aixpa",
                messages=input_chat,
                temperature=0.6,
                max_completion_tokens=1000
            ).choices[0].message.content

            # message = message.replace("\n\n","")
            # message = message.split('assistant')[-1]
            # print(generated_text)
            
            generation_outputs.append(message)

    # PREPARE OUTPUT OPTIONS

    next_turns_candidates = []
    
    for candidate_index, text_option in enumerate(generation_outputs):
        grounds_list = []
        if ground_required_dict[speaker]:
            if len(manual_selected_grounds) == 0:
                
                


                for doc_index, doc in enumerate(documents_list):
                    ground_info = dict()
                    print("________")
                    print(doc_index, documents_list[doc_index][:50])
                    print("________")
                    parital_chunks = chunker.Chunker_llama_index(
                        documents_list  = [documents_list[doc_index]], 
                        chunk_size    = 200,
                        chunk_overlap = 0
                        )  

                    partial_retr = retrieval.Retriever_bm25(
                        knowledge_base=parital_chunks,
                        name="BM25",
                        top_k=options_number
                        )
                    #full doc
                    doc_chunks = partial_retr.retrieve(text_option)



                    # for chunk_index, chunk_ground in chunks:
                    g_text =   doc_chunks[0].text
                    print("________")
                    print(doc_index, documents_list[doc_index][:50],g_text)
                    print("________")                   
                    index_start, index_end = span.find_indexes(documents_list[doc_index],g_text)

                    #generazione da chunks
                    # g_text =   chunks[candidate_index].text
                    # g_doc = chunks[candidate_index].metadata["document_id"]
                    # index_start, index_end = span.find_indexes(documents_list[g_doc],g_text)

                    ground_info["text"] = g_text
                    ground_info["file_index"] = doc_index
                    ground_info["offset_start"] = index_start
                    ground_info["offset_end"] = index_end
                    grounds_list.append(ground_info)
            
        next_turns_candidates.append(
            
            {
                "speaker": speaker,
                "turn_text": text_option,
                "ground": grounds_list

                
            }        
        )

    print(next_turns_candidates)
    return next_turns_candidates



# def aixpa_dynamic(documents_list, dialogue_list, speaker, options_number, manual_selected_grounds):
#     from start_api import start_api_kubeai_host
    
#     user = "operatore"
#     style = "informal"
    
#     adapter = "aixpa"
    
#     speaker = "speaker_2"


#     roles = {
#         "speaker_1": "user",
#         "speaker_2": "assistant",
#         "user": "speaker_1",
#         "assistant": "speaker_2"
#         } 
#     ground_required_dict =  {
#         "speaker_1": False,
#         "speaker_2": True
#         } 
#     if not options_number:
#         options_number = 1

#     # options_number = 3

#     chunks = chunker.Chunker_llama_index(
#             documents_list  = documents_list, 
#             chunk_size    = 200,
#             chunk_overlap = 0
#             )   

#     client = OpenAI(
#         base_url = start_api_kubeai_host,
#         api_key='ollama', # required, but unused
#     )

#     # print('KUBEAI HOST IS ', start_api_kubeai_host)

#     dial= dialogue.Dialogue(turns = dialogue_list)

#     # retr = retrieval.Embedder_BGE_m3_kubeai(
#     #     knowledge_base=chunks,
#     #     name="bge-m3",
#     #     top_k=options_number
#     #     )
#     retr = retrieval.Retriever_bm25(
#         knowledge_base=chunks,
#         name="BM25",
#         top_k=options_number
#         )

#     # PROMPT MANAGEMENT
#     prompt = '''You are an helpful assistant from the public administration, use a <<STYLE>> tone.
#     The user is a <<ROLE>>.
#     Your task is to provide a relevant answer to the user using the provided evidence and past dialogue history.
#     The evidence is contained in <document> tags.Be proactive asking for the information needed to help the user.
#     When you receive a question, answer by referring exclusively to the content of the document.
#     Answer in Italian.
#     <document><<DOCUMENTS>></document>'''


#     if user == "cittadino":
#         prompt = prompt.replace("<<ROLE>>", "Citizen")

#     if user == "operatore":
#         prompt = prompt.replace("<<ROLE>>", "Public Operator")

#     if style == "informale":
#         prompt = prompt.replace("<<STYLE>>", "informal")

#     if style == "formale":
#         prompt = prompt.replace("<<STYLE>>", "formal")
    
    
    


#     # Chatbot turn
#     if speaker == "speaker_2":
#         speaker2_prompt_list = []
        
#         if len(manual_selected_grounds) > 0:
#             chunks = []
#             chunks.append(" ".join(manual_selected_grounds))
            
#         else:
#             ###### genera con un chunk
#         #     query = dial.get_last_turn()
#         #     chunks = retr.retrieve(query)
#         #     chunks = random.sample(chunks, options_number)

#         # for ground in chunks:
#         #     if len(manual_selected_grounds) > 0:
#         #         prompt = prompt.replace("<<DOCUMENTS>>", ground)
#         #     else:
#         #         prompt = prompt.replace("<<DOCUMENTS>>", ground.text.strip())          
            
#             ###### genera con tutto il documento
#             prompt = prompt.replace("<<DOCUMENTS>>", "\n".join(documents_list))   
        
#             sys_prompt = [{"role": "system", "content": prompt}]
#             speaker2_prompt_list.append(sys_prompt)

#     chat = []

#     if dial.turn_numbers > 0:
#         for j, msg in enumerate(dialogue_list):
#             if speaker == 'speaker_1':
#                 role = "user"
            
#             elif speaker == 'speaker_2':
#                 role = 'assistant'

#             chat.append({"role": role, "content": msg['turn_text']})


#     # GENERATE
#     generation_outputs = []


        

#     if speaker == "speaker_2":    
#         for p in speaker2_prompt_list:
#             input_chat = p + chat
#             print('------------------------------INPUT CHAT------------------------------')
#             print(input_chat)

#             # TODO: change model paths

            
#             message = client.chat.completions.create(
#                 model="aixpa",
#                 messages=input_chat,
#                 temperature=0.6,
#                 max_completion_tokens=1000
#             ).choices[0].message.content

#             # message = message.replace("\n\n","")
#             # message = message.split('assistant')[-1]
#             # print(generated_text)
            
#             generation_outputs.append(message)

#     # PREPARE OUTPUT OPTIONS

#     next_turns_candidates = []
    
#     for candidate_index, text_option in enumerate(generation_outputs):
#         grounds_list = []
#         if ground_required_dict[speaker]:
#             if len(manual_selected_grounds) == 0:
#                 ground_info = dict()
                
#                 #full doc
#                 chunks = retr.retrieve(text_option)

#                 # for chunk_index, chunk_ground in chunks:
#                 g_text =   chunks[0].text
#                 g_doc = chunks[0].metadata["document_id"]
#                 index_start, index_end = span.find_indexes(documents_list[g_doc],g_text)

#                 #generazione da chunks
#                 # g_text =   chunks[candidate_index].text
#                 # g_doc = chunks[candidate_index].metadata["document_id"]
#                 # index_start, index_end = span.find_indexes(documents_list[g_doc],g_text)

#                 ground_info["text"] = g_text
#                 ground_info["file_index"] = g_doc
#                 ground_info["offset_start"] = index_start
#                 ground_info["offset_end"] = index_end
#                 grounds_list.append(ground_info)
            
#         next_turns_candidates.append(
            
#             {
#                 "speaker": speaker,
#                 "turn_text": text_option,
#                 "ground": grounds_list

                
#             }        
#         )

#     print(next_turns_candidates)
#     return next_turns_candidates