import json
from . import prompts
from string2string.similarity import LCSubsequenceSimilarity
from guidance import json as gen_json
import json
from guidance import system, user, assistant
from pydantic import BaseModel, field_validator
from typing import Optional
from .LanguageModel import VLLMModel

class MessageInfo(BaseModel):
    tassonomia: list[str]
    macro_ambito: list[str]
    luogo: list[str]
    model_config = dict(extra="forbid")

    @field_validator('tassonomia', 'macro_ambito', 'luogo', mode='after')
    @classmethod
    def to_lower(cls, value):
        if isinstance(value, str):
            return value.lower()
        return value

lcs_similarity = LCSubsequenceSimilarity()
    
def extract_and_parse_json(output_string: str):
    """
    Processes the output string from the model, extracts the JSON, and parses it into a dictionary.

    Args:
        output_string: The raw string output from the model, potentially starting with "```json"
                       and ending with "```".

    Returns:
        A dictionary if the JSON is successfully parsed.
        None if there is a formatting error or JSON decoding error.
    """
    json_string = output_string.strip()

    # Handle cases where the JSON is wrapped in markdown code block
    if json_string.startswith("```json") and json_string.endswith("```"):
        json_string = json_string[len("```json"): -len("```")].strip()
    elif json_string.startswith("```") and json_string.endswith("```"): # Generic code block
        # Try to parse it anyway, as it might just be a generic code block for JSON
        json_string = json_string[len("```"): -len("```")].strip()

    try:
        # Attempt to parse the JSON string
        parsed_data = json.loads(json_string)
        return parsed_data
    except json.JSONDecodeError as e:
        # If there's a JSON decoding error, print error and return None to indicate failure
        print(f"JSON decoding error: {e}")
        print(f"Problematic JSON string: \n---\n{json_string}\n---")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        print(f"Problematic string: \n---\n{json_string}\n---")
        return None

def print_query_res(query_res):
    for result in query_res:
        print(result.page_content)
        print(result.metadata)
        print("\n\n")

def get_similarity(list1, list2):
    threshold = 0.5
    similar_items= []
    if list1 == None:
        return None
    for gen_el in list1:
        for or_el in list2:
            score = lcs_similarity.compute(gen_el, or_el)
            # print(f"{gen_el} - {or_el} : {score}")
            if score >= threshold:
                similar_items.append(or_el)
    return list(set(similar_items))

def exctract_metadata(model, query, conversation, tassonomie, ambiti, luoghi):
    tassonomie_text = "\n".join([f"- {el}" for el in tassonomie])
    ambiti_text = "\n".join([f"- {el}" for el in ambiti])
    luoghi_text = "\n".join([f"- {el}" for el in luoghi])

    # replacing user query with expanded query
    # conv = conversation.copy()
    # conv[-1] = query

    conv = []
    conv.append(query)

    user_prompt = prompts.METADATA_USER_2.format(tassonomie = tassonomie_text,
                                                macro_ambiti = ambiti_text,
                                                location = luoghi_text,
                                                conversation = conv)
    
    response = model.generate_json(prompts.METADATA_SYS, [user_prompt], temperature=0.9, max_new_tokens=500)
    # postprocessing json
    loaded_json = json.loads(response)
    if 'tassonomia' in loaded_json and len(loaded_json['tassonomia']) > 0:
        if loaded_json['tassonomia'][0] != None:
            loaded_json['tassonomia'] = get_similarity(loaded_json['tassonomia'], tassonomie)
    if 'macro_ambito' in loaded_json and len(loaded_json['macro_ambito']) > 0:
        if loaded_json['macro_ambito'][0] != None:
            loaded_json['macro_ambito'] = get_similarity(loaded_json['macro_ambito'], ambiti)
    if 'luogo' in loaded_json and len(loaded_json['luogo']) > 0:
        if loaded_json['luogo'][0] != None:
            loaded_json['luogo'] = [el.lower() for el in loaded_json['luogo']]
    return loaded_json 



def expand_query(model, conversation: list) -> str:
    """
    Rewrites the last user message in a conversation to be fully self-contained,
    using context from previous turns.
    
    Args:
        model (VLLMModel): An instance of the VLLMModel class.
        conversation (list): List of dicts with keys 'speaker' and 'turn_text'.
        
    Returns:
        str: The rewritten query.
    """
    if not conversation:
        return "Error: The conversation list cannot be empty."

   
    last_turn = conversation[-1]
    context_turns = conversation[:-1]

    sys_prompt = prompts.QUERY_RWR_SYS
    user_prompt = prompts.QUERY_RWR_USER.format(conversation=context_turns,
                                                query=last_turn)
    

    messages = [user_prompt]
    try:
        # Call the VLLMModel's generate function
        rewritten_query = model.generate(
            sys_prompt=sys_prompt,
            conversation=messages,
            max_new_tokens=200,
            temperature=0.4
        ).strip()
        rewritten_query = rewritten_query.replace("<REWRITTEN_QUERY>", "").replace("</REWRITTEN_QUERY>", "")
        rewritten_query = rewritten_query.strip()
        return rewritten_query

    except Exception as e:
        print(f"Error during query rewriting: {e}")
        return "An error occurred while rewriting the query."


def no_rag_reply(model, query, conversation):
    user_prompt = prompts.REPLY_USER.format(user_message=query,
                                              conversation=conversation)
    response = model.generate(prompts.REPLY_SYS, user_prompt)
    return response

def no_rag_reply_hf(model, query, conversation):
    user_prompt = prompts.REPLY_USER_2.format(user_message=query)
    # conversation.append(user_prompt)
    response = model.generate(prompts.REPLY_SYS, conversation)
    return response

def rag_reply(model, query, conversation, retrieved_data):
    user_prompt = prompts.REPLY_RAG_USER.format(user_message=query,
                                            conversation=conversation,
                                            retrieved_actions=retrieved_data)
    response = model.generate(prompts.REPLY_RAG_SYS, user_prompt)
    return response

def rag_reply_hf(model, query, conversation, retrieved_data):
    user_prompt = prompts.REPLY_RAG_USER_2.format(user_message=query,
                                            retrieved_actions=retrieved_data)

    conv = conversation.copy()
    conv[-1] = user_prompt
    response = model.generate(prompts.REPLY_RAG_SYS, conv)
    return response

def sql_planner(model, query):
    user_prompt = prompts.SQL_PLANNER_USER.format(query=query)
    response = model.generate(prompts.SQL_PLANNER_SYS, [user_prompt])
    return response
