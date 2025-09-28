from .LanguageModel import GroqModel, HuggingFaceModel, VLLMModel
from .VectorStoreQdrant import VectorStore
from .Retriever import Retriever
from .data_preparation import extract_metadata
from langchain_core.documents import Document
# from . import prompts
import pandas as pd
import json
from . import utils
from huggingface_hub import login
from guidance.models import Transformers
from guidance.chat import ChatTemplate
import pickle
from .global_cache import _GLOBAL_RERANKERS, _GLOBAL_AMBITI, _GLOBAL_TASSONOMIE, _GLOBAL_VECTOR_STORE
import dill
# from qdrant_client import QdrantClient
# from langchain.vectorstores import Qdrant



def find_cities_in_first_lines(documents):
    """
    Find cities (from cities file) mentioned in the first line of documents.

    Args:
        documents (list[str]): List of text documents.

    Returns:
        list[str]: List of cities found in the first lines of the documents.
    """
    # Load cities from file
    
    with open("aixparag/data/cities.txt", "r", encoding="utf-8") as f:
        cities = [line.strip() for line in f if line.strip()]
    
    found_cities = []

    for doc in documents:
        first_line = doc.splitlines()[0] if doc else ""
        for city in cities:
            # Case-insensitive match
            if " "+city.lower()+" " in first_line.lower():
                found_cities.append(city.lower())

    return found_cities


def create_vector_store():
    print("Creating vector store...")
    
    # CREATE VECTOR STORE
    # load data
    with open("aixparag/data/data_and_metadata.json", 'r') as file:
        data = json.load(file)
    
    # loading or creating vector store
    my_vector_store = VectorStore(collection_name="my_app_docs",
                                  model_name='dbmdz/bert-base-italian-uncased')
                                # model_name='BAAI/bge-m3')

    # creating vs (actions as chunks)
    documents = []
    for doc_id, item in data.items():
        actions = [Document(page_content =  f"COMUNE DI: {item['place']}\n" + action['action_text'],
                            metadata = {"tassonomia": metadata['tassonomia'].lower(),
                                        "macro_ambito": metadata['macro-ambito'],
                                        "luogo": item['place'].lower(),
                                        "id": action['action_id']})
                            for action,metadata in zip(item['actions'], item['actions_metadata'])]
        documents.extend(actions)
    my_vector_store.populate_vector_store(documents)
    
    with open("aixparag/data/vector_store.pkl", "wb") as f:
        pickle.dump(my_vector_store, f)

    return my_vector_store

def load_vector_store():
    if "default" not in _GLOBAL_VECTOR_STORE:
        import pickle
        with open("aixparag/data/vector_store.pkl", "rb") as f:
            _GLOBAL_VECTOR_STORE["default"] = pickle.load(f)
    
    return _GLOBAL_VECTOR_STORE["default"]


def convert_conversation_format(dialogue_list):
    conversation = []
    for turn in dialogue_list:
        conversation.append(turn["turn_text"])
    return conversation

# # loading retriever
def rag_answer(documents_list, dialogue_list, query, options_number, hf_token, chatbot_is_first):
    my_vector_store = load_vector_store()
    my_retriever = Retriever(vector_store=my_vector_store, reranker_model_name=_GLOBAL_RERANKERS["reranker_hf_model"])

    luoghi = find_cities_in_first_lines(documents_list)

    tassonomie_dialogo = []
    ambiti_dialogo = []

    for city in luoghi:
        if city in _GLOBAL_TASSONOMIE and _GLOBAL_TASSONOMIE[city]:
            tassonomie_dialogo.extend(_GLOBAL_TASSONOMIE[city])
        if city in _GLOBAL_AMBITI and _GLOBAL_AMBITI[city]:
            ambiti_dialogo.extend(_GLOBAL_AMBITI[city])  
    
    tassonomie = list(set(tassonomie_dialogo))
    ambiti = list(set(ambiti_dialogo))

    vllm_model = VLLMModel()
    conversation = convert_conversation_format(dialogue_list)
    query = utils.expand_query(vllm_model, conversation)

    # planner_res = utils.planner(vllm_model, query, conversation)
    planner_res = "YES"
    if planner_res == "NO":
        retrieved_results = []
        return  retrieved_results   
    else:
        # response_dict contains metadata extracted from the last turn (query)
        
        router = utils.sql_planner(vllm_model, query)
        if router == "DB_QUERY":
            response_dict = utils.exctract_metadata(vllm_model, query, conversation, tassonomie, ambiti, luoghi)
            search_results = my_vector_store.db_select(filters=response_dict, limit=10)
            retrieved_results =[el.payload['page_content'] for el in search_results[0]]
            return retrieved_results

        else:
            response_dict = dict()
            response_dict['luogo'] =  luoghi
            search_results = my_retriever.retrieve(query, k=15, filters=response_dict)
            filtered_results = my_retriever.rerank(query, search_results, k=5)
            retrieved_results = [el['page_content'] for el in filtered_result]
            return retrieved_results



def rag_answer_highlight(documents_list, query, options_number, hf_token):

    my_vector_store = load_vector_store()
    

    luoghi = find_cities_in_first_lines(documents_list)

    tassonomie_dialogo = []
    ambiti_dialogo = []

    for city in luoghi:
        if city in _GLOBAL_TASSONOMIE and _GLOBAL_TASSONOMIE[city]:
            tassonomie_dialogo.extend(_GLOBAL_TASSONOMIE[city])
        if city in _GLOBAL_AMBITI and _GLOBAL_AMBITI[city]:
            ambiti_dialogo.extend(_GLOBAL_AMBITI[city])  
    
    my_retriever = Retriever(vector_store=my_vector_store, reranker_model_name=_GLOBAL_RERANKERS["reranker_hf_model"])    
    response_dict = dict()
    response_dict['luogo'] =  luoghi
    
    search_results = my_retriever.retrieve(query, k=20, filters=response_dict)
    
    filtered_results,results_scores = my_retriever.rerank_scores(query, search_results, k=5)
    

    retrieved_results = [el['page_content'] for el in filtered_results]
    return retrieved_results