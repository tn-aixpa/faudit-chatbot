from LanguageModel import GroqModel, HuggingFaceModel
from VectorStoreQdrant import VectorStore
from Retriever import Retriever
from langchain_core.documents import Document
import prompts
import pandas as pd
import json
import utils
from huggingface_hub import login
from guidance.models import Transformers
from guidance.chat import ChatTemplate

# CREATE VECTOR STORE
# load data
with open("data/data_and_metadata.json", 'r') as file:
    data = json.load(file)

# loading or creating vector store
my_vector_store = VectorStore(collection_name="my_app_docs",
                              model_name='BAAI/bge-m3')

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

# loading retriever
my_retriever = Retriever(vector_store=my_vector_store, reranker_model_name='BAAI/bge-reranker-v2-m3')

with open("data/tassonomie.txt", "r") as file:
    tassonomie = file.read().splitlines()

with open("data/ambiti.txt", "r") as file:
    ambiti = file.read().splitlines()

# groq_model = GroqModel()
hf_token = input("Enter hf token: ")
login(hf_token)
hf_model = HuggingFaceModel()
print(f"Using model: {hf_model.model_name}")

llama31_template = hf_model.tokenizer.chat_template
class Llama31ChatTemplate(ChatTemplate):
    template_str = llama31_template

    def get_role_start(self, role_name):
        if role_name == "system":
            return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        elif role_name == "user":
            return "<|start_header_id|>user<|end_header_id|>\n\n"
        elif role_name == "assistant":
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            print("Error!!")

    def get_role_end(self, role_name=None):
        return "<|eot_id|>"

gui_lm = Transformers(model=hf_model.model,
                      tokenizer=hf_model.tokenizer,
                      chat_template=Llama31ChatTemplate)

query = ""
conversation = []

while query.lower() != "exit":
    try:
        if len(conversation) % 2 == 0:
            query = input("Enter text: ")
            if query.lower() == 'exit':
                print("Exiting the program. Goodbye!")
                break
            else:
                conversation.append(query)

        # If the input is not 'exit', you can process it here
        print(f"\n\n>>> You entered: {query}\n\n")
        planner_res = utils.planner(hf_model, query, conversation)
        print("RESPONSE:", planner_res)
        if planner_res == "NO":
            print("-- No RAG module needed --")
            reply = utils.no_rag_reply_hf(hf_model, query, conversation)
            print(f"\n\n>>> ChatBot: {reply}\n\n")
            conversation.append(reply)
        else:
            print("-- RAG module activated --")
            response_dict = utils.exctract_metadata(gui_lm, query, conversation, tassonomie, ambiti)
            router = utils.sql_planner(hf_model, query)
            if router == "DB_QUERY":
                print("--> DB_QUERY")
                search_results = my_vector_store.db_select(filters=response_dict, limit=5)
                context ="\n\n".join([el.payload['page_content'] for el in search_results[0]])
                reply = utils.rag_reply_hf(hf_model, query, conversation, context)
                conversation.append(reply)
                print(f"\n\n>>> ChatBot: {reply}\n\n")
            else:
                print("--> SEMANTIC_SEARCH")
                search_results = my_retriever.retrieve(query, k=10, filters=response_dict)
                filtered_results = my_retriever.rerank(query, search_results, k=3)
                context = "\n\n".join([el['page_content'] for el in filtered_results])
                print("\n>>> EXTRATED ACTIONS <<<\n")
                for el in filtered_results:
                    print(el['page_content'])
                print("\n=======================\n")
                
                reply = utils.rag_reply_hf(hf_model, query, conversation, context)
                print(f"\n\n>>> ChatBot: {reply}\n\n")
                conversation.append(reply)
    except EOFError:
        print("\nEnd of input detected. Exiting.")
        break
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")
        break

print("Loop finished.")