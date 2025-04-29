import requests
from jprint import jprint
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', default="0.0.0.0")
parser.add_argument('--port', default=8018, type=int)
args = parser.parse_args()

url = "http://"+args.host+":"+str(args.port)

documents = [
    "Questo è il primo documento",
    "Questo è il secondo documento",
    "Questo è il terzo documento"
]

print("  Testing ground retrieval", end="\r")
try:
    request_ground = {
        "documents_list": documents,    
        "query": "dammi dei documenti",
        "options_number": 3
    }

    endpoint_url = url+"/turn_ground/"
    response = requests.post(endpoint_url, json.dumps(request_ground))

    # print(json.dumps(response.json(), indent=4, ensure_ascii=False))

    grounds = []
    for ground in response.json():
        grounds.append(ground["text"])

    request_new_turn = {
        "documents_list": documents,
        "dialogue_list":[
            {
            "speaker":"operatore",
            "turn_text":"qanti documenti ci sono?"
            }
        ],
        "user": "operatore",
        "tone": "informal",
        "chatbot_is_first": False
    }
    print("  Testing ground retrieval...............\033[42mPASSED\033[0m")
except:
    print("  Testing ground retrieval...............\033[37m\033[41mFAILED\033[0m")


print("  Testing next turn generation", end="\r")
try:
    endpoint_url = url+"/turn_generation/"
    response = requests.post(endpoint_url, json.dumps(request_new_turn))
    # print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    # print(response.json()["turn_text"])
    print("  Testing next turn generation...........\033[42mPASSED\033[0m")
except:
    print("  Testing next turn generation...........\033[37m\033[41mFAILED\033[0m")



print("  Testing next turn stream", end="\r")
message_received = "" 
try:
    endpoint_url = url+"/turn_stream/"
    response = requests.post(endpoint_url, json=request_new_turn, stream=True)
    for line in response.iter_lines():
        if line:
            message_received = message_received+line.decode('utf-8')
            # print(line.decode('utf-8'))

    print("  Testing next turn stream...............\033[42mPASSED\033[0m")
except:
    print("  Testing next turn stream...............\033[37m\033[41mFAILED\033[0m")
