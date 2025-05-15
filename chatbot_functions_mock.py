from tools import chunker, retrieval
from fastapi.responses import StreamingResponse


def stream_answer(documents_list, dialogue_list, user, tone, chatbot_is_first):

    print("mock stream_answer")
    stream = ["lorem ipsum", "dolor sit amet", "consectetur adipiscing elit"]
    
    async def event_generator():
        for chunk in stream:
            content = chunk
            if content:
                yield content

    return StreamingResponse(event_generator(), media_type="application/json")


def generate_answer(documents_list, dialogue_list, user, tone, chatbot_is_first):
    from start_api import start_api_openai_base_url, start_api_openai_key, start_api_openai_model

    print("mock generate_answer")
    
    next_turn = {
            "turn_text": "TITOLO: Consulta della Famiglia\nTASSONOMIA: Istituzione/coinvolgimento della consulta per la famiglia\nMACRO-AMBITO: Governance e azioni di rete\nOBIETTIVO: (nessuno specificato)\nDESCRIZIONE: Nel corso del 2023 si continuerа la riflessione sul ruolo della Consulta e il rinnovo della stessa, dando particolare attenzione alla scelta dei componenti che ne debbono fare parte e al ruolo specifico che la stessa deve avere. In particolare verranno identificati i soggetti che comporranno tale organo e verranno definiti gli obiettivi che la stessa potrа raggiungere.\nL'obiettivo principale è quello di coinvolgere e sensibilizzare, trasmettendo ai cittadini il senso delle\niniziative proposte, pur nella consapevolezza di non riuscire a coprire la totalitа delle singole esigenze.\nLa Consulta dovrа essere in grado di raccogliere le proposte che via via emergeranno sia da parte degli amministratori comunali che dai cittadini, al fine di affinare negli anni il piano di azione in materia di politiche familiari.",                
        }        
    
    return next_turn




def get_ground(documents_list, query, options_number):

    print("mock get_ground")

    grounds_list = []
    for g_doc in range(len(documents_list)):
        d = documents_list[g_doc]
        ground_info = dict()
        
        start = d.find("DESCRIZIONE:") 
        if start < 0: start = 0
        else: start += len("DESCRIZIONE:") 
        end = len(d) - 1
        g_text =   d[start:end]
        
        ground_info["text"] = g_text
        ground_info["file_index"] = g_doc
        ground_info["offset_start"] = start
        ground_info["offset_end"] = end
        grounds_list.append(ground_info)
        
    return grounds_list


