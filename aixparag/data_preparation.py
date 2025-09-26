import pandas as pd
import json
import collections
import re
import os
from .global_cache import _GLOBAL_AMBITI, _GLOBAL_TASSONOMIE

# def chunking(data_dict, metadata=False):
#     chunked_data = collections.defaultdict(lambda : 'Key Not Found')
#     for doc_id,text in data_dict.items():
        
#         document_title = text.split('===\n')[0].strip('=== ')
#         # document_text = '===\n'.join(text.split('===\n')[1:])
#         document_text = text.split('===\n')[-1].strip('=== ')

#         actions = [{'action_id' : f'{doc_id}_{i}', 
#                     'action_text' : action.strip()} 
#                    for i,action in enumerate(document_text.split('\n-----\n')) 
#                    if action!='\n']
#         if metadata:
#             # title's metadata (place and year)
#             regex_title = r"PIANO FAMIGLIA (?:COMUNE\s*)?(?:DI\s)?(?P<place>.+?)\s*ANNO\s*(?P<year>\d{4})"
#             match = re.search(regex_title, document_title)

#             if match:
#                 place = match.group("place")
#                 year = match.group("year")

                
#             else:
#                 print(f"No match found for: '{document_title}' in doc {doc_id}\n")
#             # actions metadata
#             regex_actions = re.compile(
#                 r"TITOLO:\s*(?P<TITOLO>.*?)\n"                 
#                 r"TASSONOMIA:\s*(?P<TASSONOMIA>.*?)\n"         
#                 r"MACRO-AMBITO:\s*(?P<MACRO_AMBITO>.*?)\n"     
#                 r"OBIETTIVO:\s*(?P<OBIETTIVO>.*?)\n"           
#                 r"DESCRIZIONE:\s*(?P<DESCRIZIONE>.*)",         
#                 re.DOTALL  # Allows '.' to match newline characters, essential for multi-line DESCRIZIONE
#             )
#             actions_metadata = []

#             for action in actions:
#                 match = regex_actions.search(action['action_text']) 
#                 if match:

                    
#                     # Extract values using the named groups
#                     extracted_data = {
#                         "action_id" : action['action_id'],
#                         "titolo": match.group("TITOLO").strip(),
#                         "tassonomia": match.group("TASSONOMIA").strip(),
#                         "macro-ambito": match.group("MACRO_AMBITO").strip(), 
#                         "obiettivo": match.group("OBIETTIVO").strip(),
#                         "descrizione": match.group("DESCRIZIONE").strip()
#                     }
#                     actions_metadata.append(extracted_data)
#                     # print("-----------------------------------------------------------------")
#                     # print("-----------------------------------------------------------------")
#                     # print("-----------------------------------------------------------------")
#                     # print("-----------------------------------------------------------------")
#                     # print("-----------------------------------------------------------------")
#                     # print(extracted_data)
#                     # print("-----------------------------------------------------------------")
#                     # print("-----------------------------------------------------------------")
#                     # print("-----------------------------------------------------------------")
#                     # print("-----------------------------------------------------------------")
#                     # print("-----------------------------------------------------------------")
#                 else:
#                     print("No full match found for all specified fields. Ensure all fields are present and in order.")
#                     print(f"{doc_id}   -   {action}")
#             chunked_data[doc_id] = {'document_title': document_title,
#                                     'document_text' : document_text,
#                                     'place': place,
#                                     'year' : year, 
#                                     'actions' : actions,
#                                     'actions_metadata': actions_metadata}
#         else:
#             chunked_data[doc_id] = {'document_title': document_title,
#                                     'document_text' : document_text,
#                                     'actions' : actions}
    
#     return chunked_data



def chunking(data_dict, metadata=False):
    """
    Chunks text data from a dictionary, splitting it into documents and actions,
    and optionally extracts structured metadata from both the title and the actions.

    Args:
        data_dict (dict): A dictionary where keys are document IDs and values are
                          the raw text content of the documents.
        metadata (bool): If True, the function will attempt to extract detailed
                         metadata. Defaults to False.

    Returns:
        collections.defaultdict: A dictionary where keys are document IDs and
                                 values are dictionaries containing the chunked
                                 and processed data for each document.
    """
    chunked_data = collections.defaultdict(lambda: 'Key Not Found')
    for doc_id, text in data_dict.items():
        
        # Safely split the text into title and main content
        parts = text.split('===\n')
        document_title = parts[0].strip('=== ') if parts else ''
        # document_text = parts[1].strip() if len(parts) > 1 else ''
        document_text = parts[-1].strip('=== ')  if parts else ''

        # Create a list of actions, filtering out any empty entries
        actions = [
            {'action_id': f'{doc_id}_{i}', 'action_text': action.strip()}
            for i, action in enumerate(document_text.split('\n-----\n'))
            if action.strip()
        ]

        if metadata:
            # --- Title Metadata Extraction ---
            place = ''
            year = ''
            regex_title = r"PIANO FAMIGLIA (?:COMUNE\s*)?(?:DI\s)?(?P<place>.+?)\s*ANNO\s*(?P<year>\d{4})"
            match_title = re.search(regex_title, document_title)

            if match_title:
                place = match_title.group("place").strip()
                year = match_title.group("year").strip()
            else:
                print(f"Warning: No title metadata match found for: '{document_title}' in doc {doc_id}\n")
            
            # --- Action Metadata Extraction (with optional fields) ---

            # Define a regex for each field we want to extract.
            # This allows each field to be optional and independent of the others.
            field_regexes = {
                "titolo": r"TITOLO:\s*(.*?)(?:\n|$)",
                "tassonomia": r"TASSONOMIA:\s*(.*?)(?:\n|$)",
                "macro-ambito": r"MACRO-AMBITO:\s*(.*?)(?:\n|$)",
                "obiettivo": r"OBIETTIVO:\s*(.*?)(?:\n|$)",
                "descrizione": r"DESCRIZIONE:\s*(.*)"  # This one can be multi-line
            }
            
            actions_metadata = []
            for action in actions:
                # Initialize a dictionary for the current action's metadata
                extracted_data = {"action_id": action['action_id']}
                
                # Loop through each regex and try to find a match in the action text
                for field_name, pattern in field_regexes.items():
                    # Use re.DOTALL flag only for the 'descrizione' field to allow multi-line matching
                    flags = re.DOTALL if field_name == "descrizione" else 0
                    match = re.search(pattern, action['action_text'], flags)
                    
                    # If a match is found, store the cleaned value. Otherwise, store an empty string.
                    if match:
                        extracted_data[field_name] = match.group(1).strip()
                    else:
                        extracted_data[field_name] = "" # Field is not present, so we leave it empty
                        
                actions_metadata.append(extracted_data)

            chunked_data[doc_id] = {
                'document_title': document_title,
                'document_text': document_text,
                'place': place,
                'year': year,
                'actions': actions,
                'actions_metadata': actions_metadata
            }
        else:
            chunked_data[doc_id] = {
                'document_title': document_title,
                'document_text': document_text,
                'actions': actions
            }
    
    return chunked_data

# annotated_data = pd.read_json("data/rag_data.json")
# dialog_documents = annotated_data.documents.to_list()
# dialog_documents = [el for docs in dialog_documents for el in docs if type(docs)==list]
# dialog_documents = list(set(dialog_documents)) # removing duplicates

def extract_metadata(doclist):
    print("Extracting metadata from documents...")
    # getting documents
    data = collections.defaultdict(lambda : 'Key Not Found')
    # for el in dialog_documents:
    #     with open(f"data/files/{el}", "r", encoding='utf8') as file:
    #         data[el] = file.read()

    for i, el in enumerate(doclist):
        data[str(i)] = el

    # chunking
    chunked_data = chunking(data, metadata=True)
    # saving
    save = True
    # print(chunked_data)
    if save:
        try:
            with open("aixparag/data/data_and_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(chunked_data, f, indent=4, ensure_ascii=False)
            # print("-----------------------------------------------------------------")
            # print("-----------------------------------------------------------------")
            print("-----------------------------------------------------------------")
            print(f"Data successfully saved to 'data/data_and_metadata.json'")
            print("-----------------------------------------------------------------")
            # print("-----------------------------------------------------------------")
            # print("-----------------------------------------------------------------")
        except IOError as e:
            # print("-----------------------------------------------------------------")
            # print("-----------------------------------------------------------------")
            print("-----------------------------------------------------------------")
            print(f"Error saving data to file: {e}")
            print("-----------------------------------------------------------------")
            # print("-----------------------------------------------------------------")
            # print("-----------------------------------------------------------------")

    ###########################
    ### DATASET DESCRIPTION ###
    ###########################

    print("=== DESCRIPTION ===")
    all_tassonomie = set()
    all_ambiti = set()
    all_cities = set()
    total_chunks = 0

    # Iterate through each document in your main dictionary
    for doc_id, doc_info in chunked_data.items():
        # Count the total number of 'actions_metadata' dictionaries, which are your "chunks"
        total_chunks += len(doc_info.get("actions_metadata", []))
        city = doc_info.get("place", [])
        all_cities.add(city)
        # print(_GLOBAL_TASSONOMIE)
        if city not in _GLOBAL_TASSONOMIE:
            _GLOBAL_TASSONOMIE[city.lower()] = []
        if city not in _GLOBAL_AMBITI:
            _GLOBAL_AMBITI[city.lower()] = []
        # Iterate through each 'action' (chunk) metadata to collect TASSONOMIE and MACRO-AMBITO
        for action_metadata in doc_info.get("actions_metadata", []):

            tassonomia = action_metadata.get("tassonomia")
            if tassonomia:
                all_tassonomie.add(tassonomia.lower())
                
                
                _GLOBAL_TASSONOMIE[city.lower()].append(tassonomia.lower())

          
            ambito = action_metadata.get("macro-ambito")  
            if ambito:
                all_ambiti.add(ambito.lower())
                
                _GLOBAL_AMBITI[city.lower()].append(ambito.lower())
  

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(_GLOBAL_AMBITI)
    # print("--------------------------------------------------")
    # print(_GLOBAL_TASSONOMIE)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
    # Calculate metrics
    num_documents = len(chunked_data.keys())
    num_chunks = total_chunks
    chunks_per_doc = num_chunks / num_documents if num_documents > 0 else 0

    with open("aixparag/data/tassonomie.txt", "w") as file:
        for t in sorted(list(all_tassonomie)):
            file.write(f"{t}\n")
        

    with open("aixparag/data/ambiti.txt", "w") as file:
        for a in sorted(list(all_ambiti)):
            file.write(f"{a}\n")

    print(f"\n\n\nTUTTE LE CITTA: {all_cities}\n\n\n")

    with open("aixparag/data/cities.txt", "w") as file:
        for a in sorted(list(all_cities)):
            file.write(f"{a}\n")

    # print(_GLOBAL_TASSONOMIE)

    # print(_GLOBAL_AMBITI)

    print(f"> # documents: {num_documents}")
    print(f"> # chunks: {num_chunks}")
    print(f"> # chunks per doc: {chunks_per_doc:.2f}") # Format to 2 decimal places
    print(f"> # TASSONOMIE: {len(all_tassonomie)}")
    print(f"> # AMBITI: {len(all_ambiti)}")

    # Optional: Print the unique lists themselves if you want to inspect them
    print("\n--- Unique TASSONOMIE ---")
    for t in sorted(list(all_tassonomie)):
        print(f"- {t}")

    print("\n--- Unique MACRO-AMBITO ---")
    for a in sorted(list(all_ambiti)):
        print(f"- {a}")
