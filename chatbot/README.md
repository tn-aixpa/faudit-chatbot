# Chatbot API

Before creating the docker change the kubeai\_host and port in the Dockerfile. Then run:

```docker build -t aixpa-chatbot-api .``` 

Default port for the server is 8018, can be changed in the Dockerfile

```docker run -p 8018:8018  aixpa-chatbot-api```

To ensure everyting is working run 

`python test.py --host 0.0.0.0 --port 8018`

The espected result is:


> Testing ground retrieval..................**PASSED** <br>
> Testing next turn generation...........**PASSED** <br>
> Testing next turn stream.................**PASSED**


# Endpoints

There are 3 endpoints available. Two for generation and one to identify in the documents the relevant parts to for the dialogue.


## ```/turn_generation```
Generates the nex turn of the dialogue.
POST request taking in input a json with: 

`user`: who is the user interacting with the chatbot, either  ***cittadino*** or ***operatore***

`tone`: the tone of the conversation, either ***formale*** or ***informale***

`documents_list`: list of the document(s) on witch the chat is based (containtng the full texts).

`dialogue_list`: list of jsons with the previous turns. Each turn contains the speaker and the text of the turn.

`chatbot_is_first`: Boolean. Indicate if the first turn is from the chatbot or from the user.

Below an example of the json


```json
{
    "user": "cittadino|operatore",
    "tone": "formale|informale",
    "documents_list": [
      "text of document 1",
      "text of document 2",
      "text of document 3"
    ],
    "dialogue_list":[
        {
        "speaker":"operatore",
        "turn_text":"turn text"
        },
        {
        "speaker":"assistant",
        "turn_text":"turn text"
        },
        {
        "speaker":"operatore",
        "turn_text":"turn text"
        }
    ],
    "chatbot_is_first": false
}
```

Returns a json with the message

```json
{
  "turn_text": "text of the next turn"
}
```

## ```/turn_stream```

Generates the nex turn of the dialogue.
POST request, the input is the same of  ```/turn_generation``` 

The output is a data stream.


## ```/turn_ground```

Given a text (e.g. the question of the user or the answer of the chatbot), retrieve from the documents the N most relevant chunks of text.

POST request taking in input a json with: 

`documents_list`: list of the document(s) from witch retrieve the chunks.

`query`: the turn/answer to be grounded

`options_number`: number of text chunks to retrieve

```json
{
    "documents_list": [
      "text of document 1",
      "text of document 2",
      "text of document 3"
    ],
  "query": "last turn/answer to be grounded",
  "options_number": int

}
```

Returns a json with a list of text, each associated with a  `file_index` (index in the `documents_list` from the input) annd the characters offsets.

```json
[
    {
        "text": "retrieved text 1",
        "file_index": 1,
        "offset_start": 0,
        "offset_end": 50
    },
    {
        "text": "retrieved text 2",
        "file_index": 0,
        "offset_start": 30,
        "offset_end": 80
    },
    {
        "text": "retrieved text 3",
        "file_index": 2,
        "offset_start": 240,
        "offset_end": 290
    }
]
```