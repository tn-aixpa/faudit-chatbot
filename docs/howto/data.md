# How to prepare vector store dataset


## Prepare the data

1. Initialize the AIxPA project.

```python
import digitalhub as dh
project = dh.get_or_create_project("faudit-chatbot")
```

2. Create the data artifact from the documents and upload it to the platform. It is expected to have the data as a collection of ``txt`` files each containing
a list of plans containing reference to title, description, objective, taxonomy element, and macro area. For example:

```
=== PIANO FAMIGLIA COMUNE DI ROMA ANNO 2025 ===

TITOLO: Titolo del'azione
DESCRIZIONE: Test di descizione dell'azione
OBIETTIVO: Sensibilizzare, informare e formare il mondo scolastico.
TASSONOMIA: Attivitа  di educazione ambientale (laboratori, giornate ecologiche, giornata del riuso, raccolta differenziata)
MACRO-AMBITO: Comunitа educante

-----

TITOLO: Titolo del'azione
DESCRIZIONE: Test di descizione dell'azione
OBIETTIVO: Sensibilizzare, informare e formare il mondo scolastico.
TASSONOMIA: Attivitа  di educazione ambientale (laboratori, giornate ecologiche, giornata del riuso, raccolta differenziata)
MACRO-AMBITO: Comunitа educante

```


```python
art = project.log_artifact("rag_documents", kind="artifact", source="./RAG_documents")
```

1. Define the processing function with the container runtime.

```python
chatbot_function = project.new_function(name="chatbot", kind="container", image="ghcr.io/tn-aixpa/faudit-chatbot:0.2.8")
```

3. Run the processing job

```python
processing_run = chatbot_function.run(action="job", args=["--data_artifact=rag_documents", "--prepare_data"])
```

This will register the ``rag_storage`` artifact to the platform. The artifact represent the serialized vector storage and may be used for chatbot service
