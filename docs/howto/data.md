# How to prepare vector store dataset


## Prepare the data

1. Initialize the AIxPA project.

```python
import digitalhub as dh
project = dh.get_or_create_project("faudit-classifier")
```

2. Create the data artifact from the documents and upload it to the platform.

```python
art = project.log_artifact("rag_documents", kind="artifact", source="./RAG_documents")
```

3. Define the processing function with the container runtime.

```python
chatbot_function = project.new_function(name="chatbot", kind="container", image="ghcr.io/tn-aixpa/faudit-chatbot:0.2.8")
```

3. Run the processing job

```python
processing_run = chatbot_function.run(action="job", args=["--data_artifact=rag_documents", "--prepare_data"])
```

This will register the ``rag_storage`` artifact to the platform. The artifact represent the serialized vector storage and may be used for chatbot service
