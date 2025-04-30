from typing import List, Dict
from openai import OpenAI
from rank_bm25 import BM25Okapi
# from FlagEmbedding import BGEM3FlagModel
import torch
from dataclasses import dataclass
import numpy as np
from tools.chunker import TextNode


TOP_K = 5
# url for using KUBEAI
BASE_URL = "https://kubeai.digitalhub-dev.smartcommunitylab.it/openai/v1"

class Retriever_llamaindex_bm25:
    """
    BM25 retriever to be used if knowledge base consist of Llamaindex chunks
    """
    def __init__(self,
                 knowledge_base,
                 name='BM25',
                 top_k=3):
        
        from llama_index.retrievers.bm25 import BM25Retriever

        self.name = name
        self.top_k = top_k
        self.knowledge_base = knowledge_base.nodes
        self.retriever = self.set_retriever()


    def set_retriever(self):
        if self.name == 'BM25':
            retriever = BM25Retriever.from_defaults(nodes=self.knowledge_base,
                                                    similarity_top_k=self.top_k)
        else:
            print("RetrieverModule Error: Select a valid retriever.")
            retriever = None
        return retriever

    def retrieve(self, query: str):
        """
        TODO
        """
        return self.retriever.retrieve(query.strip())

    def set_top_k(self, top_k):
        """
        TODO
        """
        self.top_k = top_k
        self.retriever = self.set_retriever()


class Retriever_bm25:
    """
    BM25 retriever that can be used with any knowledge base 
    (both llama index nodes and generic text nodes as defined in chunker)
    """
    def __init__(self,
                 knowledge_base,
                 name='BM25',
                 top_k=3):

        self.name = name
        self.top_k = top_k
        self.knowledge_base = list(knowledge_base.nodes)
        self.set_retriever()

    def set_retriever(self):
        if self.knowledge_base is None or len(self.knowledge_base) == 0:
            print("Error: Knowledge base is empty!")
            self.retriever = None
            return

        if self.name == 'BM25':
            self.corpus = [doc.text for doc in self.knowledge_base]  # Store corpus
            tokenized_docs = [doc.split(" ") for doc in self.corpus]

            if len(self.corpus) == 0:
                print("Error: Corpus is empty! BM25 retriever cannot be initialized.")
                self.retriever = None
                return
            self.retriever = BM25Okapi(tokenized_docs)
            print(f"BM25 retriever initialized with {len(self.corpus)} documents.")
            
        else:
            print("RetrieverModule Error: Select a valid retriever.")
            self.retriever = None

    def retrieve(self, query: str):
        """
        Retrieve top n most similar chunks to the query
        """      
        if self.retriever is None:
            raise ValueError("Retriever is not initialized. Check if knowledge base is empty.")

        text_metadata_dict = {node.dict()['text']: node.dict()['metadata'] for node in self.knowledge_base}
        tokenized_query = query.split(" ")
        # Ensure we're using the same corpus that was used to initialize BM25
        retrieved_docs = self.retriever.get_top_n(tokenized_query, self.corpus, n=self.top_k)

        return [TextNode(text=retrieved_doc.strip(), metadata=text_metadata_dict[retrieved_doc]) for retrieved_doc in retrieved_docs]

class Retriever_BGE_v2_m3:
    """
    bge-reranker-v2-m3
    """
    def __init__(self,
                 knowledge_base,
                 model_path="BAAI/bge-reranker-v2-m3",
                 name='BGE',
                 top_k=TOP_K):
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.name = name
        self.top_k = top_k
        self.knowledge_base = knowledge_base.nodes
        self.model_path = model_path
        self.retriever = self.set_retriever()
        self.tokenizer = self.set_tokenizer()

    def set_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return tokenizer

    def set_retriever(self):
        if self.name == 'BGE':
            retriever = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            retriever.eval()
        else:
            print("RetrieverModule Error: Select a valid retriever.")
            retriever = None
        return retriever

    def retrieve(self, query: str):
        """
        TODO
        """
        text_metadata_dict = {node.dict()['text']:node.dict()['metadata'] for node in self.knowledge_base}
        corpus = list(text_metadata_dict.keys())
        pairs = [[query, k] for k in text_metadata_dict.keys()]
        # scores = {k:self.retriever.compute_score([query, k])[0] for k in text_metadata_dict.keys()}

        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.retriever(**inputs, return_dict=True).logits.view(-1, ).float()

        scores_dict = dict(zip(text_metadata_dict.keys(), scores.tolist()))
        sorted_values = sorted(scores_dict.values(), reverse=True)
        retrieved_docs = [next((k for k, v in scores_dict.items() if v == value), None) for value in sorted_values[:self.top_k]]
        return [TextNode(text = retrieved_doc.strip(), metadata = text_metadata_dict[retrieved_doc]) for retrieved_doc in retrieved_docs]

    def set_top_k(self, top_k):
        """
        TODO
        """
        self.top_k = top_k

class Embedder_BGE_m3:
    """
    BAAI/bge-m3
    """
    def __init__(self,
                 knowledge_base,
                 url=BASE_URL,
                 name='bge-m3',
                 top_k=TOP_K):
        
        

        self.name = name
        self.top_k = top_k
        self.knowledge_base = knowledge_base.nodes
        self.url = url
        self.model = self.set_model()

    def set_model(self):
        if self.name == 'bge-m3':
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-m3")
        else:
            print("RetrieverModule Error: Select a valid retriever.")
            model = None
        return model

    def retrieve(self, query: str):
        """
        TODO
        """
        text_metadata_dict = {node.dict()['text']:node.dict()['metadata'] for node in self.knowledge_base}
        corpus = list(text_metadata_dict.keys())
        pairs = [[query, k] for k in text_metadata_dict.keys()]
        # scores = {k:self.retriever.compute_score([query, k])[0] for k in text_metadata_dict.keys()}

        scores = []
        for sent1, sent2 in pairs:
            emb1 = self.model.encode(sent1, convert_to_tensor=True, normalize_embeddings=False)
            emb2 = self.model.encode(sent2, convert_to_tensor=True, normalize_embeddings=False)
            dot_sim = torch.dot(emb1, emb2).item()
            scores.append(dot_sim)

        scores_dict = dict(zip(text_metadata_dict.keys(), scores))
        sorted_values = sorted(scores_dict.values(), reverse=True)
        retrieved_docs = [next((k for k, v in scores_dict.items() if v == value), None) for value in sorted_values[:self.top_k]]
        return [TextNode(text = retrieved_doc.strip(), metadata = text_metadata_dict[retrieved_doc]) for retrieved_doc in retrieved_docs]

    def set_top_k(self, top_k):
        """
        TODO
        """
        self.top_k = top_k

class Embedder_BGE_m3_kubeai:

    """
    BAAI/bge-m3
    """
    def __init__(self,
                 knowledge_base,
                 url="",
                 name='bge-m3',
                 top_k=TOP_K):

        self.name = name
        self.top_k = top_k
        self.knowledge_base = knowledge_base.nodes
        self.url = url
        self.client = self.set_model()

    def set_model(self):
        from start_api import start_api_kubeai_host
        if self.name == 'bge-m3':
            client = OpenAI(api_key="ignored", base_url=start_api_kubeai_host) 
        else:
            print("RetrieverModule Error: Select a valid retriever.")
            client = None
        return client

    def retrieve(self, query: str):
        """
        TODO
        """
        text_metadata_dict = {node.dict()['text']:node.dict()['metadata'] for node in self.knowledge_base}
        corpus = list(text_metadata_dict.keys())
        pairs = [[query, k] for k in text_metadata_dict.keys()]
        # scores = {k:self.retriever.compute_score([query, k])[0] for k in text_metadata_dict.keys()}

        embeddings = np.array([
        [
            self.client.embeddings.create(input=chunk, model="bge-m3").data[0].embedding,
            self.client.embeddings.create(input=query, model="bge-m3").data[0].embedding
        ]
        for chunk, query in pairs
        ])

        # Separate chunk and query embeddings
        chunk_embeddings = np.stack(embeddings[:, 0])  # Shape: (N, D)
        query_embeddings = np.stack(embeddings[:, 1])  # Shape: (N, D)
        scores = np.einsum('ij,ij->i', chunk_embeddings, query_embeddings)  # Shape: (N,)

        scores_dict = dict(zip(text_metadata_dict.keys(), scores.tolist()))
        sorted_values = sorted(scores_dict.values(), reverse=True)
        retrieved_docs = [next((k for k, v in scores_dict.items() if v == value), None) for value in sorted_values[:self.top_k]]
        return [TextNode(text = retrieved_doc.strip(), metadata = text_metadata_dict[retrieved_doc]) for retrieved_doc in retrieved_docs]

    def set_top_k(self, top_k):
        """
        TODO
        """
        self.top_k = top_k