"""
VectorStore Class for Qdrant
"""

from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny, SparseVectorParams
from langchain_core.documents import Document
from typing import List, Dict, Optional, Any
# from .global_cache import _GLOBAL_EMBEDDINGS
import aixparag.global_cache as global_cache

class VectorStore:
    """
    A class to manage a Qdrant vector store, providing methods for
    initialization, document addition, and similarity search.
    """

    def __init__(self, collection_name: str = "demo_collection", 
                vector_size: int = None,
                distance: Distance = Distance.COSINE,
                model_name="dbmdz/bert-base-italian-uncased"):
                # model_name="sentence-transformers/all-mpnet-base-v2"):
        
        """
        Initializes the VectorStore with an in-memory Qdrant client and creates
        a collection.
        """
        print(f"Initializing VectorStore with collection: '{collection_name}'...")
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.client = QdrantClient(":memory:")
        # self.client = QdrantClient(path="qdrant_db") 
        self.sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
        
        
        

        

        if vector_size == None:
            dummy_text = "This is a test sentence."
            vector_size = len(self.embeddings.embed_query(dummy_text))

        # Create the collection if it doesn't already exist
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"dense":VectorParams(size=vector_size, distance=distance)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=True))}
            )
            print(f"Collection '{self.collection_name}' created/recreated successfully.")
        except Exception as e:
            print(f"Error creating/recreating collection '{self.collection_name}': {e}")

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embedding,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse"
        )
        print("VectorStore initialized.")

    def populate_vector_store(self, documents: List[Document], ids=None):
        """
        Populates the vector store with a list of documents.
        """
    
        global_cache._GLOBAL_EMBEDDINGS["embeddings"] = self.embeddings
        print("Embedding model loaded and stored in global cache.")
        print(type(global_cache._GLOBAL_EMBEDDINGS["embeddings"]))
        
        if not documents:
            print("No documents provided to populate.")
            return

        print(f"Adding {len(documents)} documents to the vector store...")
        if ids == None:
            ids = [str(uuid4()) for _ in range(len(documents))]
        try:
            self.vector_store.add_documents(documents=documents, ids=ids)
            print(f"Successfully added {len(documents)} documents.")
        except Exception as e:
            print(f"Error adding documents: {e}")

    def add_document(self, document: Document, id=None):
        """
        Adds a single document to the vector store.
        """
        print(f"Adding single document: '{document.page_content[:50]}...'")
        try:
            if id == None:
                id = str(uuid4())
            self.vector_store.add_documents(documents=[document], ids=[id])
            print("Document added successfully.")
        except Exception as e:
            print(f"Error adding document: {e}")

    def search(self, query: str, k: int = 2, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Performs a similarity search in the vector store.
        """
        print(f"\nSearching for: '{query}' (k={k}) with filters: {filters}")
        qdrant_filter = None
        if filters:
            should_conditions = []
            must_conditions = []
            for key, value in filters.items():
                qdrant_key = f"metadata.{key}"
                if value == None or len(value) == 0 or value[0] == 'None':
                    continue
                elif len(value) > 1:
                    match = MatchAny(any=value)
                else:
                    match = MatchValue(value=value[0])
                if key == 'luogo':
                    must_conditions.append(
                        FieldCondition(
                            key=qdrant_key,
                            match=match
                        )
                    )
                else:
                    should_conditions.append(
                        FieldCondition(
                            key=qdrant_key,
                            match=match
                        )
                    )
            # qdrant_filter = Filter(should=should_conditions, must=must_conditions)
            qdrant_filter = Filter(must=must_conditions)


        try:
            results = self.vector_store.similarity_search(  
                query=query,
                k=k,
                filter=qdrant_filter
            )
            print(f"Found {len(results)} results.")
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    # def search(self, query: str, k: int = 2, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
    #     """
    #     Performs a similarity search in the vector store.
    #     """
    #     print(f"\nSearching for: '{query}' (k={k}) with filters: {filters}")
    #     qdrant_filter = None
    #     if filters:
    #         should_conditions = []
    #         for key, value in filters.items():
    #             qdrant_key = f"metadata.{key}"
    #             if value == None or len(value) == 0 or value[0] == 'None':
    #                 continue
    #             elif len(value) > 1:
    #                 match = MatchAny(any=value)
    #             else:
    #                 match = MatchValue(value=value[0])
    #             should_conditions.append(
    #                 FieldCondition(
    #                     key=qdrant_key,
    #                     match=match
    #                 )
    #             )
    #         qdrant_filter = Filter(should=should_conditions)

    #     try:
    #         results = self.vector_store.similarity_search(  
    #             query=query,
    #             k=k,
    #             filter=qdrant_filter
    #         )
    #         print(f"Found {len(results)} results.")
    #         return results
    #     except Exception as e:
    #         print(f"Error during search: {e}")
    #         return []


    def db_select(self, filters=None, limit=5000):
        if filters != None:
            # should_conditions = []
            # for key, value in filters.items():
            #     print(key)
            #     print(value)
            #     qdrant_key = f"metadata.{key}"
            #     should_conditions.append(
            #         FieldCondition(
            #             key=qdrant_key,
            #             match=MatchValue(value=value[0])
            #         )
            #     )
            # qdrant_filter = Filter(should=should_conditions)
            should_conditions = []
            for key, value in filters.items():
                qdrant_key = f"metadata.{key}"
                if value == None or len(value) == 0:
                    continue
                elif len(value) > 1:
                    match = MatchAny(any=value)
                else:
                    match = MatchValue(value=value[0])
                should_conditions.append(
                    FieldCondition(
                        key=qdrant_key,
                        match=match
                    )
                )
            qdrant_filter = Filter(should=should_conditions)
        else:
            qdrant_filter = None
        results = self.client.scroll(self.collection_name, scroll_filter = qdrant_filter, limit=limit)
        return results
                

# --- Example Usage ---
if __name__ == "__main__":
    # Define some documents
    document_1 = Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
    )
    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees Fahrenheit.",
        metadata={"source": "news"},
    )
    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
    )
    document_4 = Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
    )
    document_5 = Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
    )
    document_6 = Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
    )
    document_7 = Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
    )
    document_8 = Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet"},
    )
    document_9 = Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
    )
    document_10 = Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
    )

    documents_to_add = [
        document_1, document_2, document_3, document_4, document_5,
        document_6, document_7, document_8, document_9, document_10,
    ]

    # 1. Create an instance of VectorStore
    my_vector_store = VectorStore(collection_name="my_app_docs")

    # 2. Populate the vector store with multiple documents
    my_vector_store.populate_vector_store(documents_to_add)

    # 3. Add a single new document
    new_document = Document(
        page_content="New research shows AI can help predict climate change impacts.",
        metadata={"source": "research_paper"},
    )
    my_vector_store.add_document(new_document)

    # 4. Perform a search without filters
    print("\n--- Search without filters ---")
    results_no_filter = my_vector_store.search(
        query="LangChain makes building LLM apps easy",
        k=3
    )

    # 5. Perform a search with filters (e.g., only from 'tweet' source)
    print("\n--- Search with 'tweet' filter ---")
    results_with_filter = my_vector_store.search(
        query="LangGraph",
        k=2,
        filters={"source": "tweet"}
    )

    # 6. Perform another search with a different filter
    print("\n--- Search with 'news' filter ---")
    results_news_filter = my_vector_store.search(
        query="economic situation",
        k=2,
        filters={"source": "news"}
    )

    # 7. Search for something not present to see empty results
    print("\n--- Search for non-existent content ---")
    results_empty = my_vector_store.search(
        query="unicorns flying on rainbows",
        k=1
    )
