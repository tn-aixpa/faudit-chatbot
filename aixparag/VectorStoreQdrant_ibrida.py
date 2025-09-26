"""
Serializable VectorStore Class for Qdrant
"""

from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny, SparseVectorParams
from langchain_core.documents import Document
from typing import List, Dict, Optional, Any


class VectorStore:
    """
    A class to manage a Qdrant vector store, providing methods for
    initialization, document addition, and similarity search.
    Fully serializable by storing only model names and recreating
    unpickleable objects on restore.
    """

    def __init__(
        self,
        collection_name: str = "demo_collection",
        vector_size: int = None,
        distance: Distance = Distance.COSINE,
        model_name: str = "dbmdz/bert-base-italian-uncased",
        sparse_model_name: str = "Qdrant/bm25",
    ):
        """
        Initializes the VectorStore with an in-memory Qdrant client and creates
        a collection.
        """
        print(f"Initializing VectorStore with collection: '{collection_name}'...")

        # --- Store configs instead of objects (needed for serialization) ---
        self.collection_name = collection_name
        self.model_name = model_name
        self.sparse_model_name = sparse_model_name
        self.distance = distance
        self.vector_size = vector_size

        # --- Build runtime objects ---
        self._initialize_runtime()

    def _initialize_runtime(self):
        """Helper: creates embeddings, client, sparse embedding, and vector store."""
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.client = QdrantClient(":memory:")
        self.sparse_embedding = FastEmbedSparse(model_name=self.sparse_model_name)

        if self.vector_size is None:
            dummy_text = "This is a test sentence."
            self.vector_size = len(self.embeddings.embed_query(dummy_text))

        # Create collection
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=self.vector_size, distance=self.distance)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                },
            )
            print(f"Collection '{self.collection_name}' created/recreated successfully.")
        except Exception as e:
            print(f"Error creating/recreating collection '{self.collection_name}': {e}")

        # Hybrid vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embedding,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        print("VectorStore initialized.")

    # ----------------- SERIALIZATION -----------------

    def __getstate__(self):
        """Return a serializable state (no heavy objects)."""
        state = {
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "sparse_model_name": self.sparse_model_name,
            "distance": self.distance,
            "vector_size": self.vector_size,
        }
        return state

    def __setstate__(self, state):
        """Restore from serialized state by reinitializing runtime objects."""
        self.collection_name = state["collection_name"]
        self.model_name = state["model_name"]
        self.sparse_model_name = state["sparse_model_name"]
        self.distance = state["distance"]
        self.vector_size = state["vector_size"]

        # Recreate embeddings, client, and vector store
        self._initialize_runtime()

    # ----------------- METHODS -----------------

    def populate_vector_store(self, documents: List[Document], ids=None):
        """Populates the vector store with a list of documents."""
        if not documents:
            print("No documents provided to populate.")
            return

        print(f"Adding {len(documents)} documents to the vector store...")
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]
        try:
            self.vector_store.add_documents(documents=documents, ids=ids)
            print(f"Successfully added {len(documents)} documents.")
        except Exception as e:
            print(f"Error adding documents: {e}")

    def add_document(self, document: Document, id=None):
        """Adds a single document to the vector store."""
        print(f"Adding single document: '{document.page_content[:50]}...'")
        try:
            if id is None:
                id = str(uuid4())
            self.vector_store.add_documents(documents=[document], ids=[id])
            print("Document added successfully.")
        except Exception as e:
            print(f"Error adding document: {e}")

    def search(
        self,
        query: str,
        k: int = 2,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Performs a similarity search in the vector store."""
        print(f"\nSearching for: '{query}' (k={k}) with filters: {filters}")
        qdrant_filter = None
        if filters:
            should_conditions = []
            must_conditions = []
            for key, value in filters.items():
                qdrant_key = f"metadata.{key}"
                if value is None or len(value) == 0 or value[0] == "None":
                    continue
                elif len(value) > 1:
                    match = MatchAny(any=value)
                else:
                    match = MatchValue(value=value[0])
                if key == "luogo":
                    must_conditions.append(FieldCondition(key=qdrant_key, match=match))
                else:
                    should_conditions.append(FieldCondition(key=qdrant_key, match=match))
            qdrant_filter = Filter(must=must_conditions)

        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=qdrant_filter,
            )
            print(f"Found {len(results)} results.")
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def db_select(self, filters=None, limit=5000):
        """Fetch raw entries from Qdrant with optional filters."""
        if filters is not None:
            should_conditions = []
            for key, value in filters.items():
                qdrant_key = f"metadata.{key}"
                if value is None or len(value) == 0:
                    continue
                elif len(value) > 1:
                    match = MatchAny(any=value)
                else:
                    match = MatchValue(value=value[0])
                should_conditions.append(FieldCondition(key=qdrant_key, match=match))
            qdrant_filter = Filter(should=should_conditions)
        else:
            qdrant_filter = None
        results = self.client.scroll(
            self.collection_name, scroll_filter=qdrant_filter, limit=limit
        )
        return results
