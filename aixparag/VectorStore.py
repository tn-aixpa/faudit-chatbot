import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
from typing import Union

class VectorStore:
    """
    A class to manage a FAISS-based vector store for RAG applications.

    It uses SentenceTransformer for generating embeddings and FAISS for efficient
    similarity search.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the VectorStore with a SentenceTransformer embedder.

        Args:
            model_name (str): The name of the pre-trained sentence transformer model
                              to use for generating embeddings.
                              (e.g., 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2').
        """
        print(f"Initializing SentenceTransformer model: {model_name}...")
        try:
            self.embedder = SentenceTransformer(model_name)
            print("SentenceTransformer model loaded successfully.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            print("Please ensure 'sentence-transformers' library is installed (`pip install sentence-transformers`).")
            self.embedder = None # Set to None if loading fails

        self.index = None  # FAISS index (e.g., IndexFlatL2)
        self.documents = []  # List of original documents (dictionaries with 'id' and 'text')
        self.doc_embeddings = None # NumPy array of all document embeddings

    def _embed_text(self, text: Union[str, list[str]]) -> np.ndarray:
        """
        Internal helper method to generate embeddings for given text(s).

        Args:
            text (str or list[str]): The text or list of texts to embed.

        Returns:
            np.ndarray: A NumPy array of embeddings.
        """
        if self.embedder is None:
            raise RuntimeError("Embedder not initialized. Cannot generate embeddings.")
        # encode method handles both single string and list of strings
        return self.embedder.encode(text, convert_to_numpy=True)

    def add_doc(self, doc_id: str, text: str):
        """
        Adds a single document to the vector store.

        Args:
            doc_id (str): A unique identifier for the document.
            text (str): The content of the document.
        """
        if self.embedder is None:
            print("Embedder not available. Cannot add document.")
            return

        embedding = self._embed_text(text)
        # Ensure embedding is 2D for FAISS (even for a single vector)
        embedding = np.array([embedding])

        if self.index is None:
            # Initialize FAISS index if it's the first document
            dimension = embedding.shape[1]
            # Using IndexFlatL2 for Euclidean distance (common for similarity)
            self.index = faiss.IndexFlatL2(dimension)
            self.doc_embeddings = embedding
        else:
            # Append new embedding to existing ones
            self.doc_embeddings = np.vstack([self.doc_embeddings, embedding])

        # Add the embedding to the FAISS index
        self.index.add(embedding)
        # Store the original document with its ID
        self.documents.append({"id": doc_id, "text": text})
        print(f"Document '{doc_id}' added.")

    def create_vector_store(self, docs: list[dict]):
        """
        Creates the vector store from a list of documents.
        This will overwrite any existing index and documents.

        Args:
            docs (list[dict]): A list of dictionaries, where each dictionary
                               must have 'id' and 'text' keys.
                               Example: [{"id": "doc1", "text": "Content of doc1"}, ...]
        """
        if not docs:
            print("No documents provided to create vector store.")
            return
        if self.embedder is None:
            print("Embedder not available. Cannot create vector store.")
            return

        print(f"Creating vector store with {len(docs)} documents...")
        texts = [doc["text"] for doc in docs]
        doc_ids = [doc["id"] for doc in docs]

        # Generate embeddings for all documents at once
        self.doc_embeddings = self._embed_text(texts)
        self.documents = [{"id": doc_ids[i], "text": texts[i]} for i in range(len(docs))]

        # Initialize FAISS index
        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        # Add all embeddings to the index
        self.index.add(self.doc_embeddings)
        print(f"Vector store successfully created with {len(docs)} documents.")

    def search(self, query_text: str, k: int = 5) -> list[dict]:
        """
        Searches for the top-k most similar documents to the given query.

        Args:
            query_text (str): The query string.
            k (int): The number of top similar documents to retrieve.

        Returns:
            list[dict]: A list of dictionaries, each containing 'document_id',
                        'text', and 'distance' of the retrieved documents.
                        Returns an empty list if the index is not initialized.
        """
        if self.index is None or self.index.ntotal == 0:
            print("Vector store is empty or not initialized. Please add documents first.")
            return []
        if self.embedder is None:
            print("Embedder not available. Cannot perform search.")
            return []

        # Embed the query text
        query_embedding = self._embed_text(query_text)
        # FAISS search method expects a 2D array, even for a single query
        query_embedding = np.array([query_embedding])

        # Perform the search
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            # Check if a valid index was returned (FAISS returns -1 for empty slots)
            if idx != -1 and idx < len(self.documents):
                results.append({
                    "document_id": self.documents[idx]["id"],
                    "text": self.documents[idx]["text"],
                    "distance": float(distances[0][i]) # Convert numpy float to standard float
                })
        return results

    def save(self, index_path: str = "faiss_index.bin", docs_path: str = "documents.pkl"):
        """
        Saves the FAISS index and the list of original documents to disk.

        Args:
            index_path (str): The file path to save the FAISS index.
            docs_path (str): The file path to save the pickled list of documents.
        """
        if self.index is not None:
            try:
                faiss.write_index(self.index, index_path)
                with open(docs_path, 'wb') as f:
                    pickle.dump(self.documents, f)
                print(f"Vector store saved successfully to {index_path} and {docs_path}")
            except Exception as e:
                print(f"Error saving vector store: {e}")
        else:
            print("No FAISS index to save. Please create or load one first.")

    def load(self, index_path: str = "faiss_index.bin", docs_path: str = "documents.pkl") -> bool:
        """
        Loads the FAISS index and the list of original documents from disk.

        Args:
            index_path (str): The file path from which to load the FAISS index.
            docs_path (str): The file path from which to load the pickled list of documents.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"Vector store loaded successfully from {index_path} and {docs_path}")
                return True
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return False
        else:
            print(f"One or both files not found: {index_path}, {docs_path}")
            return False

# Example Usage:
if __name__ == "__main__":
    # 1. Initialize the VectorStore
    my_vector_store = VectorStore(model_name='paraphrase-multilingual-MiniLM-L12-v2')

    # Define some sample documents
    sample_docs = [
        {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog."},
        {"id": "doc2", "text": "Artificial intelligence is revolutionizing many industries."},
        {"id": "doc3", "text": "Machine learning is a subset of AI, focusing on algorithms."},
        {"id": "doc4", "text": "Dogs are known for their loyalty and companionship."},
        {"id": "doc5", "text": "Natural Language Processing (NLP) deals with human language."},
        {"id": "doc6", "text": "Quantum computing promises to solve complex problems faster."},
        {"id": "doc7", "text": "The cat sat on the mat, watching the birds."},
        {"id": "doc8", "text": "Deep learning models require vast amounts of data."},
        {"id": "doc9", "text": "The sun is shining brightly today."},
        {"id": "doc10", "text": "The internet has connected the world like never before."}
    ]

    # 2. Create the vector store from a list of documents
    my_vector_store.create_vector_store(sample_docs)

    # You can also add documents one by one after creation or if starting empty
    # my_vector_store.add_doc("doc11", "New document about data science.")

    # 3. Perform a search
    query = "What is AI and machine learning?"
    print(f"\nSearching for: '{query}'")
    search_results = my_vector_store.search(query, k=3)

    print("\nSearch Results:")
    for result in search_results:
        print(f"  ID: {result['document_id']}, Distance: {result['distance']:.4f}")
        print(f"  Text: {result['text']}\n")

    query_animals = "Animals that are loyal and good companions."
    print(f"\nSearching for: '{query_animals}'")
    search_results_animals = my_vector_store.search(query_animals, k=2)

    print("\nSearch Results (Animals):")
    for result in search_results_animals:
        print(f"  ID: {result['document_id']}, Distance: {result['distance']:.4f}")
        print(f"  Text: {result['text']}\n")

    # 4. Save the vector store
    my_vector_store.save("vector_store/my_faiss_index.bin", "vector_store/my_documents.pkl")

    # 5. Load the vector store (simulate a new session)
    print("\n--- Simulating new session and loading vector store ---")
    new_vector_store = VectorStore()
    if new_vector_store.load("vector_store/my_faiss_index.bin", "vector_store/my_documents.pkl"):
        # Perform a search with the loaded store
        query_loaded = "What is NLP?"
        print(f"\nSearching with loaded store for: '{query_loaded}'")
        loaded_search_results = new_vector_store.search(query_loaded, k=1)
        print("\nLoaded Search Results:")
        for result in loaded_search_results:
            print(f"  ID: {result['document_id']}, Distance: {result['distance']:.4f}")
            print(f"  Text: {result['text']}\n")

    # Clean up saved files (optional)
    # if os.path.exists("my_faiss_index.bin"):
    #     os.remove("my_faiss_index.bin")
    # if os.path.exists("my_documents.pkl"):
    #     os.remove("my_documents.pkl")
    # print("Cleaned up saved files.")
