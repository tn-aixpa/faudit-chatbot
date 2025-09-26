import numpy as np
import os
import pickle
from typing import Union, List, Dict, Optional

# Import LangChain components for FAISS and embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document # For creating LangChain Document objects

class VectorStore:
    """
    A class to manage a FAISS-based vector store using LangChain's FAISS integration
    for RAG applications, with support for filtering.

    It uses HuggingFaceBgeEmbeddings for generating embeddings and FAISS for efficient
    similarity search and persistence.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the VectorStore with a SentenceTransformer embedder.

        Args:
            model_name (str): The name of the pre-trained sentence transformer model
                              to use for generating embeddings.
                              (e.g., 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2').
        """
        print(f"Initializing HuggingFaceBgeEmbeddings model: {model_name}...")
        try:
            self.embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)
            print("HuggingFaceBgeEmbeddings model loaded successfully.")
        except Exception as e:
            print(f"Error loading HuggingFaceBgeEmbeddings model: {e}")
            print("Please ensure 'sentence-transformers' and 'langchain-community' libraries are installed (`pip install sentence-transformers langchain-community`).")
            self.embeddings = None # Set to None if loading fails

        self.index: Optional[FAISS] = None  # LangChain FAISS vector store object

    def add_doc(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        """
        Adds a single document to the vector store.

        Args:
            doc_id (str): A unique identifier for the document.
            text (str): The content of the document.
            metadata (Optional[Dict]): Optional dictionary of metadata for the document.
                                       This is used for filtering.
        """
        if self.embeddings is None:
            print("Embedder not available. Cannot add document.")
            return

        if metadata is None:
            metadata = {}
        metadata['doc_id'] = doc_id # Ensure doc_id is part of metadata for retrieval

        # LangChain's FAISS.add_texts expects a list of texts and a list of metadata dicts
        if self.index is None:
            # If index is not initialized, create it from the first document
            self.index = FAISS.from_texts(
                texts=[text],
                embedding=self.embeddings,
                metadatas=[metadata]
            )
            print(f"Document '{doc_id}' added and FAISS index initialized.")
        else:
            # Otherwise, add to the existing index
            self.index.add_texts(
                texts=[text],
                metadatas=[metadata]
            )
            print(f"Document '{doc_id}' added.")

    def create_vector_store(self, docs: List[Dict]):
        """
        Creates the vector store from a list of documents.
        This will overwrite any existing index and documents.

        Args:
            docs (List[Dict]): A list of dictionaries, where each dictionary
                               must have 'id' and 'text' keys, and can optionally
                               have a 'metadata' key (a dictionary).
                               Example: [{"id": "doc1", "text": "Content of doc1", "metadata": {"category": "AI"}}, ...]
        """
        if not docs:
            print("No documents provided to create vector store.")
            return
        if self.embeddings is None:
            print("Embedder not available. Cannot create vector store.")
            return

        print(f"Creating vector store with {len(docs)} documents...")

        texts_list = []
        metadatas_list = []
        for doc in docs:
            text = doc.get("text")
            doc_id = doc.get("id")
            doc_metadata = doc.get("metadata", {})

            if text is None or doc_id is None:
                print(f"Skipping document due to missing 'text' or 'id': {doc}")
                continue

            doc_metadata['doc_id'] = doc_id # Ensure doc_id is part of metadata
            texts_list.append(text)
            metadatas_list.append(doc_metadata)

        if not texts_list:
            print("No valid documents to process after filtering for missing 'text' or 'id'.")
            return

        # Create FAISS index from texts and metadatas
        self.index = FAISS.from_texts(
            texts=texts_list,
            embedding=self.embeddings,
            metadatas=metadatas_list
        )
        print(f"Vector store successfully created with {len(texts_list)} documents.")

    def search(self, query_text: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Searches for the top-k most similar documents to the given query, with optional filtering.

        Args:
            query_text (str): The query string.
            k (int): The number of top similar documents to retrieve.
            filters (Optional[Dict]): A dictionary of filters to apply to the metadata.
                                      Example: {"category": "AI", "year": 2023}.

        Returns:
            List[Dict]: A list of dictionaries, each containing 'document_id',
                        'text', 'distance', and 'metadata' of the retrieved documents.
                        Returns an empty list if the index is not initialized.
        """
        if self.index is None:
            print("Vector store is empty or not initialized. Please create or load one first.")
            return []
        if self.embeddings is None:
            print("Embedder not available. Cannot perform search.")
            return []

        print(f"Searching for '{query_text}' with k={k} and filters={filters}")
        # LangChain's similarity_search_with_score returns a list of (Document, score) tuples
        # The score is typically L2 distance (lower is better for similarity).
        retrieved_docs_with_scores = self.index.similarity_search_with_score(
            query=query_text,
            k=k,
            filter=filters # Pass the filters directly
        )

        results = []
        for doc, score in retrieved_docs_with_scores:
            # LangChain Document objects have .page_content (text) and .metadata
            result_doc = {
                "document_id": doc.metadata.get('doc_id', 'N/A'), # Retrieve doc_id from metadata
                "text": doc.page_content,
                "distance": float(score), # Score is the distance
                "metadata": doc.metadata
            }
            results.append(result_doc)
        return results

    def save(self, local_path: str = "faiss_index_lc", index_name: str = "index"):
        """
        Saves the LangChain FAISS index to a local directory.

        Args:
            local_path (str): The directory path to save the FAISS index.
        """
        if self.index is not None:
            try:
                self.index.save_local(local_path, index_name)
                print(f"Vector store saved successfully to directory: {local_path}")
            except Exception as e:
                print(f"Error saving vector store: {e}")
        else:
            print("No FAISS index to save. Please create or load one first.")

    def load(self, local_path: str = "faiss_index_lc", index_name: str = "index") -> bool:
        """
        Loads the LangChain FAISS index from a local directory.

        Args:
            local_path (str): The directory path from which to load the FAISS index.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        if self.embeddings is None:
            print("Embedder not initialized. Cannot load vector store.")
            return False

        if os.path.exists(local_path) and os.path.isdir(local_path):
            try:
                self.index = FAISS.load_local(local_path, self.embeddings, index_name=index_name, allow_dangerous_deserialization=True)
                print(f"Vector store loaded successfully from directory: {local_path}")
                return True
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return False
        else:
            print(f"Directory not found: {local_path}")
            return False

# Example Usage:
if __name__ == "__main__":
    # 1. Initialize the VectorStore
    my_vector_store = VectorStore(model_name='all-MiniLM-L6-v2') # Or 'paraphrase-multilingual-MiniLM-L12-v2'

    # Define some sample documents with metadata for filtering
    sample_docs = [
        {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog.", "metadata": {"category": "animals", "source": "fables"}},
        {"id": "doc2", "text": "Artificial intelligence is revolutionizing many industries.", "metadata": {"category": "AI", "year": 2023}},
        {"id": "doc3", "text": "Machine learning is a subset of AI, focusing on algorithms.", "metadata": {"category": "AI", "year": 2022}},
        {"id": "doc4", "text": "Dogs are known for their loyalty and companionship.", "metadata": {"category": "animals", "type": "pet"}},
        {"id": "doc5", "text": "Natural Language Processing (NLP) deals with human language.", "metadata": {"category": "AI", "subfield": "NLP"}},
        {"id": "doc6", "text": "Quantum computing promises to solve complex problems faster.", "metadata": {"category": "computing", "year": 2024}},
        {"id": "doc7", "text": "The cat sat on the mat, watching the birds.", "metadata": {"category": "animals", "type": "pet"}},
        {"id": "doc8", "text": "Deep learning models require vast amounts of data.", "metadata": {"category": "AI", "subfield": "DL"}},
        {"id": "doc9", "text": "The sun is shining brightly today.", "metadata": {"category": "nature", "weather": "sunny"}},
        {"id": "doc10", "text": "The internet has connected the world like never before.", "metadata": {"category": "technology", "era": "modern"}}
    ]

    # 2. Create the vector store from a list of documents
    my_vector_store.create_vector_store(sample_docs)

    # You can also add documents one by one after creation
    # my_vector_store.add_doc("doc11", "New document about data science.", {"category": "AI", "year": 2024})

    # 3. Perform searches with and without filters
    print("\n--- Search without filters ---")
    query_no_filter = "What is AI and machine learning?"
    search_results_no_filter = my_vector_store.search(query_no_filter, k=3)
    print("\nSearch Results (No Filter):")
    for result in search_results_no_filter:
        print(f"  ID: {result['document_id']}, Distance: {result['distance']:.4f}, Metadata: {result['metadata']}")
        print(f"  Text: {result['text']}\n")

    print("\n--- Search with category filter ('AI') ---")
    query_with_filter = "What is NLP?"
    filters_ai = {"category": "AI"}
    search_results_ai = my_vector_store.search(query_with_filter, k=3, filters=filters_ai)
    print("\nSearch Results (Category: AI):")
    for result in search_results_ai:
        print(f"  ID: {result['document_id']}, Distance: {result['distance']:.4f}, Metadata: {result['metadata']}")
        print(f"  Text: {result['text']}\n")

    print("\n--- Search with category and year filter ('AI', year=2023) ---")
    query_with_year_filter = "Recent advancements in artificial intelligence."
    filters_ai_2023 = {"category": "AI", "year": 2023}
    search_results_ai_2023 = my_vector_store.search(query_with_year_filter, k=1, filters=filters_ai_2023)
    print("\nSearch Results (Category: AI, Year: 2023):")
    for result in search_results_ai_2023:
        print(f"  ID: {result['document_id']}, Distance: {result['distance']:.4f}, Metadata: {result['metadata']}")
        print(f"  Text: {result['text']}\n")

    print("\n--- Search for animals (type: pet) ---")
    query_animals_pet = "Loyal companions."
    filters_animals_pet = {"category": "animals", "type": "pet"}
    search_results_animals_pet = my_vector_store.search(query_animals_pet, k=2, filters=filters_animals_pet)
    print("\nSearch Results (Category: Animals, Type: Pet):")
    for result in search_results_animals_pet:
        print(f"  ID: {result['document_id']}, Distance: {result['distance']:.4f}, Metadata: {result['metadata']}")
        print(f"  Text: {result['text']}\n")

    # 4. Save the vector store
    save_dir = "my_faiss_index_lc_data"
    my_vector_store.save(save_dir)

    # 5. Load the vector store (simulate a new session)
    print("\n--- Simulating new session and loading vector store ---")
    new_vector_store = VectorStore()
    if new_vector_store.load(save_dir):
        # Perform a search with the loaded store and a filter
        query_loaded_filter = "What is deep learning?"
        loaded_search_results = new_vector_store.search(query_loaded_filter, k=1, filters={"subfield": "DL"})
        print("\nLoaded Search Results (Filtered by subfield: DL):")
        for result in loaded_search_results:
            print(f"  ID: {result['document_id']}, Distance: {result['distance']:.4f}, Metadata: {result['metadata']}")
            print(f"  Text: {result['text']}\n")

    # Clean up saved files (optional)
    # import shutil
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    # print(f"Cleaned up saved directory: {save_dir}")
