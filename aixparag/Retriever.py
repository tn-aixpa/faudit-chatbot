import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder # For reranking
from .VectorStoreQdrant import VectorStore
from .global_cache import _GLOBAL_RERANKERS  # import the global cache
import statistics
from typing import Tuple

class Retriever:
    """
    A class to handle document retrieval and optional re-ranking for RAG applications.

    It wraps a VectorStore object and can optionally include a re-ranker.
    """

    def __init__(self, vector_store: VectorStore, reranker_model_name: Optional[str] = None):
        """
        Initializes the Retriever with a VectorStore and an optional re-ranker.

        Args:
            vector_store (VectorStore): An instance of the VectorStore class.
            reranker_model_name (Optional[str]): The name of the cross-encoder model
                                                  to use for re-ranking. If None,
                                                  re-ranking will not be performed.
                                                  (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2').
        """
        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must be an instance of VectorStore.")

        self.vector_store = vector_store
        self.reranker = None

        
        if reranker_model_name:
            if reranker_model_name not in _GLOBAL_RERANKERS:
                print(f"Loading reranker model once: {reranker_model_name}...")
                try:
                    _GLOBAL_RERANKERS[reranker_model_name] = CrossEncoder(reranker_model_name)
                    print("Reranker model loaded successfully.")
                except Exception as e:
                    print(f"Error loading reranker model: {e}")
                    _GLOBAL_RERANKERS[reranker_model_name] = None

            self.reranker = _GLOBAL_RERANKERS[reranker_model_name]

        # if reranker_model_name:
        #     print(f"Initializing reranker model: {reranker_model_name}...")
        #     try:
        #         self.reranker = CrossEncoder(reranker_model_name)
        #         print("Reranker model loaded successfully.")
        #     except Exception as e:
        #         print(f"Error loading reranker model: {e}")
        #         print("Please ensure 'sentence-transformers' library is installed (`pip install sentence-transformers`).")
        #         self.reranker = None # Set to None if loading fails

    def retrieve(self, query: str, k: int = 10, filters = None) -> List[Dict]:
        """
        Retrieves the top-k most relevant documents from the vector store.
        """
        # print(f"Retrieving initial top {k} documents for query: '{query}'")
        # print(f"Using filters: {filters}")
        
        # retrieved_docs = self.vector_store.search(query, k=k)

        retrieved_docs = self.vector_store.search(query, k=k, filters=filters)
        if len(retrieved_docs) == 0:
            # print("No documents retrieved from the vector store. Now running without filter.")
            retrieved_docs = self.vector_store.search(query, k=k)
        
        # print(f"Found {len(retrieved_docs)} documents during initial retrieval.")
        return retrieved_docs

    def rerank(self, query: str, documents: List[Dict], k: int = 5) -> List[Dict]:
        """
        Re-ranks a list of documents based on their relevance to the query using the reranker.

        Args:
            query (str): The original search query.
            documents (List[Dict]): A list of documents (e.g., from initial retrieval),
                                    each expected to have a 'text' key.

        Returns:
            List[Dict]: The re-ranked list of documents, with an added 'rerank_score' key.
                        If no reranker is initialized, returns the original list.
        """
        if not self.reranker:
            print("No reranker initialized. Returning documents without re-ranking.")
            return documents

        if not documents:
            print("No documents to rerank. Returning empty list.")
            return []

        # print(f"Re-ranking {len(documents)} documents for query: '{query}'")
        # Prepare pairs for the cross-encoder: (query, document_text)
        sentence_pairs = [[query, doc.page_content] for doc in documents]

        # Get scores from the cross-encoder
        # The cross-encoder outputs a single score per pair, indicating relevance.
        # Higher score means higher relevance.
        rerank_scores = self.reranker.predict(sentence_pairs)

        # Add rerank scores to the documents and sort them
        reranked_docs = []
        for i, doc in enumerate(documents):
            doc_with_score = {'page_content': doc.page_content}
            doc_with_score['rerank_score'] = float(rerank_scores[i]) # Convert numpy float
            reranked_docs.append(doc_with_score)

        # Sort by rerank_score in descending order (highest score first)
        reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

        # print("Documents re-ranked successfully.")
        return reranked_docs[:k]



    def rerank_scores(
        self, 
        query: str, 
        documents: List[Dict], 
        k: int = 5, 
        z_threshold: float = 2.0, 
        fallback_threshold: float = 0.3
    ) -> Tuple[List[Dict], List[float]]:
        """
        Re-ranks a list of documents based on their relevance to the query using the reranker.
        Stops early if the relative score drop is an outlier (statistical or rule-based).

        Args:
            query (str): The original search query.
            documents (List[Dict]): A list of documents, each expected to have a 'page_content' key.
            k (int): Maximum number of documents to return.
            z_threshold (float): Z-score threshold for detecting a significant drop (default=2.0).
            fallback_threshold (float): Absolute relative drop cutoff if too few samples (default=0.3).

        Returns:
            Tuple[List[Dict], List[float]]:
                - The re-ranked list of documents, each with a 'rerank_score' key.
                - A list of rerank scores corresponding to the returned documents.
        """
        if not self.reranker:
            print("No reranker initialized. Returning documents without re-ranking.")
            return documents, []

        if not documents:
            print("No documents to rerank. Returning empty list.")
            return [], []

        print(f"Re-ranking {len(documents)} documents for query: '{query}'")
        sentence_pairs = [[query, doc.page_content] for doc in documents]
        rerank_scores = self.reranker.predict(sentence_pairs)

        reranked_docs = [
            {"page_content": doc.page_content, "rerank_score": float(rerank_scores[i])}
            for i, doc in enumerate(documents)
        ]

        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Apply cutoff
        final_docs = [reranked_docs[0]]
        relative_drops = []

        for i in range(1, min(k, len(reranked_docs))):
            prev_score = final_docs[-1]["rerank_score"]
            curr_score = reranked_docs[i]["rerank_score"]

            relative_drop = (prev_score - curr_score) / max(prev_score, 1e-8)
            relative_drops.append(relative_drop)

            if len(relative_drops) < 3:  
                # Fallback rule for very few documents
                if relative_drop > fallback_threshold:
                    print(
                        f"Stopped early at rank {i} due to large drop "
                        f"(relative_drop={relative_drop:.4f}, threshold={fallback_threshold})"
                    )
                    break
            else:
                # Statistical cutoff using z-score
                mean_drop = statistics.mean(relative_drops[:-1])
                stdev_drop = statistics.pstdev(relative_drops[:-1]) or 1e-8
                z_score = (relative_drop - mean_drop) / stdev_drop

                if z_score > z_threshold:
                    print(
                        f"Stopped early at rank {i} due to statistical outlier "
                        f"(relative_drop={relative_drop:.4f}, z={z_score:.2f})"
                    )
                    break

            final_docs.append(reranked_docs[i])

        final_scores = [doc["rerank_score"] for doc in final_docs]

        print("Documents re-ranked successfully.")
        return final_docs, final_scores


    def evaluate(self, retrieved_documents: List[Dict], ground_truth: List[str]) -> Dict:
        """
        Evaluates the performance of the retriever.
        (Placeholder for future implementation)

        Args:
            retrieved_documents (List[Dict]): The documents retrieved by the system.
            ground_truth (List[str]): A list of ground truth relevant document IDs or texts.

        Returns:
            Dict: A dictionary containing evaluation metrics.
        """
        print("Evaluation function is a placeholder and not yet implemented.")
        return {"status": "Evaluation not implemented"}

# Example Usage:
if __name__ == "__main__":
    # 1. Initialize the VectorStore (using the simplified version for this example)
    print("--- Initializing VectorStore ---")
    my_vector_store = VectorStore(model_name='paraphrase-multilingual-MiniLM-L12-v2')
    my_vector_store.load("vector_store/my_faiss_index.bin", "vector_store/my_documents.pkl")

    # Initialize the Retriever with the VectorStore and a reranker
    print("\n--- Initializing Retriever with Reranker ---")
    my_retriever = Retriever(vector_store=my_vector_store)

    # Perform initial retrieval
    query = "Tell me about artificial intelligence and its subfields."
    print(f"\n--- Performing initial retrieval for query: '{query}' ---")
    initial_retrieved_docs = my_retriever.retrieve(query, k=5)

    print("\nInitial Retrieved Documents (before re-ranking):")
    for doc in initial_retrieved_docs:
        print(f"  ID: {doc['document_id']}, Distance: {doc['distance']:.4f}, Text: {doc['text']}")
