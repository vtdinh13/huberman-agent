from elasticsearch import Elasticsearch
import numpy as np

from sentence_transformers import SentenceTransformer
from threading import Lock
from typing import Optional, List

# from dataclasses import dataclass
# @dataclass
# class AgentConfig:
#     index_name: str = "huberman"
#     model: str = "openai:gpt-4o-mini"

class SearchTools:
    """
    Helper for encoding user queries and running vector searches against an Elasticsearch index.
    Instantiates and reuses an Elasticsearch client and sentence transformer using configured defaults.

    Attributes:
        db_url: URL of the Elasticsearch cluster.
        embedding_model: Name of the sentence transformer model to load.
        num_results: Default number of top hits to return.
        num_candidates: Default minimum number of candidates to evaluate in the kNN query.
    """

    db_url = "http://localhost:9200"
    embedding_model = "all-MPNet-base-v2"
    num_results = 15
    num_candidates = 1000
    index_name = "huberman"

    def __init__( self, index_name: Optional[str] = None, embedding_model_name: Optional[str] = None, 
                 db_url: Optional[str] = None, num_results: Optional[int] = None, 
                 num_candidates: Optional[int] = None):

        self.index_name = index_name or self.index_name
        self.db_url = db_url or self.db_url
        self.num_results = num_results or self.num_results
        self.num_candidates = num_candidates or self.num_candidates
        model_name = embedding_model_name or self.embedding_model

        self.db = Elasticsearch(self.db_url)
        self.embedding_model = SentenceTransformer(model_name)
        self._embed_lock = Lock()
        

    def embed_query(self, query:str) -> np.ndarray:
        """
        Encode a text query into a NumPy embedding using the sentence transformer.
        """
        if not query:
            raise ValueError('Query text cannot be empty.')
        with self._embed_lock:
            return self.embedding_model.encode(query, convert_to_numpy=True)

    def vector_search(self, index_name: Optional[str] = None, query: str = "", 
                      num_results: Optional[int] = None, 
                      num_candidates: Optional[int] = None) -> List[dict[str, str]]:
        """
        Run a kNN vector search against Elasticsearch for the given text query and return matched chunks with metadata.
        """

        target_index = index_name or self.index_name
        if not target_index:
            raise ValueError("index_name is required.")
        if not query:
            raise ValueError("query is required.")
        
        num_results = num_results or self.num_results
        num_candidates = num_candidates or max(num_results*5, self.num_candidates)

        query_embedding = self.embed_query(query)

        knn_query = {
            "size": num_results,
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding.tolist(),
                "k": num_results,
                "num_candidates": num_candidates
            },
            "_source": ["chunk", "episode_name", "start", "end"]
        }
        response = self.db.search(index=target_index, body=knn_query)

        results = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})

            results.append({
                "episode_name": source.get("episode_name", ""),
                "start": source.get("start", ""),
                "end": source.get("end", ""),
                "chunk": source.get("chunk", ""),
            })
        return results
