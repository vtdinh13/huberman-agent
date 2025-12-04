import os
from threading import Lock
from typing import Callable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer


class QdrantSearchClient:

    def __init__(
        self,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        use_cloud: Optional[bool] = None,
        cloud_url: Optional[str] = None,
        cloud_api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        self.payload_fields = ["chunk", "episode_name", "start", "end"]

        model_name = "all-MPNet-base-v2"
        self.embedding_model = SentenceTransformer(model_name)
        self._embed_lock = Lock()

        # Decide whether to use a cloud deployment vs the local docker container.
        env_flag = os.getenv("QDRANT_USE_CLOUD")
        if use_cloud is None:
            if env_flag is not None:
                use_cloud = env_flag.lower() == "true"
            else:
                use_cloud = bool(url or os.getenv("QDRANT_ENDPOINT"))

        provided_api_key = api_key or os.getenv("QDRANT_API_KEY")
        resolved_cloud_url = cloud_url or url or os.getenv("QDRANT_ENDPOINT")
        resolved_cloud_key = cloud_api_key or provided_api_key
        self.mode = "cloud" if use_cloud else "local"

        if self.mode == "cloud":
            self.client = QdrantClient(url=resolved_cloud_url, api_key=resolved_cloud_key, prefer_grpc=prefer_grpc)
        else:
            host = "127.0.0.1"
            port = "6333"
            self.client = QdrantClient(
                host=host,
                port=port,
                api_key=provided_api_key,
                prefer_grpc=prefer_grpc,
            )

    def embed_query(self, query: str) -> List[float]:
        if not query:
            raise ValueError("Query text cannot be empty.")
        with self._embed_lock:
            vector = self.embedding_model.encode(query, convert_to_numpy=True)
        if hasattr(vector, "tolist"):
            return vector.tolist()
        if isinstance(vector, list):
            return vector
        return list(vector)

    def search_embeddings(
        self,
        query: str,
        collection_name="transcripts",
        num_results=15,
        num_candidates=100,
    ) -> List[dict[str, str]]:
        
        if not query:
            raise ValueError("query is required.")

        limit = num_results or self.num_results
        candidates = num_candidates or max(limit * 5, self.num_candidates)
        vector = self.embed_query(query)

        params = qmodels.SearchParams(hnsw_ef=candidates, exact=False)
        response = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            with_payload=self.payload_fields,
            search_params=params,
        )

        points = getattr(response, "result", response)

        results: List[dict[str, str]] = []
        for point in points:
            payload: dict[str, str] = {}
            score = None

            if isinstance(point, tuple):
                raw_point, score = point[0], point[1]
            else:
                raw_point = point
                score = getattr(raw_point, "score", None)

            if hasattr(raw_point, "payload"):
                payload = raw_point.payload or {}
            elif isinstance(raw_point, dict):
                payload = raw_point
            else:
                # Fallback when only raw text/strings are returned; treat as chunk.
                payload = {"chunk": str(raw_point)}
            results.append(
                {
                    "episode_name": payload.get("episode_name", ""),
                    "start": payload.get("start", ""),
                    "end": payload.get("end", ""),
                    "chunk": payload.get("chunk", ""),
                    "score": score,
                }
            )
        return results


def create_qdrant_search_tools(
    **client_kwargs,
) -> tuple[
    Callable[[str], List[float]],
    Callable[[str, Optional[str], Optional[int], Optional[int]], List[dict[str, str]]],
]:
    """
    Factory that returns embedding and search callables backed by Qdrant.
    """
    client = QdrantSearchClient(**client_kwargs)
    def embedding(query: str) -> List[float]:
        vector = client.embed_query(query)
        if hasattr(vector, "tolist"):
            return vector.tolist()
        if isinstance(vector, list):
            return vector
        return list(vector)
    embedding.__name__ = "embed_query"

    def search(
        query: str,
        collection_name: Optional[str] = None,
        num_results: Optional[int] = None,
        num_candidates: Optional[int] = None,
    ) -> List[dict[str, str]]:
        target_collection = collection_name or client.collection_name if hasattr(client, "collection_name") else "transcripts"
        return client.search_embeddings(
            query=query,
            collection_name=target_collection,
            num_results=num_results or client.DEFAULT_RESULTS if hasattr(client, "DEFAULT_RESULTS") else 15,
            num_candidates=num_candidates or client.DEFAULT_CANDIDATES if hasattr(client, "DEFAULT_CANDIDATES") else 100,
        )
    search.__name__ = "vector_search"

    return embedding, search
