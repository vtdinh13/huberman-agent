import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from embedding import embed_batches, stream_chunks, stream_parquet_rows 

def create_collection(
    client: QdrantClient,
    collection_name: str,
    distance: str,
    recreate: bool,
    vector_size = 768
) -> None:
    """Ensure the target collection exists with the expected vector settings."""
    distance_map = {
        "cosine": qmodels.Distance.COSINE,
        "dot": qmodels.Distance.DOT,
        "euclid": qmodels.Distance.EUCLID,
    }
    if distance not in distance_map:
        raise ValueError(f"Unsupported distance '{distance}'.")

    if recreate and client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    if client.collection_exists(collection_name=collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size, distance=distance_map[distance]
        ),
    )


def stream_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    embedded_batches_iter: Iterable[List[Dict[str, object]]],
) -> int:
    """Upsert embedded chunk batches into Qdrant."""
    total = 0
    for batch_idx, batch in enumerate(
        tqdm(embedded_batches_iter, desc="Streaming to Qdrant"), 1
    ):
        ids = [chunk["chunk_hash"] for chunk in batch]
        vectors = [chunk["embedding"] for chunk in batch]
        payloads = []
        for chunk in batch:
            payload = dict(chunk)
            payload.pop("embedding", None)
            payloads.append(payload)
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=qmodels.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )
        total += len(batch)
        tqdm.write(
            f"Batch {batch_idx}: upserted {len(batch)} chunks (total {total})"
        )
    return total


def build_qdrant_client(args) -> QdrantClient:
    """Initialize a Qdrant client targeting either local Docker or Qdrant Cloud."""
    if args.target == "local":
        return QdrantClient(
            host="127.0.0.1",
            port=6333,
            prefer_grpc=args.use_grpc,
        )

    if not args.cloud_url:
        raise ValueError("Missing --cloud-url for Qdrant Cloud target.")
    api_key = args.cloud_api_key or os.getenv("QDRANT_API_KEY")
    if not api_key:
        raise ValueError("Provide --cloud-api-key or set QDRANT_API_KEY.")

    return QdrantClient(
        url=args.cloud_url,
        api_key=api_key,
        prefer_grpc=args.use_grpc,
        timeout=60
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload transcript embeddings to local or cloud Qdrant."
    )
    parser.add_argument(
        "--parquet-path",
        required=True,
        help="Path to transcripts.parquet.",
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Collection name to upsert into.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Words per chunk window.",
    )
    parser.add_argument(
        "--chunk-step",
        type=int,
        default=200,
        help="Sliding window stride.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=256,
        help="Rows to pull per Parquet batch.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=256,
        help="Chunks per embedding call.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max transcript rows to process.",
    )
    parser.add_argument(
        "--distance",
        choices=["cosine", "dot", "euclid"],
        default="cosine",
        help="Vector distance metric.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingesting.",
    )
    parser.add_argument(
        "--target",
        choices=["local", "cloud"],
        default="local",
        help="Destination: dockerized local Qdrant or Qdrant Cloud.",
    )
    parser.add_argument(
        "--cloud-url",
        default=os.getenv("QDRANT_ENDPOINT"),
        help="HTTPS endpoint for Qdrant Cloud.",
    )
    parser.add_argument(
        "--cloud-api-key",
        default=os.getenv("QDRANT_API_KEY"),
        help="API key for Qdrant Cloud.",
    )
    parser.add_argument(
        "--use-grpc",
        action="store_true",
        help="Use gRPC transport where supported.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point for streaming Parquet transcripts into Qdrant."""
    args = parse_args(argv)
    client = build_qdrant_client(args)

    create_collection(
        client=client,
        collection_name=args.collection_name,
        distance=args.distance,
        recreate=args.recreate,
    )

    row_stream = stream_parquet_rows(
        parquet_path=args.parquet_path,
        batch_size=args.parquet_batch_size,
        limit=args.limit,
    )
    chunk_stream = stream_chunks(
        rows=row_stream,
        chunk_size=args.chunk_size,
        chunk_step=args.chunk_step,
    )
    embedded_stream = embed_batches(
        chunks_iter=chunk_stream,
        embedding_batch_size=args.embedding_batch_size,
    )
    total = stream_to_qdrant(
        client=client,
        collection_name=args.collection_name,
        embedded_batches_iter=embedded_stream,
    )
    print(
        f"Uploaded {total} chunks to "
        f"{'local' if args.target == 'local' else 'cloud'} Qdrant collection "
        f"'{args.collection_name}'."
    )


if __name__ == "__main__":
    main()
