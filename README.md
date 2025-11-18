## Introduction
The Huberman Lab podcast consistently ranks in the top 5 across Apple and Spotify in the Health, Fitness, and Science categories, with over 7 million YouTube subscribers. While unequivocally popular, the episodes are long and often not easy to digest. Each episode averages 120 minutes based on a calculation of 354 total episodes. The longest episode, featuring Dr. Andy Galpin on "Optimal Protocols to Build Strength and Muscle," runs 279 minutes — that’s over 4.5 hours! 

The podcast offers both knowledge and valuable tools that could improve our lives, but this content is hidden in excessively long episodes. An agentic system addresses this gap by acting as a personalized coach that extracts content from the podcast's knowledge base to recommend actionable tools that users can immediately implement. By grounding recommendations in evidence-based research and expert interviews from the podcast, the system ensures that recommendations are both scientifically sound and immediately actionable.


## Setup
- Python 3.11.14 was used for development. Faster-Whisper supports Python 3.8–3.11 (no 3.12 yet), so use one of those versions; 3.11.x is recommended for best compatibility with this project.

- `docker-compose.yaml` starts Postgres + pgAdmin, Elasticsearch + Kibana, and Grafana.
  ```bash
  docker compose up -d
  ```
  Postgres listens on `localhost:5434` with `podcast/podcast`; pgAdmin on `http://localhost:8080` (admin/admin). Elasticsearch on `http://localhost:9200`, Kibana at `http://localhost:5601`, Grafana at `http://localhost:3000` (admin/admin).
- uv is used to manage Python packages and environments (`pyproject.toml`/`uv.lock` tracked in-repo).
- API keys are managed via `.direnv` (e.g., `.envrc` file loaded by direnv); ensure your environment files are present before sending requests to a LLM.

## Ingestion
- Transcription pipeline (`transcription.py`): fetch RSS feed, build the download queue, download audio, transcribe mp3s, and write transcripts to Postgres. Use `--limit` to cap how many new episodes run per invocation.

  ```bash
  python ingest-data/transcription.py \
    --rss-path ingest-data/huberman-rss.json \
    --media-dir ingest-data/media-files/huberman \
    --transcript-table transcripts \
    --database-url postgresql://podcast:podcast@localhost:5434/podcast-agent \
    --limit 20  # optional limit per run
  ```
- Chunking pipeline (`chunking.py`): turn transcript `.txt` files into chunks of words, embed them with SentenceTransformers (`all-MPNet-base-v2`), and bulk-index into Elasticsearch with duplicate-chunk protection. Includes sliding window chunking, embedding, metadata join, and index creation with a `dense_vector` field.
  ```bash
  python ingest-data/chunking.py \
  --index-name huberman \
  --meta-rss-csv ingest-data/rss-table.csv \
  --media-directory "ingest-data/media-files-sample/huberman/**.txt" \
  --limit 20 # # optional limit per run
  ```
