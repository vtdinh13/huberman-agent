## Setup
- Python 3.11.14 was used for development. Faster-Whisper supports Python 3.8–3.11 (no 3.12 yet), so use one of those versions; 3.11.x is recommended for best compatibility with this project.
- `docker-compose.yaml` starts Postgres + pgAdmin, Elasticsearch + Kibana, and Grafana.
  ```bash
  docker compose up -d
  ```
  Postgres listens on `localhost:5434` with `podcast/podcast`; pgAdmin on `http://localhost:8080` (admin/admin). Elasticsearch on `http://localhost:9200`, Kibana at `http://localhost:5601`, Grafana at `http://localhost:3000` (admin/admin).

## Ingestion
- Transcription pipeline: fetch RSS (`fetch_rss_feed`), build the download queue (`make_queue`), download audio (`download_media_file`), transcribe mp3s (`transcribe_audio_file`), and write transcripts to Postgres (`ingest`).
  ```bash
  python ingest-data/transcription.py \
    --rss-path ingest-data/huberman-rss.json \
    --media-dir ingest-data/media-files/huberman \
    --transcript-table transcripts \
    --database-url postgresql://podcast:podcast@localhost:5434/podcast-agent
  ```
- Chunking pipeline: turn transcript `.txt` files into timestamped word chunks, embed them with SentenceTransformers (`all-MPNet-base-v2`), and bulk-index into Elasticsearch with duplicate-chunk protection. Includes sliding window chunking, embedding, metadata join, and index creation with a `dense_vector` field.
  ```bash
  python ingest-data/chunking.py \
  --index-name huberman \
  --meta-rss-csv ingest-data/rss-table.csv \
  --media-directory "ingest-data/media-files-sample/huberman/**.txt" \
  --limit 20
  ```
