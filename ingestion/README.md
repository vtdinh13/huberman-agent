## Ingestion
- Transcription pipeline (`transcription.py`): fetch RSS feed, build the download queue, download audio, transcribe `.mp3` files, and write transcripts to Postgres. Use `--limit` to cap how many new episodes are transcribed per run. Use `--skip-postgres` to leave transcripts on current working directory without inserting into the database. By default, transcript `.txt` files are deleted after a successful Postgres insert; pass `--keep-transcripts` to retain them on disk. Run the following on CLI:

  ```
  python ingestion/transcription.py \
    --media-dir ingestion/media-files \
    --transcript-table transcripts
  # --limit 20           # only transcribe 20 new episodes on this run
  # --keep-transcripts   # keep transcript .txt files even after inserting into Postgres
  # --skip-postgres      # skip writing to Postgres; leaves transcripts on disk only
  ```

