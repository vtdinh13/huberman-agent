import requests
import xml.etree.ElementTree as ET
import io

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm
import os
import logging
import pandas as pd

import psycopg
from psycopg import Connection, sql
from psycopg.rows import dict_row

from faster_whisper import WhisperModel

model = WhisperModel(model_size_or_path='tiny', device='cpu', compute_type='int8')

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")

def fetch_rss_feed(rss_url:str, outpath:Path) -> Path:
    """
    Fetch an RSS feed, parse episode metadata, and write it to a JSON file.

    Args:
        rss_url: URL of the RSS feed to fetch.
        outpath: Destination path for the parsed JSON metadata.

    Returns:
        Path to the written JSON metadata file.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    meta = []
    categories = []
    namespaces = {}

    xml = requests.get(rss_url).text

    for event, elem in ET.iterparse(io.StringIO(xml), events=('start-ns',)):
        prefix, uri = elem
        namespaces[prefix] = uri
    
    # itunes_namespace = namespaces


    root = ET.fromstring(xml)
    name_of_podcast = root.findtext('channel/title')
    language = root.findtext('channel/language')

    
    for cat in root.findall('channel/itunes:category', namespaces=namespaces):
        main_cat = cat.get('text')
        subcats = [sub.get('text') for sub in cat.findall('itunes:category', namespaces=namespaces)]
        categories.append({
            'topic': main_cat,
            'subtopic': subcats
        })

    for i, item in enumerate(root.findall('channel/item')):
        ep_name = item.findtext('title')
        media_url = item.find('enclosure')
        duration = item.findtext('itunes:duration', namespaces=namespaces)
        pubDate = item.findtext('pubDate')
        
        meta.append(
            {'name_of_podcast': name_of_podcast, 
            'categories': categories,
            'language': language,
            'ep_name': ep_name, 
            'pub_date': pubDate,
            'duration': duration,
            'media_url': media_url.get('url')}
    )
    
    with open(outpath, 'w', encoding='utf-8') as f_out:
        json.dump(meta, f_out)
        
    return outpath



def fix_episode_name(ep_name: str) -> str:
    """
    Make an episode name safe for filesystem usage.

    Args:
        ep_name: The original episode name from the RSS feed.

    Returns:
        String suitable for filenames.
    """
    return ep_name.replace("/", "|")


def make_queue(rss_path: Path, media_directory: Path) -> List[Dict[str, Any]]:
    """
    Build a JSON-serializable list of RSS episodes that are missing from the media directory.

    Args:
        rss_path: Path to the JSON file produced by `fetch_rss_feed`.
        media_directory: Folder holding downloaded media/transcripts; existing filenames (stems)
            are compared against RSS `ep_name` values to find gaps.

    Returns:
        List[Dict[str, Any]] for each missing episode with all metadata needed downstream.
    """

    rss_path = Path(rss_path)
    media_directory = Path(media_directory)
    media_directory.mkdir(exist_ok=True)

    directory_list = os.listdir(media_directory)
    directory_list_names = {fix_episode_name(Path(i).stem) for i in directory_list}

    rss_df = pd.read_json(rss_path)
    rss_df["safe_ep_name"] = rss_df["ep_name"].apply(fix_episode_name)

    to_download_mask = ~rss_df["safe_ep_name"].isin(directory_list_names)
    rss_to_download = rss_df[to_download_mask]

    # Convert the filtered dataframe to plain Python dicts so the queue is JSON serializable.
    download_queue = rss_to_download.to_dict(orient='records')

    print(f"Total number of files left to process: {len(download_queue)}")
    return download_queue



def download_media_file(media_file: dict, media_directory: Path) -> None:
    """
    Download an episode's audio to the media directory using the raw `ep_name` as the stem.

    Args:
        media_file: Episode metadata containing `ep_name` and `media_url`.
        media_directory: Target directory for the resulting mp3 file.

    Returns:
        None. Logs failures and raises on invalid filenames or download errors.
    """

    media_directory = Path(media_directory)
    media_directory.mkdir(parents=True, exist_ok=True)

    ep_name = media_file["ep_name"]
    safe_ep_name = media_file.get("safe_ep_name") or fix_episode_name(ep_name)
    media_url = media_file["media_url"]

    try:
        audiofile = media_directory / f"{safe_ep_name}.mp3"
    except Exception as exc:  # noqa: BLE001
        logging.exception("Invalid episode name for filesystem: %s", ep_name)
        raise ValueError(f"Invalid episode name for filesystem: {ep_name}") from exc

    try:
        with requests.get(media_url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            with audiofile.open('wb') as f_out, tqdm(total=total, unit='B', unit_scale=True, desc=str(audiofile)) as bar:
                for chunk in r.iter_content(1024 * 1024):
                    f_out.write(chunk)
                    bar.update(len(chunk))
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to download %s from %s", ep_name, media_url)
        raise

    if audiofile.exists() and audiofile.stat().st_size > 0:
        print(f"Download successful: {audiofile}.")
    else:
        logging.warning("Download not successful or zero size: %s", audiofile)
        print(f"Download not successful: {audiofile}")

def format_time(seconds: float) -> str:
    """
    Convert seconds into HH:MM:SS for transcript timestamps.

    Args:
        seconds: Floating-point duration returned by the transcription model.

    Returns:
        Timestamp string padded to hours:minutes:seconds.
    """
    sec = int(seconds)
    hour, remainder = divmod(sec, 3600)
    min, sec = divmod(remainder, 60)
    return f"{hour:02d}:{min:02d}:{sec:02d}"


def transcribe_audio_file(media_directory: Path) -> str:
    """
    Transcribe every mp3 file in the target directory with Faster-Whisper and write .txt outputs.

    Args:
        media_directory: Folder containing the mp3 files to process.

    Returns:
        'SUCCESS' after all detected files are transcribed.
    """
    media_directory = Path(media_directory)
    media_directory.mkdir(parents=True, exist_ok=True)
    audiofile = sorted(media_directory.glob("*.mp3"))

    if not audiofile:
        print(f"No audio files found in {media_directory}.")

    for f in audiofile:
        segments, info = model.transcribe(str(f), vad_filter=True)
        name_of_episode = f.stem

        transcript_path = media_directory / f"{name_of_episode}.txt"

        with transcript_path.open('w', encoding='utf-8') as f_out, \
        tqdm(total=float(info.duration), unit='s', desc=f"Transcribing: {name_of_episode}") as bar:
            for s in segments:
                line = f"({format_time(s.start)}) {s.text.strip()}"
                f_out.write(line + '\n')
                f_out.flush()  # ensures that data is physically written to disk and not held in memory

                bar.n = min(s.end, info.duration)
                bar.refresh()
            bar.n = bar.total
            bar.refresh()

    return 'SUCCESS'


def fetch_episode_metadata(conn: Connection, episode_name: str) -> Dict[str, str]:
    """
    Fetch RSS metadata for a given episode name; raises if the record is missing.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, ep_name, name_of_podcast
            FROM rss
            WHERE ep_name = %s;
            """,
            (episode_name,),
        )
        row = cur.fetchone()

    if row is None:
        raise ValueError(f"No RSS row found with name: {episode_name}")
    return row

def insert_transcript(
    conn: Connection,
    table_name: str,
    rss_id: int,
    name_of_podcast: str,
    ep_name: str,
    transcript: str,
):
    """
    Insert a transcript into the target table if a row with the same `rss_id` does not exist.
    """
    with conn.cursor() as cur:
        query = sql.SQL(
            """
            INSERT INTO {table} (
                rss_id, name_of_podcast, ep_name, transcript
            )
            SELECT %s, %s, %s, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM {table} WHERE rss_id = %s
            );
            """
        ).format(table=sql.Identifier(table_name))
        cur.execute(
            query,
            (rss_id, name_of_podcast, ep_name, transcript, rss_id)
        )
        if cur.rowcount:
            print('Inserted new transcript for episode:', ep_name)
        else:
            print('Duplicate entry detected; skipped insert.')

def ingest(
    media_directory: Path,
    transcript_table: str, 
    database_url: str,
    limit: Optional[int] = None,
    skip_postgres: bool = False,
    keep_transcripts: bool = False,
    rss_path: Optional[Path] = None,
) -> str:
    """
    Download pending RSS episodes, transcribe them, and optionally write transcripts to Postgres.

    Args:
        media_directory: Directory where audio files and transcripts are stored.
        transcript_table: Postgres table name for transcript storage.
        database_url: Connection string for Postgres.
        limit: Optional maximum number of new episodes to process this run.
        skip_postgres: If True, leave transcripts on disk and skip database writes.
        keep_transcripts: If False, delete transcript files after a successful Postgres insert.
        rss_path: Optional path to an existing RSS JSON; if absent a fresh copy is fetched.

    Returns:
        str: The function prints counts of processed transcripts and counts of transcripts sent to Postgres.
    """
    rss_url = "https://feeds.megaphone.fm/hubermanlab"

    if rss_path is not None:
        rss_path = Path(rss_path)
        if not rss_path.exists():
            logging.info("RSS path %s missing; downloading fresh feed.", rss_path)
            rss_path = fetch_rss_feed(rss_url=rss_url, outpath=rss_path)
    else:
        default_rss = Path(f"rss-{datetime.now().strftime('%Y-%m-%d')}.json")
        rss_path = fetch_rss_feed(rss_url=rss_url, outpath=default_rss)

    media_directory = Path(media_directory)
    media_directory.mkdir(parents=True, exist_ok=True)

    download_queue = make_queue(rss_path, media_directory)

    if limit is not None:
        download_queue = download_queue[:limit]
        print(f"Remaining number of files to process in this run: {len(download_queue)}")

    if not download_queue:
        print("No new episodes to ingest.")
     
    downloaded_transcribed_count = 0
    postgres_count = 0
    skipped = []
    conn = None
    if not skip_postgres:
        conn = psycopg.connect(database_url, autocommit=False)

    for episode in download_queue:
        ep_name = episode.get("ep_name", "")
        now_ts = datetime.now(timezone.utc).isoformat()
        if not ep_name:
            logging.warning("Skipping episode with missing name.")
            skipped.append({"ep_name": ep_name, "error": "missing name", "timestamp": now_ts})
            continue

        safe_ep_name = episode.get("safe_ep_name") or fix_episode_name(ep_name)

        try:
            audio_file_path = media_directory / f"{safe_ep_name}.mp3"
            transcript_path = media_directory / f"{safe_ep_name}.txt"

            try:
                download_media_file(media_file=episode, media_directory=media_directory)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Download failed for %s; skipping episode.", ep_name)
                skipped.append({"ep_name": ep_name, "error": f"download failed: {exc}", "timestamp": now_ts})
                continue

            transcribe_audio_file(media_directory=media_directory)

            if transcript_path.exists() and transcript_path.stat().st_size > 0 and audio_file_path.exists():
                audio_file_path.unlink()
                downloaded_transcribed_count +=1
                print(f"Number of files processed in current run: {downloaded_transcribed_count}/{len(download_queue)}")
                print(f'Deleted: {audio_file_path}')
            
            if skip_postgres:
                print("Skip-postgres enabled; leaving transcript on disk only.")
            else:
                print(f"Preparing to write transcript to postgres: {transcript_path}")
                metadata = fetch_episode_metadata(conn, ep_name)
                postgres_count +=1
                
                with transcript_path.open("r", encoding="utf-8") as f_in:
                        transcript_text = f_in.read()
                insert_transcript(
                            conn=conn,
                            table_name=transcript_table,
                            rss_id=metadata["id"],
                            name_of_podcast=metadata["name_of_podcast"],
                            ep_name=metadata["ep_name"],
                            transcript=transcript_text,
                        )
                conn.commit()
                if not keep_transcripts:
                    try:
                        transcript_path.unlink(missing_ok=True)
                        print(f"Removed local transcript: {transcript_path}")
                    except OSError as cleanup_error:
                        logging.warning("Failed to remove %s: %s", transcript_path, cleanup_error)
                
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Skipping episode due to error: %s", ep_name)
            skipped.append({"ep_name": ep_name, "error": str(exc), "timestamp": now_ts})
            continue
    if conn:
        conn.close()
    if skipped:
        logging.warning("Skipped %d episodes due to errors.", len(skipped))
        for item in skipped:
            logging.warning("Skipped: %s error: %s", item.get("ep_name", ""), item.get("error", ""))
    try:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "skipped_episodes.json"
        with log_file.open("w", encoding="utf-8") as f_out:
            json.dump(skipped, f_out, indent=2)
        logging.info("Wrote skipped episodes to %s", log_file)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to write skipped episodes log: %s", exc)
    return print(f"Total transcribed:{downloaded_transcribed_count}; Total written to postgres:{postgres_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Download new podcast episodes and generate transcripts.',
    )
    parser.add_argument(
        '--rss-path',
        type=Path,
        default=None,
        help='Path to the RSS JSON file containing episode metadata.',
    )
    parser.add_argument(
        '--media-dir',
        type=Path,
        required=True,
        help='Directory where media files and transcripts are stored.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of episodes to process (default: no limit).',
    )
    parser.add_argument(
        '--transcript-table',
        type=str,
        default="transcripts",
        help='Postgres table name where transcripts will be stored.',
    )
    parser.add_argument(
        '--database-url',
        type=str,
        default="postgresql://podcast:podcast@localhost:5434/podcast-agent",
        help='Postgres connection string for storing transcripts.',
    )
    parser.add_argument(
        '--skip-postgres',
        action='store_true',
        help='Skip writing transcripts to Postgres; leave transcript files on disk only.',
    )
    parser.add_argument(
        '--keep-transcripts',
        action='store_true',
        help='Keep transcript .txt files on disk even after writing to Postgres (default: delete once stored).',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ingest(
        rss_path=args.rss_path,
        media_directory=args.media_dir,
        transcript_table=args.transcript_table,
        database_url=args.database_url,
        limit=args.limit,
        skip_postgres=args.skip_postgres,
        keep_transcripts=args.keep_transcripts or args.skip_postgres,
    )


if __name__ == '__main__':
    main()
