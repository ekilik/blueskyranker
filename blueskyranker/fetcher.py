#!/usr/bin/env python3
"""
Bluesky Author Feed Scraper (via public AppView using `atproto`).

Responsibilities:
- Fetch PUBLIC posts for one or more Bluesky handles with engagement metrics and
  embedded news link metadata.
- Normalise timestamps: store canonical UTC TEXT in `createdAt` and an epoch nanosecond
  integer in `createdAt_ns` for robust sorting and window filtering.
- Upsert into SQLite by URI (refresh engagement on repeats) and optionally export CSVs.

CLI examples live in the README; this module is used programmatically by the pipeline.
"""

import csv
import os
import argparse
import time
import sqlite3
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from atproto import Client, models
from tqdm import tqdm

import logging

logger = logging.getLogger('BSRlog')

APPVIEW_XRPC = "https://public.api.bsky.app/xrpc"
PAGE_LIMIT = 100
SLEEP_SEC = 0.20  # polite pacing

# Canonical post fields used for exports and SQLite
CSV_HEADERS = [
    "uri","cid","author_handle","author_did","indexedAt","createdAt","text",
    "reply_root_uri","reply_parent_uri","is_repost",
    "like_count","repost_count","reply_count","quote_count",
    "news_title","news_description","news_uri","news_content", "news_actors" 
]

# SQLite helpers
def ensure_db(path: str) -> sqlite3.Connection:
    """Open or create the SQLite DB with required schema."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            uri TEXT PRIMARY KEY,
            cid TEXT,
            author_handle TEXT,
            author_did TEXT,
            indexedAt TEXT,
            createdAt TEXT,
            createdAt_ns INTEGER,
            text TEXT,
            reply_root_uri TEXT,
            reply_parent_uri TEXT,
            is_repost INTEGER,
            like_count INTEGER,
            repost_count INTEGER,
            reply_count INTEGER,
            quote_count INTEGER,
            news_title TEXT,
            news_description TEXT,
            news_uri TEXT,
            news_content TEXT,
            news_actors TEXT
        )
        """
    )
    # Pragmas for better concurrent reads and reasonable durability
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        logger.debug("PRAGMA setup failed; continuing with defaults")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_handle)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(createdAt)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_created_ns ON posts(createdAt_ns)")
    return conn

def upsert_rows(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> int:
    """Upsert rows by uri. Returns number of rows affected (inserted or updated)."""
    if not rows:
        return 0
    # Normalize embed fields: store NULL for empty strings
    def _norm(r: Dict[str, Any]) -> Dict[str, Any]:
        for k in ("news_title", "news_description", "news_uri", "news_content", "news_actors"):
            v = r.get(k)
            if isinstance(v, str) and v.strip() == "":
                r[k] = None
        return r
    rows = [_norm(dict(r)) for r in rows]
    # Accept optional createdAt_ns if present in rows
    cols = list(CSV_HEADERS)
    if rows and 'createdAt_ns' in rows[0]:
        cols = cols.copy() + ['createdAt_ns']
    placeholders = ",".join([":"+c for c in cols])
    update_clause = ",".join([f"{c}=excluded.{c}" for c in cols if c != "uri"])  # leave PK as-is
    sql = f"""
        INSERT INTO posts ({','.join(cols)}) VALUES ({placeholders})
        ON CONFLICT(uri) DO UPDATE SET {update_clause}
    """
    with conn:
        conn.executemany(sql, rows)
    return len(rows)

def latest_created_at_in_db(conn: sqlite3.Connection, handle: str) -> Optional[datetime]:
    """Return latest createdAt based on createdAt_ns when available, else fallback to createdAt.

    Uses UTC-aware datetimes for safe comparison downstream.
    """
    # Prefer high-precision epoch nanoseconds
    cur = conn.execute("SELECT MAX(createdAt_ns) FROM posts WHERE author_handle=?", (handle,))
    row = cur.fetchone()
    if row and row[0]:
        try:
            ns = int(row[0])
            return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)
        except Exception:
            pass
    # Fallback to canonical ISO UTC string
    cur = conn.execute("SELECT MAX(createdAt) FROM posts WHERE author_handle=?", (handle,))
    row = cur.fetchone()
    if not row:
        return None
    max_created = row[0]
    return iso_to_dt(max_created) if max_created else None

def export_db_to_csv(conn: sqlite3.Connection, output_dir: str = ".", include_combined: bool = True) -> List[str]:
    """Export posts per handle to CSV files and optionally a combined CSV.

    Returns list of written file paths.
    """
    out_paths: List[str] = []
    cur = conn.execute("SELECT DISTINCT author_handle FROM posts ORDER BY author_handle")
    handles = [r[0] for r in cur.fetchall() if r[0]]
    combined_rows: List[Dict[str, Any]] = []
    for h in handles:
        rows_cur = conn.execute("SELECT * FROM posts WHERE author_handle=? ORDER BY createdAt", (h,))
        rows = [dict(zip([c[0] for c in rows_cur.description], r)) for r in rows_cur.fetchall()]
        # Normalize to CSV headers order
        rows_csv = [{k: r.get(k) for k in CSV_HEADERS} for r in rows]
        path = os.path.join(output_dir, f"{h.replace('.', '_')}_author_feed.csv")
        ensure_headers(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            for row in rows_csv:
                w.writerow(row)
        out_paths.append(path)
        combined_rows.extend(rows_csv)

    if include_combined and combined_rows:
        cpath = os.path.join(output_dir, "all_handles_author_feed.csv")
        ensure_headers(cpath)
        with open(cpath, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            for row in combined_rows:
                w.writerow(row)
        out_paths.append(cpath)
    return out_paths


def import_csvs_to_db(conn: sqlite3.Connection, csv_paths: List[str]) -> int:
    """Import existing per-handle CSVs into SQLite (upsert by uri).

    Returns number of rows processed across all CSVs.
    """
    total = 0
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            batch: List[Dict[str, Any]] = []
            for row in reader:
                batch.append({h: row.get(h) for h in CSV_HEADERS})
                if len(batch) >= 1000:
                    upsert_rows(conn, batch)
                    total += len(batch)
                    batch = []
            if batch:
                upsert_rows(conn, batch)
                total += len(batch)
    return total


def load_posts_df(conn: sqlite3.Connection, handle: Optional[str] = None, limit: Optional[int] = None,
                  order_by: str = "createdAt", descending: bool = False):
    """Return posts as a Polars DataFrame, optionally filtered by handle.

    Example: df = load_posts_df(conn, handle='news-flows-nl.bsky.social', limit=1000)
    """
    import polars as pl
    where = []
    args: List[Any] = []
    if handle:
        where.append("author_handle = ?")
        args.append(handle)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    order_sql = f" ORDER BY {order_by} {'DESC' if descending else 'ASC'}"
    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    # Include createdAt_ns (not in CSV headers) so ranker/pipeline can prefer it for robust time ops
    select_cols = CSV_HEADERS + ["createdAt_ns"]
    sql = f"SELECT {', '.join(select_cols)} FROM posts{where_sql}{order_sql}{limit_sql}"
    cur = conn.execute(sql, args)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return pl.from_dicts(rows)

def iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # handle trailing Z or timezone-less strings
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def ensure_headers(path: str) -> bool:
    """(Legacy) Create file with header if it doesn't exist. Return True if newly created."""
    if os.path.exists(path):
        return False
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()
    return True

def extract_news_embed(item_post: Any, record: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Try both hydrated embed (post.embed.external) and record embed (record.embed.external).
    Return (title, description, uri).
    """
    def safe_get_external(from_obj: Any):
        if not from_obj:
            return None
        # Hydrated view path: post.embed.external (AppBskyEmbedExternal.ViewExternal)
        ext = getattr(from_obj, "external", None)
        if ext and getattr(ext, "title", None) is not None:
            title = getattr(ext, "title", None)
            desc = getattr(ext, "description", None)
            uri  = getattr(ext, "uri", None) or getattr(ext, "url", None)
            return (title, desc, uri)
        return None

    # Try hydrated (view) embed on post
    view_embed = getattr(item_post, "embed", None)
    got = safe_get_external(view_embed)
    if got:
        return got

    # Try record embed
    rec_embed = getattr(record, "embed", None) if record else None
    got = safe_get_external(rec_embed)
    if got:
        return got

    # Some implementations keep external directly (rare)
    if view_embed and (getattr(view_embed, "title", None) and (getattr(view_embed, "uri", None) or getattr(view_embed, "url", None))):
        return (
            getattr(view_embed, "title", None),
            getattr(view_embed, "description", None),
            getattr(view_embed, "uri", None) or getattr(view_embed, "url", None),
        )

    return (None, None, None)

def extract_article_content(news_uri: str) -> Optional[str]:
    """Fetch the news_uri and extract main article text using newspaper4k.
    Returns the article text or None on failure.
    """
    import newspaper
    try:
        article = newspaper.Article(news_uri)
        article.download()
        article.parse()
        text = article.text
        return text if text.strip() else None
    except Exception as e:
        logger.debug(f"Failed to extract article content from {news_uri}: {e}")
        return None

def _build_prompt(system_prompt: str, main_prompt: str):
    """Create prompts in the format expected by Ollama"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": main_prompt
        }
    ]
    return messages

def extract_actors_from_content(news_content: str, ollama_model: str = 'gpt-oss:20b') -> Optional[str]:
    """Extract news actors from news content using Ollama."""
    import ollama

    if not news_content or len(news_content.strip()) < 50:
        return None    

    try:
        available_models = [m.model for m in ollama.list().models]
        if ollama_model not in available_models:
            print(f"Pulling Ollama model: {ollama_model}")
            ollama.pull(ollama_model)
        
    except Exception as e:
        logger.debug(f"Could not check/pull model: {e}")

    try:
        # reat the prompt from file
        prompt_file_path = os.path.join(os.path.dirname(__file__), 'actor_extraction_prompt.txt')
        if not os.path.exists(prompt_file_path):
            print(f"Prompt file not found: {prompt_file_path}")
            return None
        
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read()

        # Split the prompt content into system and main prompts
        if "SYSTEM_PROMPT:" not in prompt_content or "MAIN_PROMPT:" not in prompt_content:
            print("Prompt file is missing required sections.")
            return None
        
        else:
            parts = prompt_content.split("MAIN_PROMPT:")
            system_prompt = parts[0].replace("SYSTEM_PROMPT:", "").strip()
            main_prompt = parts[1].strip()

        prompt = _build_prompt(system_prompt=system_prompt, main_prompt=main_prompt + "\n" + news_content)
        # pull the model if not available

        response = ollama.chat(model=ollama_model, messages=prompt, options={'temperature': 0.0, 'seed': 0})
        response_content = response['message']['content'].strip()
        cleaned_response = _clean_and_parse_actors(response_content)
        
        return cleaned_response

    except Exception as e:
        logger.debug(f"Ollama actor extraction failed: {e}")
        return None

def _clean_and_parse_actors(llm_response: str) -> Optional[str]:
    """Clean LLM response and extract structured actor data.
    Returns string with list of actor dictionaries, or None if parsing fails.
    """
    
    try:
        # First try to find the actors array pattern
        # Look for "actors": [ ... ] or actors: [ ... ]
        actors_pattern = r'"?actors"?\s*:\s*\[(.*?)\]'
        actors_match = re.search(actors_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        if not actors_match:
            logger.debug("No actors array pattern found in LLM response")
            return None
        
        actors_content = actors_match.group(1).strip()
        if not actors_content:
            logger.debug("Empty actors array found")
            return None
        
        # Try to parse the actors content as JSON
        try:
            # Wrap in array brackets and try to parse
            actors_json = f"[{actors_content}]"
            actors_list = json.loads(actors_json)
        except json.JSONDecodeError:
            # If direct JSON parsing fails, try to extract individual actor objects
            logger.debug("Direct JSON parsing failed, trying pattern extraction")
            actors_list = _extract_actor_json_from_output(actors_content)
        
        if not actors_list or not isinstance(actors_list, list):
            return None
        
        # Clean and validate each actor
        cleaned_actors = []
        for actor in actors_list:
            if isinstance(actor, dict) and 'actor_name' in actor:
                cleaned_actor = {
                    'actor_name': str(actor.get('actor_name', '')).strip(),
                    'actor_function': str(actor.get('actor_function', '')).strip(),
                    'actor_pp': str(actor.get('actor_pp', '')).strip()
                }
                # Only add if actor has a name
                if cleaned_actor['actor_name']:
                    cleaned_actors.append(cleaned_actor)
        
        # Return clean JSON string
        return json.dumps({'actors': cleaned_actors}) if cleaned_actors else None
        
    except Exception as e:
        logger.debug(f"Failed to clean actor response: {e}")
        return None


def _extract_actor_json_from_output(actors_text: str) -> List[Dict[str, str]]:
    """Extract actor dictionaries from text using pattern matching.
    Fallback when JSON parsing fails.
    """
    import re
    actors = []
    
    # Look for individual actor objects like {"actor_name": "...", "actor_function": "...", "actor_pp": "..."}
    actor_pattern = r'\{[^}]*"?actor_name"?\s*:\s*"([^"]*)"[^}]*"?actor_function"?\s*:\s*"([^"]*)"[^}]*"?actor_pp"?\s*:\s*"([^"]*)"\s*[^}]*\}'
    
    matches = re.findall(actor_pattern, actors_text, re.IGNORECASE)
    
    for match in matches:
        actor_name, actor_function, actor_pp = match
        if actor_name.strip():  # Only add if has a name
            actors.append({
                'actor_name': actor_name.strip(),
                'actor_function': actor_function.strip(),
                'actor_pp': actor_pp.strip()
            })
    
    # If structured pattern fails, try simpler line-by-line extraction
    if not actors:
        actors = _extract_actor_info_from_text(actors_text)
    
    return actors


def _extract_actor_info_from_text(actors_text: str) -> List[Dict[str, str]]:
    """Extract actors from text that might be in a simpler format.
    Ultimate fallback for badly formatted LLM responses.
    """
    
    actors = []
    lines = actors_text.split('\n')
    
    current_actor = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for key-value patterns
        name_match = re.search(r'"?actor_name"?\s*:\s*"?([^",\n]+)"?', line, re.IGNORECASE)
        function_match = re.search(r'"?actor_function"?\s*:\s*"?([^",\n]+)"?', line, re.IGNORECASE)
        pp_match = re.search(r'"?actor_pp"?\s*:\s*"?([^",\n]*)"?', line, re.IGNORECASE)
        
        if name_match:
            # If we have a current actor, save it
            if current_actor.get('actor_name'):
                actors.append(current_actor)
            # Start new actor
            current_actor = {
                'actor_name': name_match.group(1).strip().strip('"'),
                'actor_function': '',
                'actor_pp': ''
            }
        
        if function_match and current_actor:
            current_actor['actor_function'] = function_match.group(1).strip().strip('"')
        
        if pp_match and current_actor:
            current_actor['actor_pp'] = pp_match.group(1).strip().strip('"')
    
    # Don't forget the last actor
    if current_actor.get('actor_name'):
        actors.append(current_actor)
    
    return actors

def _parse_and_normalise_created_at(created_at: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """Parse a source timestamp into canonical UTC string and epoch nanoseconds.

    - Accepts either trailing 'Z' or explicit offsets like '+00:00'.
    - Returns (iso_utc, epoch_ns). If parsing fails, returns (original_string, None).
    """
    if not created_at:
        return None, None
    s = str(created_at)
    # Normalise: replace trailing 'Z' with '+00:00' for fromisoformat
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return created_at, None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    # Canonical string with microseconds and Z
    iso = dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    ns = int(dt.timestamp() * 1_000_000_000)
    return iso, ns


def flatten_item(item: models.AppBskyFeedDefs.FeedViewPost, extract_articles: bool = False, extract_actors: bool = False, ollama_model: Optional[str] = None) -> Dict[str, Any]:    
    # Set default model if actor extraction is enabled but no model specified
    if extract_actors and ollama_model is None:
        ollama_model = 'gpt-oss:20b'
        
    post = item.post
    record = getattr(post, "record", None)
    author = getattr(post, "author", None)
    reply = getattr(item, "reply", None)
    reason = getattr(item, "reason", None)

    def get_uri(x):
        return getattr(x, "uri", None) if x else None

    reply_root_uri = get_uri(getattr(reply, "root", None)) if reply else None
    reply_parent_uri = get_uri(getattr(reply, "parent", None)) if reply else None

    news_title, news_description, news_uri = extract_news_embed(post, record)

    if extract_articles and news_uri:    
        news_content = extract_article_content(news_uri)
        news_actors = extract_actors_from_content(news_content, ollama_model) if (extract_actors and news_content) else None
    else:
        news_content = None
        news_actors = None

    # Normalise createdAt and compute epoch ns
    created_iso, created_ns = _parse_and_normalise_created_at(getattr(record, "created_at", None) if record else None)

    return {
        "uri": getattr(post, "uri", None),
        "cid": getattr(post, "cid", None),
        "author_handle": getattr(author, "handle", None) if author else None,
        "author_did": getattr(author, "did", None) if author else None,
        "indexedAt": getattr(post, "indexed_at", None),
        "createdAt": created_iso,
        "createdAt_ns": created_ns,
        "text": getattr(record, "text", None) if record else None,
        "reply_root_uri": reply_root_uri,
        "reply_parent_uri": reply_parent_uri,
        "is_repost": 1 if (reason and getattr(reason, "$type", "") == "app.bsky.feed.defs#reasonRepost") else 0,
        "like_count": getattr(post, "like_count", 0) or 0,
        "repost_count": getattr(post, "repost_count", 0) or 0,
        "reply_count": getattr(post, "reply_count", 0) or 0,
        "quote_count": getattr(post, "quote_count", 0) or 0,
        "news_title": news_title,
        "news_description": news_description,
        "news_uri": news_uri,
        "news_content": news_content,
        "news_actors": news_actors
    }

def fetch_author_feed_all(
    client: Client,
    actor: str,
    posts_pbar: tqdm,
    include_pins: bool,
    cutoff_dt: Optional[datetime],
    cutoff_check_every: int = 1,
    extract_articles: bool = False, 
    extract_actors: bool = False,
    ollama_model: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch all pages for one handle, honoring cutoff_dt (createdAt >= cutoff_dt).
    Update global posts progress bar. Return (flattened_rows, stats_dict).

    Early-stop decision is evaluated every `cutoff_check_every` pages. Rows outside
    the cutoff are never written.
    """
    handle_start = time.perf_counter()
    pages = 0
    total_posts = 0
    rows: List[Dict[str, Any]] = []

    originals = 0
    reposts = 0
    replies = 0
    sum_likes = 0
    sum_reposts = 0
    sum_replies = 0
    sum_quotes = 0
    first_dt = None
    last_dt = None
    embed_empty_title = 0
    embed_empty_desc = 0
    embed_empty_uri = 0

    cursor: Optional[str] = None
    page_bar = tqdm(desc=f"[{actor}] pages", unit="page", dynamic_ncols=True, leave=False)

    def within_window(created_at: Optional[str]) -> bool:
        if cutoff_dt is None:
            return True
        dt = iso_to_dt(created_at)
        return (dt is not None) and (dt >= cutoff_dt)

    # Retry configuration (exponential backoff with jitter)
    max_retries = 3
    base_delay = 0.5

    while True:
        params = models.AppBskyFeedGetAuthorFeed.Params(
            actor=actor, limit=PAGE_LIMIT, cursor=cursor, include_pins=include_pins
        )
        resp = None
        for attempt in range(max_retries):
            try:
                resp = client.app.bsky.feed.get_author_feed(params)
                break
            except Exception as e:
                delay = base_delay * (2 ** attempt)
                # small jitter to avoid thundering herd
                jitter = 0.05 * delay
                logger.warning(
                    f"get_author_feed failed for {actor} (cursor={cursor}) attempt {attempt+1}/{max_retries}: {e}"
                )
                time.sleep(delay + jitter)
        if resp is None:
            # Exhausted retries for this handle/cursor
            logger.error(f"Aborting fetch for {actor} after {max_retries} failed attempts")
            break
        feed = resp.feed or []

        # If we got nothing, stop
        if not feed:
            break

        # Track last seen to show in bar
        last_created_seen = None

        page_keep: List[models.AppBskyFeedDefs.FeedViewPost] = []
        for item in feed:
            created_at = getattr(getattr(item.post, "record", None), "created_at", None)
            last_created_seen = created_at or last_created_seen
            if within_window(created_at):
                page_keep.append(item)

        # Always keep in-window posts on this page
        for item in page_keep:
            flat = flatten_item(item, extract_articles, extract_actors, ollama_model)  
            # Track empty-string anomalies in embed fields
            if isinstance(flat.get("news_title"), str) and flat["news_title"].strip() == "":
                embed_empty_title += 1
            if isinstance(flat.get("news_description"), str) and flat["news_description"].strip() == "":
                embed_empty_desc += 1
            if isinstance(flat.get("news_uri"), str) and flat["news_uri"].strip() == "":
                embed_empty_uri += 1
            

            rows.append(flat)
            total_posts += 1
            posts_pbar.update(1)

            sum_likes += flat["like_count"]
            sum_reposts += flat["repost_count"]
            sum_replies += flat["reply_count"]
            sum_quotes += flat["quote_count"]

            if flat["is_repost"]:
                reposts += 1
            else:
                if flat["reply_parent_uri"]:
                    replies += 1
                else:
                    originals += 1

            dt = iso_to_dt(flat["createdAt"])
            if dt:
                if not first_dt or dt < first_dt:
                    first_dt = dt
                if not last_dt or dt > last_dt:
                    last_dt = dt

        cursor = resp.cursor
        pages += 1

        # Page bar description with last seen timestamp and window status
        status_suffix = ""
        if last_created_seen:
            dt_seen = iso_to_dt(last_created_seen)
            if dt_seen:
                w = (cutoff_dt is None) or (dt_seen >= cutoff_dt)
                status_suffix = f" — last: {dt_seen.isoformat()} — within window: {'yes' if w else 'no'}"
        page_bar.set_description(f"[{actor}] pages{status_suffix}")
        page_bar.update(1)

        # Early-stop decision: if nothing on the page was in-window AND it's time to check
        all_outside = (cutoff_dt is not None and not page_keep)
        should_check_now = (pages % max(1, cutoff_check_every) == 0)

        if all_outside and should_check_now:
            break

        if not cursor:
            break
        time.sleep(SLEEP_SEC)

    page_bar.close()
    elapsed = time.perf_counter() - handle_start
    rate = total_posts / elapsed if elapsed > 0 else 0.0

    stats = {
        "handle": actor,
        "pages": pages,
        "posts": total_posts,
        "originals": originals,
        "reposts": reposts,
        "replies": replies,
        "sum_likes": sum_likes,
        "sum_reposts": sum_reposts,
        "sum_replies": sum_replies,
        "sum_quotes": sum_quotes,
        "avg_likes": (sum_likes / total_posts) if total_posts else 0.0,
        "avg_reposts": (sum_reposts / total_posts) if total_posts else 0.0,
        "avg_replies": (sum_replies / total_posts) if total_posts else 0.0,
        "avg_quotes": (sum_quotes / total_posts) if total_posts else 0.0,
        "first_post": dt_to_iso(first_dt),
        "last_post": dt_to_iso(last_dt),
        "elapsed_sec": elapsed,
        "rate_posts_per_sec": rate,
        # Embed anomaly flags (empty strings should not occur; indicates upstream issue)
        "embed_empty_title": embed_empty_title,
        "embed_empty_description": embed_empty_desc,
        "embed_empty_uri": embed_empty_uri,
    }
    return rows, stats

def final_report(per_handle_stats: List[Dict[str, Any]]) -> str:
    # Aggregate totals
    lines = []
    grand = defaultdict(float)
    for s in per_handle_stats:
        for k, v in s.items():
            if k in {"handle","first_post","last_post"}:
                continue
            if isinstance(v, (int, float)):
                grand[k] += v

    lines.append("\n" + "="*72)
    lines.append("FINAL REPORT")
    lines.append("="*72)
    for s in per_handle_stats:
        lines.append(f"\nHandle: {s['handle']}")
        lines.append(f"  Pages fetched         : {s['pages']}")
        lines.append(f"  Posts fetched         : {s['posts']}")
        lines.append(f"    - originals         : {s['originals']}")
        lines.append(f"    - replies           : {s['replies']}")
        lines.append(f"    - reposts           : {s['reposts']}")
        lines.append(f"  Engagement (sums)")
        lines.append(f"    - likes             : {s['sum_likes']}")
        lines.append(f"    - reposts           : {s['sum_reposts']}")
        lines.append(f"    - replies           : {s['sum_replies']}")
        lines.append(f"    - quotes            : {s['sum_quotes']}")
        lines.append(f"  Engagement (averages per post)")
        lines.append(f"    - likes             : {s['avg_likes']:.2f}")
        lines.append(f"    - reposts           : {s['avg_reposts']:.2f}")
        lines.append(f"    - replies           : {s['avg_replies']:.2f}")
        lines.append(f"    - quotes            : {s['avg_quotes']:.2f}")
        lines.append(f"  Time range            : {s['first_post']}  →  {s['last_post']}")
        lines.append(f"  Time taken            : {s['elapsed_sec']:.2f}s")
        lines.append(f"  Effective rate        : {s['rate_posts_per_sec']:.2f} posts/sec")
        # Flag anomalies from upstream embed extraction scripts
        if s.get('embed_empty_title') or s.get('embed_empty_description') or s.get('embed_empty_uri'):
            lines.append("  WARN embed anomalies  :")
            lines.append(f"    - empty news_title  : {s.get('embed_empty_title', 0)}")
            lines.append(f"    - empty news_descr. : {s.get('embed_empty_description', 0)}")
            lines.append(f"    - empty news_uri    : {s.get('embed_empty_uri', 0)}")

    total_posts = int(grand["posts"])
    lines.append("\n" + "-"*72)
    lines.append("All handles combined")
    lines.append("-"*72)
    lines.append(f"  Total pages           : {int(grand['pages'])}")
    lines.append(f"  Total posts           : {total_posts}")
    lines.append(f"    - originals         : {int(grand['originals'])}")
    lines.append(f"    - replies           : {int(grand['replies'])}")
    lines.append(f"    - reposts           : {int(grand['reposts'])}")
    lines.append(f"  Engagement (sums)")
    lines.append(f"    - likes             : {int(grand['sum_likes'])}")
    lines.append(f"    - reposts           : {int(grand['sum_reposts'])}")
    lines.append(f"    - replies           : {int(grand['sum_replies'])}")
    lines.append(f"    - quotes            : {int(grand['sum_quotes'])}")
    if total_posts:
        lines.append(f"  Engagement (averages per post)")
        lines.append(f"    - likes             : {grand['sum_likes']/total_posts:.2f}")
        lines.append(f"    - reposts           : {grand['sum_reposts']/total_posts:.2f}")
        lines.append(f"    - replies           : {grand['sum_replies']/total_posts:.2f}")
        lines.append(f"    - quotes            : {grand['sum_quotes']/total_posts:.2f}")
    
    return "\n".join(lines)

class Fetcher():
    """Fetcher to get public Bluesky posts with max-age cutoff, incremental updates, and embed extraction.

    Supports writing to CSV (append + de-dup) or SQLite (upsert by uri).
    """
    def __init__(self, xrpc_base=APPVIEW_XRPC):
        self.client = Client(xrpc_base)

    def fetch(self, handles: Optional[List[str]] = None,
        max_age_days=7,
        cutoff_check_every=1,
        include_pins=False,
        sqlite_path: Optional[str] = None,
        refresh_window: bool = False,
        extract_articles: bool = False,
        extract_actors: bool = False,
        ollama_model: Optional[str] = None
        ):
        if handles is None:
            handles = [
                "news-flows-nl.bsky.social",
                "news-flows-ir.bsky.social",
                "news-flows-cz.bsky.social",
                "news-flows-fr.bsky.social",
            ]
        
        # Validate parameter dependencies
        if extract_actors and not extract_articles:
            raise ValueError("extract_actors=True requires extract_articles=True")
        
        if extract_actors and ollama_model is None:
            ollama_model = 'gpt-oss:20b'  

        """Fetch public Bluesky posts (SQLite only).

        Parameters:

        handles (list of strings): Bluesky handles to fetch
        max_age_days (int): Only keep posts whose createdAt is within the last N days.
        cutoff_check_every (int)": How many pages between early-stop checks against the cutoff (default: 1 = check every page).
        include_pins (bool): Include pinned posts at the top
        sqlite_path (str|None): path to sqlite DB (default: ./newsflows.db)
        refresh_window (bool): If True, re-fetch the entire N-day window even if posts already exist in DB (refresh engagement metrics). If False (default), fetch incrementally from the latest saved timestamp.
        extract_articles (bool): If True, extract full article text from news URLs (default: False). This significantly slows down fetching.
        extract_actors (bool): If True, extract actors from news articles using Ollama (requires extract_articles=True). This significantly slows down fetching.
        ollama_model (str|None): Ollama model name for actor extraction. Only required when extract_actors=True. Defaults to 'gpt-oss:20b' when needed.
        """

        db_path = sqlite_path or "newsflows.db"
        conn: sqlite3.Connection = ensure_db(db_path)

        posts_pbar = tqdm(desc="Posts fetched (all handles)", unit="post", dynamic_ncols=True)
        combined_new_rows: List[Dict[str, Any]] = []
        per_handle_stats: List[Dict[str, Any]] = []

        handles_pbar = tqdm(handles, desc="Handles", unit="handle", dynamic_ncols=True)
        start_all = time.perf_counter()

        for handle in handles_pbar:
            out_path = f"{handle.replace('.', '_')}_author_feed.csv"  # legacy filename for status messages only

            # Choose cutoff per handle
            cutoff_dt: Optional[datetime] = None
            if max_age_days is not None:
                rel_cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
                if refresh_window:
                    # Refresh the entire window N days back
                    cutoff_dt = rel_cutoff
                else:
                    # Incremental: only fetch newer than what we have
                    latest_saved = latest_created_at_in_db(conn, handle)
                    cutoff_dt = max(latest_saved, rel_cutoff) if latest_saved else rel_cutoff

            # Fetch rows for the handle
            handle_rows, stats = fetch_author_feed_all(
                client=self.client,
                actor=handle,
                posts_pbar=posts_pbar,
                include_pins=include_pins,
                cutoff_dt=cutoff_dt,
                cutoff_check_every=cutoff_check_every,
                extract_articles=extract_articles, 
                extract_actors=extract_actors,
                ollama_model=ollama_model  
            )

            # Upsert all rows for this handle
            affected = upsert_rows(conn, handle_rows)
            combined_new_rows.extend(handle_rows)

            # Report: how many NEW and their time frame
            handles_pbar.write(
                f"✅ DONE {handle}: upserted {len(handle_rows)} posts into SQLite"
            )

            # Keep scrape stats (before de-dup)
            per_handle_stats.append(stats)

        elapsed_all = time.perf_counter() - start_all
        posts_pbar.close()

        logger.debug(
            f"\nSQLite upsert complete → {db_path} ({elapsed_all:.2f}s total)."
        )

        # Final detailed report for the run (scraped rows prior to de-dup)
        return final_report(per_handle_stats)

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description="Fetch public Bluesky posts with progress, time window cutoff, optional full-window refresh, and embed extraction (SQLite storage).")
    parser.add_argument(
        "--handles",
        nargs="+",
        default=[
            "news-flows-nl.bsky.social",
            "news-flows-ir.bsky.social",
            "news-flows-cz.bsky.social",
            "news-flows-fr.bsky.social",
        ],
        help="Bluesky handles to fetch (space-separated)."
    )
    parser.add_argument(
        "--xrpc-base",
        default=APPVIEW_XRPC,
        help="XRPC base URL (default: AppView public API)."
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="Only keep posts whose createdAt is within the last N days (window cutoff)."
    )
    parser.add_argument(
        "--cutoff-check-every",
        type=int,
        default=1,
        help="How many pages between early-stop checks against the cutoff (default: 1 = check every page)."
    )
    parser.add_argument(
        "--include-pins",
        dest="include_pins",
        action="store_true",
        help="Include pinned posts at the top (default: False)."
    )
    parser.add_argument(
        "--no-include-pins",
        dest="include_pins",
        action="store_false",
        help="Exclude pinned posts (default)."
    )
    parser.add_argument(
        "--sqlite-path",
        default="newsflows.db",
        help="Path to SQLite database file."
    )
    parser.add_argument(
        "--refresh-window",
        action="store_true",
        help="Re-fetch the entire N-day window (refresh engagement), ignoring latest saved timestamp."
    )
    parser.add_argument(
    "--extract-articles",
    action="store_true",
    help="Extract full article text from news URLs (probably will slow down fetching significantly)."
    )

    parser.add_argument(
    "--extract-actors",
    action="store_true",
    help="Extract actors/entities from news articles using Ollama (requires --extract-articles). Will slow down fetching significantly."
    )

    parser.add_argument(
    "--ollama-model",
    type=str,
    default=None,
    help="Ollama model to use for actor extraction (default: gpt-oss:20b when extract_actors is enabled)."
    )

    parser.set_defaults(include_pins=False)
    args = parser.parse_args()

    # Validate command-line argument dependencies
    if args.extract_actors and not args.extract_articles:
        parser.error("--extract-actors requires --extract-articles")

    fetcher = Fetcher(args.xrpc_base)
    results = fetcher.fetch(
    handles = args.handles,
    max_age_days=args.max_age_days,
    cutoff_check_every=args.cutoff_check_every,
    include_pins=args.include_pins,
    sqlite_path=args.sqlite_path,
    refresh_window=args.refresh_window,
    extract_articles=args.extract_articles, 
    extract_actors=args.extract_actors,
    ollama_model=args.ollama_model  
    )
    print(results)

   

if __name__ == "__main__":
    main()
