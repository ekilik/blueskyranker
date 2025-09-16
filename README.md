# blueskyranker
Bluesky topic ranker and pipeline for the NEWSFLOWS project. Fetch public posts into SQLite, cluster by topic, rank by cluster engagement, and optionally push ranked posts to a feed generator. Supports user‑controlled time windows and a dry‑run mode.

## Table of Contents
- Features
- Installation
- Quickstart (SQLite sample or Live Fetch)
- Usage and Examples
- Time Windows and Pipeline
- Dry Run
- Cluster Report
- Configuration (.env)
- Data Schema (SQLite)
- Troubleshooting
- End‑to‑end demo / Notebook

## Features

- SQLite storage with upsert-by-URI (engagement refresh on repeat runs)
- Rankers: trivial, popularity-based, and topic-based (TF‑IDF / count / SBERT)
- Optional posting of priorities to a feed‑generator API
- End‑to‑end fetch → rank (per handle) → push pipeline with user‑controlled time windows

## Installation

- Python 3.10–3.11 recommended
- Install dependencies: `pip install -r requirements.txt`
- Optional (console scripts): `pip install -e .` to enable `bsr-pipeline`, `bsr-fetch`, and `bsr-report` commands.

## Quickstart

1) Install requirements
```bash
pip install -r requirements.txt
```

2) Option A: Create a small sample SQLite from the bundled CSV (no network)
```bash
python -m blueskyranker.sample_db --db newsflows_sample.db
```
Then load rows directly from SQLite:
```python
from blueskyranker.fetcher import ensure_db, load_posts_df
conn = ensure_db('newsflows_sample.db')
df = load_posts_df(conn)
print(df.head())
```

2) Option B: Fetch recent public posts (defaults shown)
```bash
python blueskyranker/fetcher.py \
  --handles news-flows-nl.bsky.social news-flows-ir.bsky.social news-flows-cz.bsky.social news-flows-fr.bsky.social \
  --max-age-days 7 --sqlite-path newsflows.db
```
Notes:
- Incremental by default: fetches newer posts since the latest saved timestamp, within the last N days.
- Full refresh of the window: add `--refresh-window` to re‑fetch the entire N‑day window (updates engagement for all posts in that period).

3) Rank per handle (TF‑IDF topic clustering example)
```python
from blueskyranker.fetcher import ensure_db, load_posts_df
from blueskyranker.ranker import TopicRanker

conn = ensure_db('newsflows_sample.db')  # or 'newsflows.db' if you fetched live
df_nl = load_posts_df(conn, handle='sample.handle', limit=2000)  # adjust handle if needed

ranker = TopicRanker(
    returnformat='dataframe',
    method='networkclustering-tfidf',
    descending=True,
    similarity_threshold=0.2,
    vectorizer_stopwords='english',
    # Optional time windows (days):
    cluster_window_days=7,        # how far back to consider for clustering
    engagement_window_days=3,     # how far back to compute engagement per cluster
    push_window_days=1,           # how far back to include items in the output
)
ranking_nl = ranker.rank(df_nl)
print(ranking_nl.head())
```

4) Post the priorities (optional)
```python
from blueskyranker.ranker import TopicRanker
from dotenv import load_dotenv
load_dotenv('.env')

ranker = TopicRanker(returnformat='dataframe', method='networkclustering-tfidf', descending=True)
ranking = ranker.rank(df_nl)
ranker.post(test=False)
```
Set `FEEDGEN_HOSTNAME` and `PRIORITIZE_API_KEY` in `.env`. We use `python-dotenv` to load these.

## Usage and Examples

Priority semantics: the feed API treats HIGHER numbers as higher priority. The pipeline assigns priorities starting at 1000 for the first item, then 999, 998, … in order. Rankers output the most important item first; use `descending=True` to put strongest items at the top.

Basic rankers (programmatic):
```python
from blueskyranker import TrivialRanker, PopularityRanker

trivial = TrivialRanker(returnformat='id', metric=None, descending=True)
popular = PopularityRanker(returnformat='dicts', metric='reply_count', descending=True)
```

TopicRanker (clusters + engagement ordering): see Quickstart and Time Windows sections for parameters.

### Fetch and rank per handle
You can fetch recent public posts, then run the topic ranker separately per handle. For example:

```python
from blueskyranker.fetcher import ensure_db, load_posts_df, Fetcher
from blueskyranker.ranker import TopicRanker

# Download recent public posts for default handles (writes to SQLite)
fetcher = Fetcher()
print(fetcher.fetch())

# Rank per handle using TF–IDF topical clustering (load from SQLite)
conn = ensure_db('newsflows.db')
df_nl = load_posts_df(conn, handle='news-flows-nl.bsky.social', limit=2000)
ranker = TopicRanker(returnformat='dataframe', method='networkclustering-tfidf', descending=True)
ranking_nl = ranker.rank(df_nl)
print(ranking_nl.head())
```

### Storage: SQLite (default)
Fetched posts are stored in a SQLite database and upserted by `uri` (so engagement metrics and metadata are refreshed on subsequent runs). The database defaults to `newsflows.db` in the working directory.

```bash
python blueskyranker/fetcher.py --max-age-days 7 --sqlite-path newsflows.db
```

If you need CSVs (for exploration or interoperability), export them from the DB in a short Python snippet:

```python
from blueskyranker.fetcher import ensure_db, export_db_to_csv
conn = ensure_db('newsflows.db')
paths = export_db_to_csv(conn, output_dir='.', include_combined=True)
print(paths)
```

This writes per-handle CSVs like `news-flows-nl_bsky_social_author_feed.csv` and a combined CSV `all_handles_author_feed.csv`.

### Load posts from SQLite into Polars
You can work directly from the database without exporting to CSV:

```python
import polars as pl
from blueskyranker.fetcher import ensure_db, load_posts_df

conn = ensure_db('newsflows.db')
df_nl = load_posts_df(conn, handle='news-flows-nl.bsky.social', limit=2000, order_by='createdAt', descending=False)
print(df_nl.head())
```

### Migrate existing CSVs into SQLite
If you have older CSVs from previous runs, you can import them into the database (upsert by `uri`) to keep history and let future runs refresh engagement counts:

```python
from blueskyranker.fetcher import ensure_db, import_csvs_to_db

conn = ensure_db('newsflows.db')
csvs = [
  'news-flows-nl_bsky_social_author_feed.csv',
  'news-flows-ir_bsky_social_author_feed.csv',
  'news-flows-cz_bsky_social_author_feed.csv',
  'news-flows-fr_bsky_social_author_feed.csv',
]
rows = import_csvs_to_db(conn, csvs)
print(f"Imported {rows} rows from CSVs into SQLite")
```

## End‑to‑end demo / Notebook
Check out `example.ipynb` to see how we first download the data and then rank it. The notebook expects a SQLite DB to exist; create one via the sample (`python -m blueskyranker.sample_db --db newsflows_sample.db`) or the fetcher.

## Time windows and pipeline

You can control how many days of posts are used for:
- Clustering (`cluster_window_days`)
- Engagement ranking of clusters (`engagement_window_days`)
- Final push to the API (`push_window_days`)

A CLI runs the whole flow per handle and logs pushed clusters:

```bash
python -m blueskyranker.pipeline \
  --handles news-flows-nl.bsky.social news-flows-fr.bsky.social news-flows-ir.bsky.social news-flows-cz.bsky.social \
  --method networkclustering-sbert \
  --similarity-threshold 0.5 \
  --cluster-window-days 7 \
  --engagement-window-days 1 \
  --push-window-days 1 \
  --demote-last \
  --demote-window-hours 48 \
  --refresh-window \
  --log-path push.log \
  --no-test
```

### Ordering logic

The TopicRanker orders posts using three time windows:

- Clustering window: builds topical clusters from posts in this window (e.g., 7 days).
- Engagement window: computes cluster engagement totals (likes + replies + quotes + reposts) and derives `cluster_engagement_rank` (1 = most engaged).
- Push window: restricts the final output to only posts in this window (e.g., 24–48h).

Ordering steps:
- Filter to the push window first; only eligible posts participate in ordering.
- Order clusters by `cluster_engagement_rank` (most engaged first).
- Within each cluster, sort posts by recency (newest first).
- Interleave round‑robin across clusters in that rank order (1, 2, 3, … then repeat) to create the final sequence.

Implications:
- The first post is the most recent item (within the push window) from the most‑engaged cluster (measured over the engagement window).
- If a high‑ranked cluster has no posts in the push window, it is skipped.
- The “Top clusters” block in the export is computed over the pushed subset and may not exactly match the engagement window used for ranking (by design), but will generally align closely.

Programmatic one‑liner:

```python
from blueskyranker.pipeline import run_fetch_rank_push

run_fetch_rank_push(
    handles=['news-flows-nl.bsky.social'],
    method='networkclustering-tfidf', similarity_threshold=0.2,
    cluster_window_days=7, engagement_window_days=3, push_window_days=1,
    include_pins=False, test=False, log_path='push.log')
```

Each push appends a concise summary to the log, e.g.:

```text
2025-09-08T15:45:02+0000 INFO     [OK] handle=news-flows-nl.bsky.social posts=42 method=networkclustering-tfidf threshold=0.2 windows=(cluster=7d, engagement=3d, push=1d)
  cluster=12 size=10 engagement=538 keywords="europe policy migration"
  cluster=4  size=8  engagement=410 keywords="covid vaccine health"
  cluster=9  size=6  engagement=295 keywords="energy gas price"
  - demoted last-run URIs: 7
  - priorities: first=1000, decreasing by 1
```

### Dry run (no API call)

Add `--dry-run` to print an intelligible summary and a priority preview instead of calling the API:

```bash
python -m blueskyranker.pipeline \
  --handles news-flows-nl.bsky.social \
  --method networkclustering-tfidf --similarity-threshold 0.2 \
  --cluster-window-days 7 --engagement-window-days 3 --push-window-days 1 \
  --dry-run
```

Sample output (abbreviated):

```
=== Dry Run: handle=news-flows-nl.bsky.social posts=42 method=networkclustering-tfidf threshold=0.2 windows=(cluster=7d, engagement=3d, push=1d)
  cluster=12 size=10 engagement=538 keywords="europe policy migration"
  cluster=4  size=8  engagement=410 keywords="covid vaccine health"
  cluster=9  size=6  engagement=295 keywords="energy gas price"
  Priority preview (top 15):
      0  c=12    2025-09-08T...  at://...  | Title of the top story
      1  c=4     2025-09-08T...  at://...  | Another headline
      ...
```
The JSON export also includes `counts.demoted`.

### Demote previously pushed items (default on)

Background: previously pushed high-priority posts could linger at the top until overwritten. To counter this, the pipeline can include the last run’s URIs that are missing from the current push and explicitly set their priority to 0.

- Enable/disable via CLI: `--demote-last` (default) / `--no-demote-last`.
- Window: by default, considers the last 48 hours. Configure via `--demote-window-hours 48`.
- Logging: `push.log` shows `demoted last-run URIs: <count>`.
- Export: the per-run JSON in `push_exports/` includes `counts.demoted`.

Notes:
- Demotion is per handle and only applies to URIs present in the most recent export for that handle.
- Demoted URIs are appended to the POST payload as `{ "priority": 0, "uri": "..." }` and are ignored if they already appear in the current push set.

## Cluster Report

You can generate a per-handle topic report (top clusters with keywords, sizes, engagement, and sample headlines) straight from SQLite.

CLI
```bash
python -m blueskyranker.cluster_report --db newsflows.db --output cluster_report.md \
  --method networkclustering-sbert --sample-max 300 --similarity-threshold 0.2 --stopwords english
```

Programmatic
```python
from blueskyranker.cluster_report import generate_cluster_report
generate_cluster_report(db_path='newsflows.db', output_path='cluster_report.md',
                        method='networkclustering-tfidf', sample_max=600,
                        similarity_threshold=0.2, vectorizer_stopwords='english')
```

## Internals (short)

- Fetcher uses the public AppView (`https://public.api.bsky.app/xrpc`) via `atproto` and paginates with polite pacing.
- Incremental fetching: for each handle, we stop early once posts are older than `--max-age-days`; we also use the latest `createdAt` in the DB to avoid refetching older content.
- SQLite schema columns: `uri, cid, author_handle, author_did, indexedAt, createdAt, text, reply_root_uri, reply_parent_uri, is_repost, like_count, repost_count, reply_count, quote_count, news_title, news_description, news_uri`.
- Rankers
  - TrivialRanker: keeps input order.
  - PopularityRanker: sorts by a chosen engagement metric.
  - TopicRanker: builds a similarity graph (TF‑IDF/Count/SBERT), thresholds it, then applies Leiden clustering; clusters are ordered by aggregate engagement and interleaved for diversity.
- Priority semantics: the feed API treats HIGHER numbers as higher priority (the first item is assigned the highest numeric priority). Rankers therefore output the most important item first; use `descending=True` to put strongest items at the top.

## Configuration (.env)
For posting to the feed generator:

```ini
FEEDGEN_HOSTNAME=feed.example.org
PRIORITIZE_API_KEY=...secret...
```

Use `test=True` for a test request that doesn’t persist, or `--dry-run` to avoid calling the API entirely.

### Server response handling
When pushing priorities, the server’s response is printed if short. If it’s long (> ~2000 chars), it is saved to `push_exports/prioritize_response_{handle}_{YYYY-MM-DDTHH-mm-ssZ}.{json|txt}` and the path is printed and logged.

### Export filenames
Push exports are written to `push_exports/` with a human-readable UTC timestamp:

```
push_{handle}_{YYYY-MM-DDTHH-mm-ssZ}.json
```
For example: `push_news-flows-nl_bsky_social_2025-09-15T14-50-24Z.json`.

### Fetch behavior (engagement refresh)

The pipeline’s fetch step refreshes engagement metrics for all posts within the effective window by default. The effective fetch window is `max(cluster_window_days, engagement_window_days, push_window_days)` unless overridden via `--fetch-max-age-days`.

- Default: `--refresh-window` is enabled, which re-fetches the entire effective window and upserts to SQLite (updating likes/replies/quotes/reposts).
- To opt out (incremental only): pass `--no-refresh-window` (will fetch only posts newer than the latest in DB and not refresh older rows inside the window).

## Data Schema (SQLite)
Table `posts` (upsert by `uri`):

`uri, cid, author_handle, author_did, indexedAt, createdAt, text, reply_root_uri, reply_parent_uri, is_repost, like_count, repost_count, reply_count, quote_count, news_title, news_description, news_uri`.

`createdAt` is stored as ISO‑8601 text; the code parses to datetime on demand (assumed UTC).

## Troubleshooting
- igraph/leidenalg install issues: ensure system packages for igraph are installed before pip installing python‑igraph and leidenalg (platform‑specific).
- Empty ranking: widen `push_window_days` or check that your DB contains recent posts.
- SBERT method: slower and memory‑heavier; start with smaller batches.

### Assumptions
- Public AppView endpoints are reachable; engagement counts may drift between runs.
- Posts can be multilingual; clustering quality improves with language-aware preprocessing and stopwords.
- Embedded link metadata (title/description/uri) may be missing. Empty strings are normalized to NULL in SQLite. The fetcher flags any empty-string anomalies in its final report (these indicate an upstream embed extraction issue).

## To‑Do / Roadmap

- Expand example notebook to showcase SQLite workflows and SBERT best practices.
