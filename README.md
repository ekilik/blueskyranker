# blueskyranker
Bluesky topic ranker and pipeline for the NEWSFLOWS project. Fetch public posts into SQLite, cluster by topic, rank by cluster engagement, and optionally push ranked posts to a feed generator. Supports user‑controlled time windows and a dry‑run mode.

## Concepts at a Glance

- **Flow**: Fetch → Store in SQLite → Rank (topic clustering) → Push / Export.
- **Time windows**: independently control clustering, engagement scoring, and push eligibility.
- **Priority ordering**: higher numbers = higher priority; starts at 1000 and decrements.
- **Defaults**: SBERT topic ranker uses `similarity_threshold=0.5` with the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model; TF‑IDF / Count use `0.2`.

## Table of Contents
- [Concepts at a Glance](#concepts-at-a-glance)
- [Features](#features)
- [Quickstart Map](#quickstart-map)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Use-Case Playbook](#use-case-playbook)
- [Examples](#examples)
  - [Fetch and Rank per Handle](#fetch-and-rank-per-handle)
  - [Storage: SQLite](#storage-sqlite)
  - [Load Posts with Polars](#load-posts-with-polars)
  - [Import Historical CSVs](#import-historical-csvs)
- [Pipeline](#pipeline)
  - [Core Workflow](#core-workflow)
  - [Tuning & Flags](#tuning--flags)
  - [Dry Run](#dry-run)
- [Reports & Data](#reports--data)
  - [Cluster Report](#cluster-report)
  - [Data Schema (SQLite)](#data-schema-sqlite)
- [Configuration (.env)](#configuration-env)
- [Advanced Details](#advanced-details)
- [Troubleshooting](#troubleshooting)
- [End-to-end demo / Notebook](#end-to-end-demo--notebook)
- [To-Do / Roadmap](#to-do--roadmap)

## Features

- SQLite storage with upsert-by-URI (engagement refresh on repeat runs)
- Rankers: trivial, popularity-based, and topic-based (TF‑IDF / count / SBERT)
- Optional posting of priorities to a feed-generator API
- Opt-in diagnostics: SBERT centroid distances per post, cluster span metadata, and keyword/KeyBERT summaries for quick validation

## Quickstart Map

| Goal | Run this | Read next |
| --- | --- | --- |
| Explore sample data without network | `python -m blueskyranker.sample_db --db newsflows_sample.db` | [Quickstart](#quickstart) |
| Fetch live posts into SQLite | `python blueskyranker/fetcher.py --max-age-days 7 --sqlite-path blueskyranker/newsflows.db` | [Fetch and Rank per Handle](#fetch-and-rank-per-handle) |
| Run the full pipeline | `python -m blueskyranker.pipeline` | [Core Workflow](#core-workflow) |
| Preview priorities only | `python -m blueskyranker.pipeline --dry-run` | [Dry Run](#dry-run) |
| Generate a topic report | `python -m blueskyranker.cluster_report --db blueskyranker/newsflows.db --output cluster_report.md` | [Cluster Report](#cluster-report) |

## Installation

- Python 3.10–3.11 recommended
- Install dependencies: `pip install -r requirements.txt`
- Optional (cluster insights / KeyBERT): `pip install keybert keyphrase-vectorizers`
- Optional (console scripts): `pip install -e .` to enable `bsr-pipeline`, `bsr-fetch`, and `bsr-report` commands.

## Quickstart

### Step 1 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2A – Create a sample database (offline)
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

### Step 2B – Fetch live posts
```bash
python blueskyranker/fetcher.py \
  --handles news-flows-nl.bsky.social news-flows-ir.bsky.social news-flows-cz.bsky.social news-flows-fr.bsky.social \
  --max-age-days 7 --sqlite-path blueskyranker/newsflows.db
```
Notes:
- Incremental by default: fetches newer posts since the latest saved timestamp within the last `--max-age-days`.
- Add `--refresh-window` to re-fetch the entire window and refresh engagement metrics.

### Step 3 – Rank posts per handle
```python
from blueskyranker.fetcher import ensure_db, load_posts_df
from blueskyranker.ranker import TopicRanker

conn = ensure_db('newsflows_sample.db')  # or 'blueskyranker/newsflows.db' if you fetched live
df_nl = load_posts_df(conn, handle='sample.handle', limit=2000)  # adjust handle as needed

ranker = TopicRanker(
    returnformat='dataframe',
    method='networkclustering-tfidf',
    descending=True,
    similarity_threshold=0.2,
    vectorizer_stopwords='english',
    # Optional time windows (days):
    cluster_window_days=7,
    engagement_window_days=3,
    push_window_days=1,
)
ranking_nl = ranker.rank(df_nl)
print(ranking_nl.head())
```

### Step 4 – Post priorities (optional)
```python
from blueskyranker.ranker import TopicRanker
from dotenv import load_dotenv
load_dotenv('.env')

ranker = TopicRanker(returnformat='dataframe', method='networkclustering-tfidf', descending=True)
ranking = ranker.rank(df_nl)
ranker.post(test=False)
```
Set `FEEDGEN_HOSTNAME` and `PRIORITIZE_API_KEY` in `.env`. We use `python-dotenv` to load these.

## Use-Case Playbook

<details>
<summary>Test the entire pipeline once</summary>

**Status**: Implemented (pytest smoke tests with stubbed network/push layers).

**CLI**
```bash
python -m pytest tests/test_windows.py -k "pipeline" --maxfail=1 -q
```

**What you get**: Runs both `test_pipeline_smoke` and `test_pipeline_dry_run`, writing an isolated SQLite db and push log locally. Failures point to window filtering, ranking order, or logging regressions.

**Make it even better**:
- Capture the artifacts with `pytest --basetemp .pytest_artifacts` so logs/DBs are saved for inspection.
</details>

<details>
<summary>Update all entries for the past <code>X</code> days</summary>

**Status**: Fully supported by the fetcher CLI (refreshes engagement metrics inside the window).

**CLI**
```bash
X_DAYS=7
python blueskyranker/fetcher.py \
  --handles news-flows-nl.bsky.social news-flows-ir.bsky.social news-flows-cz.bsky.social news-flows-fr.bsky.social \
  --max-age-days "$X_DAYS" \
  --refresh-window \
  --sqlite-path blueskyranker/newsflows.db
```

**What you get**: Forces a full refresh for every handle inside the `X_DAYS` window (likes/reposts/replies), then backfills anything older than your last incremental scrape.

**Make it even better**:
- Pair with `--cutoff-check-every 2` when rate limits spike; it halves API polling without missing new posts.
- Append `--include-pins` when you also want pinned posts refreshed inside the window.
</details>

<details>
<summary>Create topic clustering and ranking with custom windows</summary>

**Status**: Implemented via the pipeline CLI; runs clustering per handle and prints a dry-run summary.

**CLI**
```bash
CLUSTER_DAYS=7
ENGAGEMENT_DAYS=1
PUSH_DAYS=1
python -m blueskyranker.pipeline \
  --sqlite-path blueskyranker/newsflows.db \
  --cluster-window-days "$CLUSTER_DAYS" \
  --engagement-window-days "$ENGAGEMENT_DAYS" \
  --push-window-days "$PUSH_DAYS" \
  --method networkclustering-sbert \
  --similarity-threshold 0.5 \
  --dry-run \
  --log-path push.log
```

**What you get**: Clusters all eligible handles, interleaves high-engagement topics, writes a push summary to `push.log`, and exports a JSON preview to `push_exports/`—all without hitting the feed generator.

**Make it even better**:
- Swap in `--method networkclustering-tfidf --vectorizer-stopwords english` for language-specific debugging sessions.
- Add `--handles` with a subset when replaying historical incidents to shorten iterations.
- Chain `python -m blueskyranker.cluster_report` right after to generate a human-readable Markdown report for newsroom review.
</details>

> Potential addition: consider an export-focused recipe (e.g. “Generate cluster QA report for the last week”) to round out the playbook for editorial analytics.

## Examples

Priority semantics: the feed API treats HIGHER numbers as higher priority. The pipeline assigns priorities starting at 1000 for the first item, then 999, 998, … in order. Rankers output the most important item first; use `descending=True` to put strongest items at the top.

Basic rankers (programmatic):
```python
from blueskyranker import TrivialRanker, PopularityRanker

trivial = TrivialRanker(returnformat='id', metric=None, descending=True)
popular = PopularityRanker(returnformat='dicts', metric='reply_count', descending=True)
```

TopicRanker (clusters + engagement ordering): see Quickstart and Time Windows sections for parameters.

### Fetch and Rank per Handle
You can fetch recent public posts, then run the topic ranker separately per handle. For example:

```python
from blueskyranker.fetcher import ensure_db, load_posts_df, Fetcher
from blueskyranker.ranker import TopicRanker

# Download recent public posts for default handles (writes to SQLite)
fetcher = Fetcher()
print(fetcher.fetch())

# Rank per handle using TF–IDF topical clustering (load from SQLite)
conn = ensure_db('blueskyranker/newsflows.db')
df_nl = load_posts_df(conn, handle='news-flows-nl.bsky.social', limit=2000)
ranker = TopicRanker(returnformat='dataframe', method='networkclustering-tfidf', descending=True)
ranking_nl = ranker.rank(df_nl)
print(ranking_nl.head())
```

### Storage: SQLite
Fetched posts are stored in a SQLite database and upserted by `uri` (so engagement metrics and metadata are refreshed on subsequent runs). The database defaults to `blueskyranker/newsflows.db` inside the package directory.

```bash
python blueskyranker/fetcher.py --max-age-days 7 --sqlite-path blueskyranker/newsflows.db
```

If you need CSVs (for exploration or interoperability), export them from the DB in a short Python snippet:

```python
from blueskyranker.fetcher import ensure_db, export_db_to_csv
conn = ensure_db('blueskyranker/newsflows.db')
paths = export_db_to_csv(conn, output_dir='.', include_combined=True)
print(paths)
```

This writes per-handle CSVs like `news-flows-nl_bsky_social_author_feed.csv` and a combined CSV `all_handles_author_feed.csv`.

### Load Posts with Polars
You can work directly from the database without exporting to CSV:

```python
import polars as pl
from blueskyranker.fetcher import ensure_db, load_posts_df

conn = ensure_db('blueskyranker/newsflows.db')
df_nl = load_posts_df(conn, handle='news-flows-nl.bsky.social', limit=2000, order_by='createdAt', descending=False)
print(df_nl.head())
```

### Import Historical CSVs
If you have older CSVs from previous runs, you can import them into the database (upsert by `uri`) to keep history and let future runs refresh engagement counts:

```python
from blueskyranker.fetcher import ensure_db, import_csvs_to_db

conn = ensure_db('blueskyranker/newsflows.db')
csvs = [
  'news-flows-nl_bsky_social_author_feed.csv',
  'news-flows-ir_bsky_social_author_feed.csv',
  'news-flows-cz_bsky_social_author_feed.csv',
  'news-flows-fr_bsky_social_author_feed.csv',
]
rows = import_csvs_to_db(conn, csvs)
print(f"Imported {rows} rows from CSVs into SQLite")
```

## Pipeline

### Core Workflow

You can control how many days of posts contribute to each stage of the pipeline:
- **Clustering window** (`--cluster-window-days`): builds the topical graph.
- **Engagement window** (`--engagement-window-days`): scores each cluster.
- **Push window** (`--push-window-days`): limits which posts can be prioritised.

Running the CLI without extra flags fetches, ranks, and (by default) performs a test push for the four NEWSFLOWS handles:
- handles: `news-flows-[nl|fr|ir|cz].bsky.social`.
- method: `networkclustering-sbert` (defaults to `similarity_threshold=0.5`).
- windows: clustering `7d`, engagement `1d`, push `1d`.
- demotion: enabled for the last 48 hours (`--demote-last --demote-window-hours 48`).

```bash
python -m blueskyranker.pipeline
```

Override any default as needed:

```bash
python -m blueskyranker.pipeline \
  --handles news-flows-nl.bsky.social news-flows-fr.bsky.social \
  --method networkclustering-sbert --similarity-threshold 0.5 \
  --cluster-window-days 7 --engagement-window-days 1 --push-window-days 1 \
  --demote-last --demote-window-hours 48 \
  --log-path push.log --no-test
```

Programmatic orchestration mirrors the CLI:

```python
from blueskyranker.pipeline import run_fetch_rank_push

run_fetch_rank_push(
    handles=['news-flows-nl.bsky.social'],
    method='networkclustering-sbert', similarity_threshold=0.5,
    cluster_window_days=7, engagement_window_days=1, push_window_days=1,
    include_pins=False, test=False, log_path='push.log')
```

Ordering highlights:
- Filter to the push window first; only eligible posts participate in ranking.
- Clusters are ordered by engagement (higher first) with deterministic tie-breaks.
- Posts are interleaved round-robin across clusters, newest first within each cluster.
- Priorities start at 1000 and decrease by 1 (never below 1; demoted posts are set to 0).

Sample log excerpt:

```text
2025-09-08T15:45:02+0000 INFO [OK] handle=news-flows-nl.bsky.social posts=42 method=networkclustering-sbert threshold=0.5 windows=(cluster=7d, engagement=1d, push=1d)
  cluster=12 size=10 engagement=538 keywords="europe policy migration"
  cluster=4  size=8  engagement=410 keywords="covid vaccine health"
  cluster=9  size=6  engagement=295 keywords="energy gas price"
```

### Tuning & Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--handles` | Four NEWSFLOWS feeds | Space-separated list; fetches, ranks, pushes per handle. |
| `--method` | `networkclustering-sbert` | Alternatives: `networkclustering-tfidf`, `networkclustering-count`. |
| `--similarity-threshold` | `0.5` for SBERT, `0.2` otherwise | Leave unset to accept per-method defaults. |
| `--cluster-window-days` | `7` | Context window for clustering; extends fetch if larger than engagement/push windows. |
| `--engagement-window-days` | `1` | Computes cluster engagement; influences ordering. |
| `--push-window-days` | `1` | Restricts which posts are prioritised. |
| `--demote-last` / `--no-demote-last` | Enabled | Demotes items from the last `--demote-window-hours` not in the current push. |
| `--demote-window-hours` | `48` | Time horizon for demotion bookkeeping. |
| `--fetch-max-age-days` | None | Caps both fetch phases; useful for quick tests. |
| `--include-pins` | Disabled | Include pinned posts from AppView responses. |
| `--test` / `--no-test` | `--test` | `--no-test` persists priorities to the feed generator. |
| `--log-path` | `push.log` | Location of the per-run summary log. |

See [Advanced Details](#advanced-details) for logging, JSON export, and demotion internals.

### Dry Run

Preview the ranked order without calling the feed generator:

```bash
python -m blueskyranker.pipeline --dry-run
```

The command prints a summary (top clusters, priority preview) and writes the JSON export to `push_exports/`. The `counts.demoted` field reports how many recent posts were explicitly demoted.

## Reports & Data

### Cluster Report

You can generate a per-handle topic report (top clusters with keywords, sizes, engagement, and sample headlines) straight from SQLite.

CLI
```bash
python -m blueskyranker.cluster_report --db blueskyranker/newsflows.db --output cluster_report.md \
  --method networkclustering-sbert --sample-max 300 --similarity-threshold 0.5 --stopwords english
```

Programmatic
```python
from blueskyranker.cluster_report import generate_cluster_report
generate_cluster_report(db_path='blueskyranker/newsflows.db', output_path='cluster_report.md',
                        method='networkclustering-sbert', sample_max=600,
                        similarity_threshold=0.5, vectorizer_stopwords='english')
```

### Data Schema (SQLite)
Table `posts` (upsert by `uri`):

`uri, cid, author_handle, author_did, indexedAt, createdAt, text, reply_root_uri, reply_parent_uri, is_repost, like_count, repost_count, reply_count, quote_count, news_title, news_description, news_uri`.

`createdAt` is stored as ISO-8601 text; the code parses to datetime on demand (assumed UTC).

## Configuration (.env)
Keep your secrets in `blueskyranker/.env` (this file is excluded from version control). Copy `.env.example` as a starting point and fill in your values. For posting to the feed generator:

```ini
FEEDGEN_HOSTNAME=feed.example.org
PRIORITIZE_API_KEY=...secret...
```

Optional overrides:

```ini
# When the feed generator listens on a different host/port locally
FEEDGEN_LISTENHOST=localhost:3020

# Offline SBERT usage (set both to reuse a local Hugging Face cache)
SBERT_MODEL_PATH=/path/to/.cache/huggingface/hub
SBERT_LOCAL_ONLY=1
```

Use `test=True` for a test request that doesn’t persist, or `--dry-run` to avoid calling the API entirely.

## Advanced Details

<details>
<summary>Pipeline internals</summary>

- Fetcher uses the public AppView (`https://public.api.bsky.app/xrpc`) via `atproto` with polite pagination.
- Incremental fetching stops once posts fall outside the active window and reuses the latest `createdAt` stored in SQLite.
- Rankers available: Trivial (keeps order), Popularity (sorts by a single engagement metric), Topic (TF‑IDF / Count / SBERT clustering via Leiden).
- Priorities start at 1000 and decrease by 1; higher numbers mean higher priority.

</details>

<details>
<summary>Demotion & priority bookkeeping</summary>

Background: previously pushed high-priority posts could linger at the top until overwritten. The pipeline can include the last run’s URIs that are missing from the current push and explicitly set their priority to 0.

- Enable/disable via CLI: `--demote-last` (default) / `--no-demote-last`.
- Window: by default, considers the last 48 hours. Configure via `--demote-window-hours 48`.
- Logging: `push.log` records `demoted (time-window): <count>` per handle.
- Export: the per-run JSON in `push_exports/` includes `counts.demoted`.

Notes:
- Demotion is per handle and only applies to URIs present in the most recent export for that handle.
- Demoted URIs are appended to the POST payload as `{ "priority": 0, "uri": "..." }` and are ignored if they already appear in the current push set.

</details>

<details>
<summary>Cluster diagnostics & validation helpers</summary>

The topic ranker can now emit additional validation columns when requested (disabled by default so existing pipelines keep running light):

- **SBERT centroid metrics** – set `include_embedding_metrics=True` (Python) or `embedding_metrics = TRUE` (R bridge) to add per-post cosine similarity and distance to each cluster centroid.
- **Cluster span timestamps** – the ranker now records the first/last post timestamps and exact duration (in hours) for every cluster.
- **Keyword/KeyBERT labels** – pass `cluster_insights = ["distinct_words"]`, `cluster_insights = ["keybert"]`, or both. Distinct words are drawn from a TF–IDF comparison across clusters; KeyBERT uses the same SentenceTransformer instance as the SBERT ranker.
- **Configuration knobs** – supply nested dictionaries/lists via `cluster_insight_options` (Python) or `cluster_insight_options = list(...)` (R) to tweak `top_n`, `max_docs_per_cluster`, `max_chars`, `diversity`, etc.

> **KeyBERT install**: optional. If you want semantic cluster labels, install `keybert` (`pip install keybert keyphrase-vectorizers`) inside the same environment. When it is missing, the ranker simply skips that insight and logs a warning.

The default SBERT model is now `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which is lighter and multilingual. Set `SBERT_MODEL_PATH` and `SBERT_LOCAL_ONLY=1` if you need to run completely offline.

</details>

<details>
<summary>Logging & exports</summary>

- Server response handling: short responses print to stdout; longer responses (> ~2000 chars) are saved to `push_exports/prioritize_response_{handle}_{timestamp}.{json|txt}`.
- Export filenames: `push_exports/push_{handle}_{YYYY-MM-DDTHH-mm-ssZ}.json` (UTC timestamp, safe characters).
- JSON export content: includes `run` metadata, `counts.*`, `top_clusters`, and the ordered `items` list (with priorities, URIs, metadata).

</details>

<details>
<summary>Time & ranking specifics</summary>

- Fetch behaviour: the pipeline refreshes engagement metrics for the largest of the cluster/engagement/push windows (or `--fetch-max-age-days` if provided) before extending to the clustering horizon.
- Time handling: timestamps are stored in UTC (`createdAt`, `createdAt_ns`); exports add `createdAt_local` (Europe/Amsterdam) for validation only.
- Ranking inputs: cluster engagement sums likes + replies + quotes + reposts over the engagement window; deterministic tie-breaks rely on recency and cluster id.

</details>

## Troubleshooting
- igraph/leidenalg install issues: ensure system packages for igraph are installed before pip installing python‑igraph and leidenalg (platform‑specific).
- Empty ranking: widen `push_window_days` or check that your DB contains recent posts.
- SBERT method: slower and memory‑heavier; start with smaller batches.

### Assumptions
- Public AppView endpoints are reachable; engagement counts may drift between runs.
- Posts can be multilingual; clustering quality improves with language-aware preprocessing and stopwords.
- Embedded link metadata (title/description/uri) may be missing. Empty strings are normalized to NULL in SQLite. The fetcher flags any empty-string anomalies in its final report (these indicate an upstream embed extraction issue).

## End-to-end demo / Notebook
Check out `example.ipynb` to see how we first download the data and then rank it. The notebook expects a SQLite DB to exist; create one via the sample (`python -m blueskyranker.sample_db --db newsflows_sample.db`) or the fetcher.

## To‑Do / Roadmap

- Expand example notebook to showcase SQLite workflows and SBERT best practices.

## Changelog

<details>
<summary><strong>v1.1 (2025-10-08)</strong></summary>

- **Additions**
  - New `bridge.py` helpers for the R/reticulate workflow.
  - Progress callbacks and richer per-handle progress reporting in the fetcher.
  - Enhanced TopicRanker diagnostics (cluster durations, optional embedding metrics).
  - CLI scripts for dry-run pipeline/ranking flows without network access.
  - Refreshed example notebook to mirror the updated workflow.
- **Changes**
  - Default SQLite path now resolves to `blueskyranker/newsflows.db`.
  - SBERT default model switched to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  - Automatic similarity thresholds when omitted (0.5 for SBERT, 0.2 for TF‑IDF/Count).
  - Dry-run output optionally quieter via `--quiet-dry-run`.
  - Logging filenames use safe handle strings and UTC timestamps (`YYYYMMDD_HHMMSS`).
- **Fixes**
  - Normalization of timestamps and text fields to handle R bridge edge cases.
  - More defensive handling when saves or callbacks fail during fetch.

**Maintaining v1.0 behaviour**
- Run the fetcher/pipeline with `--sqlite-path newsflows.db` (or set `DEFAULT_SQLITE_PATH` back to the repo root) to keep the original DB location.
- Explicitly pass `--similarity-threshold 0.2` (TF‑IDF) / `0.5` (SBERT) if you rely on the previous defaults.
- Leave `progress_callback=None` to retain the familiar tqdm progress bars during fetch.
- Keep using `FEEDGEN_HOSTNAME` (without scheme) in `.env`; the new resolver also accepts `FEEDGEN_LISTENHOST` or full URLs but behaves the same when `FEEDGEN_HOSTNAME` is set.

</details>

<details>
<summary><strong>v1.0 (2025-09-25)</strong></summary>

- Initial version deployed on the NEWSFLOWS server with manual SQLite path (`newsflows.db`), base topic ranking, and the original README.

</details>
