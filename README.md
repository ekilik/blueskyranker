# blueskyranker
This is just a small building block for larger NEWSFLOWS project.

Very, very much work in progress. 

Idea: take a list of bluesky posts and re-rank them.

## Features

- SQLite storage with upsert-by-URI (engagement refresh on repeat runs)
- Rankers: trivial, popularity-based, and topic-based (TF‑IDF / count / SBERT)
- Optional posting of priorities to a feed‑generator API

## Quickstart

1) Install requirements
```
pip install -r requirements.txt
```

2) Fetch recent public posts (defaults shown)
```
python blueskyranker/fetcher.py \
  --handles news-flows-nl.bsky.social news-flows-ir.bsky.social news-flows-cz.bsky.social news-flows-fr.bsky.social \
  --max-age-days 7 --sqlite-path newsflows.db
```

3) Rank per handle (TF‑IDF topic clustering example)
```
from blueskyranker.fetcher import ensure_db, load_posts_df
from blueskyranker.ranker import TopicRanker

conn = ensure_db('newsflows.db')
df_nl = load_posts_df(conn, handle='news-flows-nl.bsky.social', limit=2000)

ranker = TopicRanker(returnformat='dataframe', method='networkclustering-tfidf', descending=True)
ranking_nl = ranker.rank(df_nl)
print(ranking_nl.head())
```

4) Post the priorities (optional)
```
from blueskyranker.ranker import TopicRanker
from dotenv import load_dotenv
load_dotenv('.env')

ranker = TopicRanker(returnformat='dataframe', method='networkclustering-tfidf', descending=True)
ranking = ranker.rank(df_nl)
ranker.post(test=False)
```
Set `FEEDGEN_HOSTNAME` and `PRIORITIZE_API_KEY` in `.env`. We use `python-dotenv` to load these.

## Usage

### Try it out!
Just run `python3 blueskyranker/ranker.py` for a demo

### Using it in your own script
Let's first run a simple ranker that, in fact, doesn't rank but just returns the output
```
from blueskyranker import TrivialRanker

ranker = TrivialRanker(returnformat='id')

rankedposts = ranker.rank(data)
```
Here, `data` is either a polars dataframe or a list of dicts with the Bluesky-posts you want to rank.

This ranker is set up such that it just returns the ids of the ranked posts. Alternatively, you can get the full posts, either as list of dicts, or as a polars dataframe by using `returnformat='dicts'`, or `returnformat='dataframe`.

You can also rank by popularity:
```
from blueskyranker import PopularityRanker
ranker = PopularityRanker(returnformat='dicts', metric= "reply_count", descending=True)  # or "like_count", etc.
```

Finally, and most importantly, you can implement much more advanced rankers, like this one, that clusters all posts, and then ranks posts such that posts from much-engaged clusters (!) are ranked higher. (DETAILED DESCRIPTION TO FOLLOW)

Priority semantics: our feed API treats LOWER numeric values as higher priority (priority 0 is highest). The rankers therefore keep the top item first. Use `descending=True` when you want the “most important” items (e.g., highest engagement/topic) at the top of the list.

```
from blueskyranker import TopicRanker
    
ranker1 = TopicRanker(returnformat='dataframe', method = 'networkclustering-tfidf', descending=True)
ranker2 = TopicRanker(returnformat='dataframe', method = 'networkclustering-count', descending=True)
# SBERT is slower but higher quality on semantics; consider smaller batches
ranker3 = TopicRanker(returnformat='dataframe', method = 'networkclustering-sbert', descending=True)
```

Advanced parameters:
- `similarity_threshold` (float, default 0.2): raise for fewer/tighter clusters.
- `vectorizer_stopwords` ('english' | list[str] | None): stopwords for TF‑IDF/Count vectorizers.

If you then want to post the ranked posts to a server, you can --- after having called `.rank()` simply call `.post()`:
```
ranker3.post(test=False)
```
**For this to work, you need to edit the file `blueskyranker/.env` in and add the server address (without https://) and the API-key. **

We use `python-dotenv` to load these values; make sure it is installed (`pip install -r requirements.txt`).

### Fetch and rank per handle
You can fetch recent public posts, then run the topic ranker separately per handle. For example:

```
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

```
python blueskyranker/fetcher.py --max-age-days 7 --sqlite-path newsflows.db
```

If you need CSVs (for exploration or interoperability), export them from the DB in a short Python snippet:

```
from blueskyranker.fetcher import ensure_db, export_db_to_csv
conn = ensure_db('newsflows.db')
paths = export_db_to_csv(conn, output_dir='.', include_combined=True)
print(paths)
```

This writes per-handle CSVs like `news-flows-nl_bsky_social_author_feed.csv` and a combined CSV `all_handles_author_feed.csv`.

### Load posts from SQLite into Polars
You can work directly from the database without exporting to CSV:

```
import polars as pl
from blueskyranker.fetcher import ensure_db, load_posts_df

conn = ensure_db('newsflows.db')
df_nl = load_posts_df(conn, handle='news-flows-nl.bsky.social', limit=2000, order_by='createdAt', descending=False)
print(df_nl.head())
```

### Migrate existing CSVs into SQLite
If you have older CSVs from previous runs, you can import them into the database (upsert by `uri`) to keep history and let future runs refresh engagement counts:

```
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

## End‑to‑end demo
Check out `example.ipynb` to see how we first download the data and then rank it!

## Cluster Report

You can generate a per-handle topic report (top clusters with keywords, sizes, engagement, and sample headlines) straight from SQLite.

CLI
```
python -m blueskyranker.cluster_report --db newsflows.db --output cluster_report.md \
  --method networkclustering-sbert --sample-max 300 --similarity-threshold 0.2 --stopwords english
```

Programmatic
```
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
- Priority semantics: the feed API treats LOWER numbers as higher priority (0 is highest). Rankers therefore output the most important item first; use `descending=True` to put strongest items at the top.

### Assumptions
- Public AppView endpoints are reachable; engagement counts may drift between runs.
- Posts can be multilingual; clustering quality improves with language-aware preprocessing and stopwords.
- Embedded link metadata (title/description/uri) may be missing. Empty strings are normalized to NULL in SQLite. The fetcher flags any empty-string anomalies in its final report (these indicate an upstream embed extraction issue).

## To‑Do / Roadmap

- Language detection + stopwording to improve topical purity in multilingual streams.
- Optional GPU acceleration / batching tips for SBERT; add README notes.
- Small CLI for DB queries (top engaged per handle, last N posts, etc.).
- More unit tests around fetch pagination and embed extraction.
- Expand example notebook to showcase SQLite workflows and SBERT best practices.
