# blueskyranker
This is just a small building block for larger NEWSFLOWS project.

Very, very much work in progress. 

Idea: take a list of bluesky posts and re-rank them.

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

If you then want to post the ranked posts to a server, you can --- after having called `.rank()` simply call `.post()`:
```
ranker3.post(test=False)
```
**For this to work, you need to edit the file `blueskyranker/.env` in and add the server address (without https://) and the API-key. **

We use `python-dotenv` to load these values; make sure it is installed (`pip install -r requirements.txt`).

### Fetch and rank per handle
You can fetch recent public posts, then run the topic ranker separately per handle. For example:

```
import polars as pl
from blueskyranker.fetcher import Fetcher
from blueskyranker.ranker import TopicRanker

# Download recent public posts for default handles
fetcher = Fetcher()
print(fetcher.fetch())

# Rank per handle using TF–IDF topical clustering
df_nl = pl.read_csv('news-flows-nl_bsky_social_author_feed.csv')
ranker = TopicRanker(returnformat='dataframe', method='networkclustering-tfidf', descending=True)
ranking_nl = ranker.rank(df_nl)
print(ranking_nl.head())
```

### Storing data: CSV or SQLite
By default, fetched posts are appended to per-handle CSVs (with de-duplication by `uri`) and a combined CSV. Alternatively, you can store everything in a SQLite database and upsert rows by `uri` (so engagement metrics are refreshed instead of duplicated).

Use the CLI to choose the storage backend:

```
# CSV (default)
python blueskyranker/fetcher.py --max-age-days 7 --storage csv

# SQLite (upsert-by-uri)
python blueskyranker/fetcher.py --max-age-days 7 --storage sqlite --sqlite-path newsflows.db
```

In SQLite mode, posts are written to a `posts` table with `uri` as the primary key. Re-fetching upserts updated engagement counts and other metadata.

## Demo of the whole pipeline
Check out  `example.ipynb` to see how we first download the data and then rank it!


## Detailed Documentation

(to follow)

## References

(to follow)
