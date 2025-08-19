# blueskyranker
This is just a small building block for larger NEWSFLOWS project.

Very, very much work in progress. 

Idea: take a list of bluesky posts and re-rank them.

## Usage

### Try it out!
Just run `python3 blueskyranker.py` for a demo

### Using it in your own script
Let's first run a simple ranker that, in fact, doesn't rank but just returns the output
```
from blueskyranker import TrivialRanker

ranker = Trivialranker(returnformat='id')

rankedposts = rankder.rank(data)
```
Here, `data` is either a polars dataframe or a list of dicts with the Bluesky-posts you want to rank.

This ranker is set up such that it just returns the ids of the ranked posts. Alternatively, you can get the full posts, either as list of dicts, or as a polars dataframe by using `returnformat='dicts'`, or `returnformat='dataframe`.

You can also rank by popularity:
```
from blueskyranker imort PopularityRanker
ranker = PopularityRanker(returnformat='dicts', metric= "reply_count")  # you can also select metrics like "like_count" etc.
```

Finally, and most importantly, you can implement much more advanced rankers, like this one, that clusters all posts, and then ranks posts such that posts from much-engaged clusters (!) are ranked higher. (DETAILED DESCRIPTION TO FOLLOW)

```
from blueskyranker import TopicRanker
    
ranker1 = TopicRanker(returnformat='dataframe', method = 'networkclustering-tfidf')
ranker2 = TopicRanker(returnformat='dataframe', method = 'networkclustering-count')
# the following one is very slow and not recommended unless you have a GPU or very few documents
ranker3 = TopicRanker(returnformat='dataframe', method = 'networkclustering-sbert')
```


## Demo of the whole pipeline
1. Fetch post and engagement data (by running `./fetch_bsky_auth-r_feeds_progress.py`)
2. Play with the rankings in `example.ipynb`


## Detailed Documentation

(to follow)

## References

(to follow)
