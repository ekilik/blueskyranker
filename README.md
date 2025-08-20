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

ranker = Trivialranker(returnformat='id')

rankedposts = ranker.rank(data)
```
Here, `data` is either a polars dataframe or a list of dicts with the Bluesky-posts you want to rank.

This ranker is set up such that it just returns the ids of the ranked posts. Alternatively, you can get the full posts, either as list of dicts, or as a polars dataframe by using `returnformat='dicts'`, or `returnformat='dataframe`.

You can also rank by popularity:
```
from blueskyranker import PopularityRanker
ranker = PopularityRanker(returnformat='dicts', metric= "reply_count")  # you can also select metrics like "like_count" etc.
```

Finally, and most importantly, you can implement much more advanced rankers, like this one, that clusters all posts, and then ranks posts such that posts from much-engaged clusters (!) are ranked higher. (DETAILED DESCRIPTION TO FOLLOW)

We use descending=False to be compatible with the prioritzie-Endpoint to which we want to posts, which expectes *higher* numbers to get more priority.

```
from blueskyranker import TopicRanker
    
ranker1 = TopicRanker(returnformat='dataframe', method = 'networkclustering-tfidf', descending=False)
ranker2 = TopicRanker(returnformat='dataframe', method = 'networkclustering-count', descending=False)
# the following one is very slow and not recommended unless you have a GPU or very few documents
ranker3 = TopicRanker(returnformat='dataframe', method = 'networkclustering-sbert', descending=False)
```

If you then want to post the ranked posts to a server, you can --- after having called `.rank()` simply call `.post()`:
```
ranker3.post(test=False)
```
**For this to work, you need to edit the file `blueskyranker/.env` in and add the server address (without https://) and the API-key. **

## Demo of the whole pipeline
Check out  `example.ipynb` to see how we first download the data and then rank it!


## Detailed Documentation

(to follow)

## References

(to follow)
