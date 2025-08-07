import pandas as pd


class _BaseRanker():
    def __init__(self):
        pass

    def rank(self, data: list[dict]) -> list[dict]:
        """The rank method takes a set of bluesky posts (as list of dicts), and returns them in a ranked order.
        Overwrite this method with a function that implements your ranking algorithm"""
        # TODO: Determine wether only the ID or the whole ranking should be returned
        raise NotImplementedError



class TrivialRanker(_BaseRanker):
    def rank(self, data: list[dict]) -> list[dict]:
        return data





def sampledata(filename="example_news.csv"):
    """provides sample data for offline testing"""
    data = pd.read_csv(filename).to_dict(orient='records')
    required_keys = {'cid', 'indexed_at', 'like_count',  'news_description',  'news_title',  'news_uri',  'quote_count',  'reply_count',  'repost_count',  'text',  'uri'} 
    assert required_keys.issubset(data[0].keys()) 
    return data


if __name__=="__main__":
    print("Testing the bluesky ranker...")
    data = sampledata()
    ranker = TrivialRanker()
    print(ranker.rank(data))
