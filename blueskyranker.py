import polars as pl
from typing import Literal

class _BaseRanker():
    def __init__(self, returnformat: Literal["id","dicts","dataframe"], metric=None):
        self.required_keys = None
        self.returnformat = returnformat
        self.metric = metric

    def _transform_input(self, data) -> pl.DataFrame:
        """not fully implemented yet, ensure that we can process dataframes but also list of dicts """
        if type(data) is list:
            if self.required_keys is not None:
                assert self.required_keys.issubset(data[0].keys()) 
            return pl.from_dicts(data)
        elif type(data) is pl.DataFrame:
            if self.required_keys is not None:
                assert self.required_keys.issubset(data.keys()) 
            return data
        else:
            raise ValueError("Data format not supported")


    def _transform_output(self, data):
        if self.returnformat == 'id':
            return data['cid'].to_list()
        elif self.returnformat == "dicts":
            return data.to_dicts()
        elif self.returnformat == "dataframe":
            return data
        else:
            raise NotImplementedError


    def rank(self, data: list[dict] | pl.DataFrame) -> list[dict] | pl.DataFrame | list[str]:
        """The rank method takes a set of bluesky posts (as list of dicts), and returns them in a ranked order.
        Overwrite this method with a function that implements your ranking algorithm"""
        # TODO: Determine wether only the ID or the whole ranking should be returned
        raise NotImplementedError


class TrivialRanker(_BaseRanker):
    def rank(self, data: list[dict] | pl.DataFrame) -> list[dict] | pl.DataFrame | list[str]:
        """This ranker just keeps the order of the input"""
        df = self._transform_input(data)
        return self._transform_output(df)


class PopularityRanker(_BaseRanker):
    def __init__(self, returnformat: Literal["id","dicts","dataframe"], metric: Literal['quote_count',  'reply_count',  'repost_count'] ):
        self.required_keys = {'cid', 'indexed_at', 'like_count',  'news_description',  'news_title',  'news_uri',  'quote_count',  'reply_count',  'repost_count',  'text',  'uri'} 
        self.returnformat = returnformat

    def rank(self, data: list[dict] | pl.DataFrame) -> list[dict] | pl.DataFrame | list[str]:
        """This ranker just keeps the order of the input"""
        df = self._transform_input(data)
        df.sort(by="reply_count", descending=True)
        return self._transform_output(df)




def sampledata(filename="example_news.csv"):
    """provides sample data for offline testing"""
    data = pl.read_csv(filename).to_dicts()
    return data


if __name__=="__main__":
    print("Testing the bluesky ranker...")
    data = sampledata()
    ranker = TrivialRanker(returnformat='id')
    ranker2 = PopularityRanker(returnformat='dataframe', metric= "reply_count")

    print(ranker.rank(data)[:10])
    print(ranker2.rank(data)[:10])

