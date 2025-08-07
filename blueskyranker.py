import polars as pl
from typing import Literal
import logging
logging.basicConfig(level=logging.DEBUG)

# for topic ranker:
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import igraph as ig
import leidenalg



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
    def __init__(self, returnformat: Literal["id","dicts","dataframe"], metric: Literal['like_count', 'quote_count',  'reply_count',  'repost_count'] ):
        self.required_keys = {'cid', 'like_count', 'quote_count',  'reply_count',  'repost_count'} 
        self.returnformat = returnformat
        self.metric = metric

    def rank(self, data: list[dict] | pl.DataFrame) -> list[dict] | pl.DataFrame | list[str]:
        """This ranker just keeps the order of the input"""
        df = self._transform_input(data)
        df.sort(by=self.metric, descending=True)
        return self._transform_output(df)


class TopicRanker(_BaseRanker):
    def __init__(self, returnformat: Literal["id","dicts","dataframe"], metric: Literal['quote_count',  'reply_count',  'repost_count'] ):
        self.required_keys = {'cid', 'indexed_at', 'like_count',  'news_description',  'news_title',  'news_uri',  'quote_count',  'reply_count',  'repost_count',  'text',  'uri'} 
        self.returnformat = returnformat
        self.metric = metric
    
    def _cluster(self, data: pl.DataFrame, threshold = .2):
        """This function clusters the texts using the Leiden Algorithm by Traag, Waltman, & Van Eck (2019),
        following the approach Trilling & Van Hoof (2020) suggest news event clustering.
        It adds a column "cluster" to the dataframe.
        """
        logging.debug("Creating cosine similarity matrix...")
        vectorizer = TfidfVectorizer()   # Or, if wanted, CountVectorizer
        bow = vectorizer.fit_transform(data['text'])
        cosine_sim_matrix = cosine_similarity(bow)
        logging.debug(f"Removing all entries below a threshold of {threshold}")
        filtered_matrix = np.where(cosine_sim_matrix >= threshold, cosine_sim_matrix, 0)
        sparsity = 1.0 -(np.count_nonzero(filtered_matrix) / float(filtered_matrix.size) )
        logging.debug(f"The new matrix is {sparsity:.2%} sparse")
        logging.debug("Creating a graph")
        g = ig.Graph.Weighted_Adjacency(filtered_matrix.tolist(), mode="UNDIRECTED", attr="weight")
        g.vs["cid"] = data['cid']
        logging.debug("Apply network clustering using the Leiden Algorithm")
        part = leidenalg.find_partition(g, leidenalg.SurpriseVertexPartition)
        partitions = []
        for subgraph in part.subgraphs():
            partitions.append([node['cid'] for node in subgraph.vs])
        logging.debug(f"The {len(data)} items were grouped into {len(partitions)} clusters.")

        clusterassignment = [[article, cluster] for cluster, articles in enumerate(partitions) for article in articles]
        clusterassignment_df = pl.DataFrame(clusterassignment,orient='row', schema=["cid","cluster"])
        data = data.join(clusterassignment_df, on='cid')
        
        return data
        
    def rank(self, data: list[dict] | pl.DataFrame) -> list[dict] | pl.DataFrame | list[str]:
        """This ranker just keeps the order of the input"""
        df = self._transform_input(data)
        clusters = self._cluster(df)
        # TODO THERE IS NO RANKING YET, WE JUST RETURN THE ADDED CLUSTER
        
        return self._transform_output(clusters)


def sampledata(filename="example_news.csv"):
    """provides sample data for offline testing"""
    data = pl.read_csv(filename).to_dicts()
    return data




if __name__=="__main__":
    print("Testing the bluesky ranker...")
    data = sampledata()
    #ranker = TrivialRanker(returnformat='id')
    #ranker2 = PopularityRanker(returnformat='dicts', metric= "reply_count")
    ranker3 = TopicRanker(returnformat='dataframe', metric= None)

    #print(ranker.rank(data)[:10])
    #print(ranker2.rank(data)[:10])
    print(ranker3.rank(data)[:10])

