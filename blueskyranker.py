#!/usr/bin/env python3
import polars as pl
from typing import Literal
from itertools import zip_longest

import logging
logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
    #datefmt='%Y-%m-%d %H:%M:%S')
    datefmt='%H:%M:%S')


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
    def __init__(self, 
            returnformat: Literal["id","dicts","dataframe"],
            method: Literal['networkclustering-tfidf', 'networkclustering-count', 'networkclustering-sbertf'],
            metric: Literal['like_count', 'quote_count',  'reply_count',  'repost_count', 'engagement'] = 'engagement'):
        self.required_keys = {'cid', 'indexed_at', 'like_count',  'news_description',  'news_title',  'news_uri',  'quote_count',  'reply_count',  'repost_count',  'text',  'uri'} 
        self.returnformat = returnformat
        if metric != 'engagement':
            raise NotImplementedError
        self.metric = metric
        self.method = method
    
    def _cluster(self, data: pl.DataFrame, threshold = .2, similarity: Literal['tfidf-cosine', 'count-cosine', 'sbert-cosine'] = 'tfidf-cosine'):
        """This function clusters the texts using the Leiden Algorithm by Traag, Waltman, & Van Eck (2019),
        See also suggestions by Trilling & Van Hoof (2020) for  news event clustering.
        It adds a column "cluster" and a column "clustersize" to the dataframe.
        """
        logging.debug("Creating cosine similarity matrix...")
        if similarity =='tfidf-cosine':
            vectorizer = TfidfVectorizer()   
            bow = vectorizer.fit_transform(data['text'])
            sim_matrix = cosine_similarity(bow)
        elif similarity =='count-cosine':
            vectorizer = CountVectorizer()   
            bow = vectorizer.fit_transform(data['text'])
            sim_matrix = cosine_similarity(bow)
        elif similarity == 'sbert-cosine':
            # import here to avoid forcing everyone to install this huge library
            from sentence_transformers import SentenceTransformer
            sbert_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2',  model_kwargs={"torch_dtype": "float16"})
            embeddings = sbert_model.encode(data['text'], show_progress_bar=True)
            sim_matrix = cosine_similarity(embeddings)
        else:
            raise NotImplementedError(f"Simiarity {similarity} is not implemented")

        logging.debug(f"Removing all entries below a threshold of {threshold}")
        filtered_matrix = np.where(sim_matrix >= threshold, sim_matrix, 0)
        sparsity = 1.0 -(np.count_nonzero(filtered_matrix) / float(filtered_matrix.size) )
        logging.debug(f"The new matrix is {sparsity:.2%} sparse")
        logging.debug("Creating a graph")
        g = ig.Graph.Weighted_Adjacency(filtered_matrix.tolist(), mode="UNDIRECTED", attr="weight")
        g.vs["cid"] = data['cid']
        logging.debug("Apply network clustering using the Leiden Algorithm")
        # The Surprise didn't work too well on these data, resort to Modularity:
        # part = leidenalg.find_partition(g, leidenalg.SurpriseVertexPartition, weights='weight')
        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights='weight')
        partitions = []
        for subgraph in part.subgraphs():
            partitions.append([node['cid'] for node in subgraph.vs])
        logging.debug(f"The {len(data)} items were grouped into {len(partitions)} clusters.")

        clusterassignment = [[article, cluster] for cluster, articles in enumerate(partitions) for article in articles]
        clusterassignment_df = pl.DataFrame(clusterassignment,orient='row', schema=["cid","cluster"])
        #clusterassignment_df = clusterassignment_df.join(clusterassignment_df.group_by('cluster').len(), on='cluster').rename({'len':'clustersize'})
        data = data.join(clusterassignment_df, on='cid')
        data = self._add_cluster_stats(data)
        return data

    def _add_cluster_stats(self, df_with_clusters):
        """takes dataframe with cluster labels and adds cluster statistics"""
        clusterstats = df_with_clusters.group_by('cluster').agg(pl.col('like_count').sum(),
            pl.col('reply_count').sum(),
            pl.col('quote_count').sum(),
            pl.col('repost_count').sum(),
            pl.len())
        # TODO: MAYBE WEIGH THIS, SUCH THAT A LIKE COUNTS LESS THAN A REPLY?
        clusterstats = clusterstats.with_columns(sum=pl.sum_horizontal("reply_count", "like_count",'repost_count','quote_count'))
        clusterstats = clusterstats.with_columns(engagement_rank=clusterstats["sum"].rank(method='random', descending=True)).rename({'sum':'engagement_count', 'len':'size'})
        clusterstats = clusterstats.rename(lambda x: f"cluster_{x}").rename({"cluster_cluster": "cluster"})
        df_with_clusters = df_with_clusters.join(clusterstats, on="cluster")
        return df_with_clusters
        
        
    def rank(self, data: list[dict] | pl.DataFrame) -> list[dict] | pl.DataFrame | list[str]:
        """This ranker clusters the input by topic, and then creates a feedback loop by
        upranking the most popular topic"""
        df = self._transform_input(data)

        if self.method == 'networkclustering-tfidf':
            df_with_clusters = self._cluster(df, similarity='tfidf-cosine')
        if self.method == 'networkclustering-count':
            df_with_clusters = self._cluster(df, similarity='count-cosine')
        if self.method == 'networkclustering-sbert':
            if len(data)>100:
                logger.warning(f"Do you really want to do this? You have {len(data)} texts, calculating sentence embeddings will be REALLY slow")
                logger.warning(f"Consider using another method, or submitting less document")
            df_with_clusters = self._cluster(df, similarity='sbert-cosine')
              
        # OPTION 1
        # This here would be a very simplistic ranking, we simply rank by cluster size
        # Hence, the most published about topic gets more popular.
        # df_with_clusters = df_with_clusters.sort(by='clustersize', descending=True)
        # return self._transform_output(df_with_clusters)

        # OPTION2
        # But we do sth more fancy:
        # We now return the articles sorted by the cluster engagement rank

        df_with_clusters = df_with_clusters.sort(by='cluster_engagement_rank')
        
        # TODO: Implement saturation here, such that we do not have just the same things from the same cluster
        # TODO: (iteratively construct feed accounting for a topic saturation penalty (if post is already covered; see example below).)

        #a = ['a1','a2']
        #b = ['b1','b2','b3']
        #c = ['c1']
        # [item for cluster in zip_longest(a,b,c) for item in cluster if item is not None]
        # gives: ['a1', 'b1', 'c1', 'a2', 'b2', 'b3']

        # AND NOW THIS FOR OUR DATA
        # We take the first article from the most-engaged cluster, then the first article from the second-most-engaged-cluster, then from the third, etc.
        # If we exausted, we start from the beginning
        # If a cluster is exausted, we just skip it
        # (see toy example abouve)
        list_of_clusters_gen = (cluster.rows(named=True)  for _, cluster in df_with_clusters.group_by('cluster_engagement_rank', maintain_order=True))
        fancyranking = [item for cluster in zip_longest(*list_of_clusters_gen) for item in cluster if item is not None]
        final_ranking = pl.DataFrame(fancyranking)

        return self._transform_output(final_ranking)


def sampledata(filename="example_news.csv"):
    """provides sample data for offline testing"""
    data = pl.read_csv(filename).to_dicts()
    return data




if __name__=="__main__":
    print("Testing the bluesky ranker...")
    data = sampledata()
    #ranker = TrivialRanker(returnformat='id')
    #ranker2 = PopularityRanker(returnformat='dicts', metric= "reply_count")
    ranker3 = TopicRanker(returnformat='dataframe', method = 'networkclustering-tfidf')
    #ranker4 = TopicRanker(returnformat='dataframe', method = 'networkclustering-sbert')

    #print(ranker.rank(data)[:10])
    #print(ranker2.rank(data)[:10])
    print(ranker3.rank(data)[:10])

    # SBERT IS TOO SLOW TO RUN ON 1000 DOCS, we only do less
    #print(ranker4.rank(data[:40])[:10])


