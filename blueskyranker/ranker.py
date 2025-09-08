#!/usr/bin/env python3
import polars as pl
from typing import Literal
from itertools import zip_longest
from dotenv import load_dotenv
import os
import requests



import logging
logger = logging.getLogger('BSRlog')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    #datefmt='%Y-%m-%d %H:%M:%S')
    datefmt='%H:%M:%S')
logger.setLevel(logging.DEBUG)

# for topic ranker:
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Delay heavy imports (igraph, leidenalg) until needed in _cluster

load_dotenv(".env")


class _BaseRanker():
    """Base class for rankers.

    - returnformat: one of 'id' | 'dicts' | 'dataframe'
    - descending: when True, rankers should place the most important items FIRST.
      The post() method preserves row order and assigns priority 0 to the first row
      (lower numbers = higher priority on the feed API).
    """
    def __init__(self, returnformat: Literal["id","dicts","dataframe"], metric=None, descending = False):
        self.required_keys = None
        self.returnformat = returnformat
        self.metric = metric
        self.descending = descending  # True: 1=highest rank, False: 1=lowest rank
        self.ranking = None # will only be populated after .rank() is called (think of .fit() in scikit-learn)

    def _transform_input(self, data) -> pl.DataFrame:
        """not fully implemented yet, ensure that we can process dataframes but also list of dicts """
        if type(data) is list:
            if self.required_keys is not None:
                assert self.required_keys.issubset(data[0].keys()), f"Not all required keys ({self.required_keys} are in the dataset (present keys: {data[0].keys()}). Missing keys: {set(self.required_keys) - set(data[0].keys())}" 
            return pl.from_dicts(data)
        elif type(data) is pl.DataFrame:
            if self.required_keys is not None:
                assert self.required_keys.issubset(data.columns), (
                    f"Not all required keys ({self.required_keys}) are in the dataset "
                    f"(present columns: {data.columns}). Missing keys: "
                    f"{set(self.required_keys) - set(data.columns)}"
                )
            return data
        else:
            raise ValueError("Data format not supported")

    def _transform_output(self, data):
        # Do not change order here. Rankers must output final order
        # (top row = highest priority = priority index 0).
        if self.returnformat == 'id':
            return data['uri'].to_list()
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
        self.ranking = None  
        raise NotImplementedError
        

    def post(self, test = True,):
        """Send the ranking to the server that generates new feeds
        Essentially a port of https://github.com/JBGruber/newsflows-bsky-feed-generator/blob/main/scripts/prioritise.r
        
        Parameters:
        test (bool): If True, only does a test run without changing the database
        """
        server = f"https://{os.getenv('FEEDGEN_HOSTNAME', 'localhost:3020')}"
        headers = {"api-key": os.getenv("PRIORITIZE_API_KEY")}
        params = {"test": str(test).lower()}

        assert self.ranking is not None, "You need to call .rank() first to rank the posts before you can post them"
        
        post_list = self.ranking [['uri']].with_row_index().rename({'index':'priority'}).rows(named=True)
        logger.debug(f"Sending this post_list:\n{post_list}")
        resp = requests.post(f"{server}/api/prioritize", headers=headers, params=params, json=post_list)
        if resp.status_code==200:
            logger.debug(resp)
            logger.debug(resp.text)
            return True
        else:
            logger.error(resp)
            logger.error(resp.text)
            return False

        

class TrivialRanker(_BaseRanker):
    def rank(self, data: list[dict] | pl.DataFrame) -> list[dict] | pl.DataFrame | list[str]:
        """This ranker just keeps the order of the input"""
        df = self._transform_input(data)
        self.ranking = df
        return self._transform_output(df)


class PopularityRanker(_BaseRanker):
    def __init__(self, returnformat: Literal["id","dicts","dataframe"], metric: Literal['like_count', 'quote_count',  'reply_count',  'repost_count'], descending: bool ):
        self.required_keys = {'uri', 'cid', 'like_count', 'quote_count',  'reply_count',  'repost_count'} 
        self.returnformat = returnformat
        self.metric = metric
        self.descending = descending
        self.ranking = None # will only be populated after .rank() is called (think of .fit() in scikit-learn)


    def rank(self, data: list[dict] | pl.DataFrame) -> list[dict] | pl.DataFrame | list[str]:
        """Sorts by the selected engagement metric.

        With descending=True, highest metric appears first (priority 0).
        """
        df = self._transform_input(data)
        df = df.sort(by=self.metric, descending=self.descending)  # returns new DataFrame
        self.ranking = df
        return self._transform_output(df)


class TopicRanker(_BaseRanker):
    """Topic-driven ranker using text similarity + Leiden clustering.

    method:
      - 'networkclustering-tfidf'  : cosine over TF–IDF vectors
      - 'networkclustering-count'  : cosine over raw counts
      - 'networkclustering-sbert'  : cosine over SBERT embeddings

    descending=True will prioritize posts from the most engaged clusters first.
    """
    def __init__(self, 
            returnformat: Literal["id","dicts","dataframe"],
            method: Literal['networkclustering-tfidf', 'networkclustering-count', 'networkclustering-sbert'],
            descending: bool,
            metric: Literal['like_count', 'quote_count',  'reply_count',  'repost_count', 'engagement'] = 'engagement',
            similarity_threshold: float = 0.2,
            vectorizer_stopwords: str | list[str] | None = None,
            cluster_window_days: int | None = None,
            engagement_window_days: int | None = None,
            push_window_days: int | None = None,
        ):
        self.required_keys = {'uri', 'cid', 'like_count',  'news_description',  'news_title',  'news_uri',  'quote_count',  'reply_count',  'repost_count',  'text',  'uri', 'createdAt'} 
        self.returnformat = returnformat
        if metric != 'engagement':
            raise NotImplementedError
        self.metric = metric
        self.method = method
        self.descending = descending
        self.similarity_threshold = similarity_threshold
        self.stopwords = vectorizer_stopwords
        self.cluster_window_days = cluster_window_days
        self.engagement_window_days = engagement_window_days
        self.push_window_days = push_window_days
        self.ranking = None # will only be populated after .rank() is called (think of .fit() in scikit-learn)

    
    def _cluster(self, data: pl.DataFrame, similarity: Literal['tfidf-cosine', 'count-cosine', 'sbert-cosine'] = 'tfidf-cosine'):
        """This function clusters the texts using the Leiden Algorithm by Traag, Waltman, & Van Eck (2019),
        See also suggestions by Trilling & Van Hoof (2020) for  news event clustering.
        It adds a column "cluster" and a column "clustersize" to the dataframe.
        """
        logger.debug("Creating cosine similarity matrix...")
        if similarity =='tfidf-cosine':
            vectorizer = TfidfVectorizer(stop_words=self.stopwords)   
            bow = vectorizer.fit_transform(data['text'])
            sim_matrix = cosine_similarity(bow)
        elif similarity =='count-cosine':
            vectorizer = CountVectorizer(stop_words=self.stopwords)   
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

        logger.debug(f"Removing all entries below a threshold of {self.similarity_threshold}")
        filtered_matrix = np.where(sim_matrix >= self.similarity_threshold, sim_matrix, 0)
        sparsity = 1.0 -(np.count_nonzero(filtered_matrix) / float(filtered_matrix.size) )
        logger.debug(f"The new matrix is {sparsity:.2%} sparse")
        logger.debug("Creating a graph")
        # Lazy import to avoid hard dependency during tests or when stubbing clustering
        import igraph as ig
        import leidenalg
        g = ig.Graph.Weighted_Adjacency(filtered_matrix.tolist(), mode="UNDIRECTED", attr="weight")
        g.vs["cid"] = data['cid']
        logger.debug("Apply network clustering using the Leiden Algorithm")
        # The Surprise didn't work too well on these data, resort to Modularity:
        # part = leidenalg.find_partition(g, leidenalg.SurpriseVertexPartition, weights='weight')
        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights='weight')
        partitions = []
        for subgraph in part.subgraphs():
            partitions.append([node['cid'] for node in subgraph.vs])
        logger.debug(f"The {len(data)} items were grouped into {len(partitions)} clusters.")

        clusterassignment = [[article, cluster] for cluster, articles in enumerate(partitions) for article in articles]
        clusterassignment_df = pl.DataFrame(clusterassignment,orient='row', schema=["cid","cluster"])
        #clusterassignment_df = clusterassignment_df.join(clusterassignment_df.group_by('cluster').len(), on='cluster').rename({'len':'clustersize'})
        data = data.join(clusterassignment_df, on='cid')
        data = self._add_cluster_stats(data)
        return data

    def _add_cluster_stats(self, df_with_clusters, stats_subset: pl.DataFrame | None = None):
        """Takes dataframe with cluster labels and adds cluster statistics.

        engagement_rank is computed so that 1 = highest engagement within the dataset,
        independent of the outer 'descending' flag.
        """
        src = stats_subset if stats_subset is not None else df_with_clusters
        clusterstats = src.group_by('cluster').agg(pl.col('like_count').sum(),
            pl.col('reply_count').sum(),
            pl.col('quote_count').sum(),
            pl.col('repost_count').sum(),
            pl.len())
        # TODO: MAYBE WEIGH THIS, SUCH THAT A LIKE COUNTS LESS THAN A REPLY?
        clusterstats = clusterstats.with_columns(sum=pl.sum_horizontal("reply_count", "like_count",'repost_count','quote_count'))
        # Lower engagement_rank means higher engagement; independent of outer descending
        clusterstats = clusterstats.with_columns(
            engagement_rank=clusterstats["sum"].rank(method='random', descending=True)
        ).rename({'sum':'engagement_count', 'len':'size'}) 
        clusterstats = clusterstats.rename(lambda x: f"cluster_{x}").rename({"cluster_cluster": "cluster"})
        df_with_clusters = df_with_clusters.join(clusterstats, on="cluster")
        return df_with_clusters
        
        
    def rank(self, data: list[dict] | pl.DataFrame) -> list[dict] | pl.DataFrame | list[str]:
        """This ranker clusters the input by topic, and then creates a feedback loop by
        upranking the most popular topic"""
        from datetime import datetime, timedelta, timezone
        df = self._transform_input(data)

        # Parse createdAt to datetime for windowing
        if 'createdAt_dt' not in df.columns:
            df = df.with_columns(createdAt_dt=pl.col('createdAt').str.strptime(pl.Datetime, strict=False))

        now = datetime.now(timezone.utc)
        # Determine effective cluster window so every later subset is covered
        windows = [w for w in [self.cluster_window_days, self.engagement_window_days, self.push_window_days] if w is not None]
        effective_cluster_days = max(windows) if windows else None
        if effective_cluster_days is not None:
            cluster_cutoff = now - timedelta(days=int(effective_cluster_days))
            df_cluster_base = df.filter(pl.col('createdAt_dt') >= cluster_cutoff)
        else:
            df_cluster_base = df

        if self.method == 'networkclustering-tfidf':
            df_with_clusters = self._cluster(df_cluster_base, similarity='tfidf-cosine')
        if self.method == 'networkclustering-count':
            df_with_clusters = self._cluster(df_cluster_base, similarity='count-cosine')
        if self.method == 'networkclustering-sbert':
            if len(df_cluster_base)>100:
                logger.warning(f"Do you really want to do this? You have {len(df_cluster_base)} texts, calculating sentence embeddings will be REALLY slow")
                logger.warning(f"Consider using another method, or submitting less document")
            df_with_clusters = self._cluster(df_cluster_base, similarity='sbert-cosine')
              
        # OPTION 1
        # This here would be a very simplistic ranking, we simply rank by cluster size
        # Hence, the most published about topic gets more popular.
        # df_with_clusters = df_with_clusters.sort(by='clustersize', descending=self.descending)
        # return self._transform_output(df_with_clusters)

        # OPTION2
        # But we do sth more fancy:
        # We now return the articles sorted by the cluster engagement rank

        # With engagement_rank defined as 1 = highest engagement, we put
        # most-engaged clusters first when descending=True.
        # Compute engagement stats over an optional engagement window (subset of clustered rows)
        if self.engagement_window_days is not None:
            engagement_cutoff = now - timedelta(days=int(self.engagement_window_days))
            stats_subset = df_with_clusters.filter(pl.col('createdAt_dt') >= engagement_cutoff)
        else:
            stats_subset = None
        df_with_clusters = self._add_cluster_stats(df_with_clusters, stats_subset=stats_subset)

        df_with_clusters = df_with_clusters.sort(
            by='cluster_engagement_rank', descending=not self.descending
        )
        
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

        # Optionally restrict the output to a push window
        if self.push_window_days is not None:
            push_cutoff = now - timedelta(days=int(self.push_window_days))
            final_ranking = final_ranking.filter(pl.col('createdAt_dt') >= push_cutoff)

        self.ranking = final_ranking
        return self._transform_output(final_ranking)


def sampledata(filename: str | None = None):
    """Provides sample data for offline testing.

    If filename is None, loads the bundled CSV next to this module.
    """
    from pathlib import Path
    if filename is None:
        filename = str(Path(__file__).with_name('example_news.csv'))
    data = pl.read_csv(filename).to_dicts()
    return data




if __name__=="__main__":
    print("Testing the bluesky ranker...")
    data = sampledata()
    ranker = TrivialRanker(returnformat='id', descending=True)
    #ranker2 = PopularityRanker(returnformat='dicts', metric= "reply_count")
    ranker3 = TopicRanker(returnformat='dataframe', method = 'networkclustering-tfidf', descending=True)
    #ranker4 = TopicRanker(returnformat='dataframe', method = 'networkclustering-sbert')

    print(ranker.rank(data)[:10])
    #print(ranker2.rank(data)[:10])
    # SBERT IS TOO SLOW TO RUN ON 1000 DOCS, we only do less
    #print(ranker4.rank(data[:40])[:10])
    
    ranking = ranker3.rank(data)
    print("Printing the results in the chosen format (dataframe (full dataframe), dicts (list of dicts for generic use), or id only (for just sending the ID to a server)")
    print("Printing only the first 10 items")
    print()
    print(ranking[:10])

    ranker3_reverse = TopicRanker(returnformat='dataframe', method = 'networkclustering-tfidf', descending=False)
    ranking = ranker3_reverse.rank(data)
    print(ranking[-10:]) # show LAST 10 in this case

    ranker3.post()
