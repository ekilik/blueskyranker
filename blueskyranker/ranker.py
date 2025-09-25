#!/usr/bin/env python3
"""
Topic ranking: cluster posts into topics and order them by cluster engagement.

Key principles used throughout the module:
- Time is handled in UTC; when available, `createdAt_ns` (epoch nanoseconds) is preferred
  for filtering and sorting. Human readable local timestamps are only added to exports.
- Three time windows drive behavior:
  * clustering window → build topic graph (no engagement computed here)
  * engagement window → compute cluster engagement and deterministic ranks
  * push window → eligible posts for the final, interleaved priority list
- Deterministic tie‑breaks: higher engagement first; ties broken by higher average recency,
  then by the cluster id to ensure full reproducibility.
"""
import polars as pl
from typing import Literal
from itertools import zip_longest
from dotenv import load_dotenv
import os
import requests
import json
from datetime import datetime, timezone



import logging
logger = logging.getLogger('BSRlog')

# for topic ranker:
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Delay heavy imports (igraph, leidenalg) until needed in _cluster

load_dotenv(".env")
load_dotenv("blueskyranker/.env")


class _BaseRanker():
    """Base class for rankers.

    - returnformat: one of 'id' | 'dicts' | 'dataframe'
    - descending: when True, rankers should place the most important items FIRST.
      The post() method preserves row order and assigns the HIGHEST numeric
      priority to the first row (higher numbers = higher priority on the feed API).
    """
    def __init__(self, returnformat: Literal["id","dicts","dataframe"], metric=None, descending = False):
        self.required_keys = None
        self.returnformat = returnformat
        self.metric = metric
        self.descending = descending  # True: 1=highest rank, False: 1=lowest rank
        self.ranking = None # will only be populated after .rank() is called (think of .fit() in scikit-learn)

    def _transform_input(self, data) -> pl.DataFrame:
        """Ensure we can process DataFrames and list[dict]; validate required keys; handle empty lists."""
        if isinstance(data, list):
            if not data:
                return pl.DataFrame([])
            if self.required_keys is not None and not self.required_keys.issubset(set(data[0].keys())):
                missing = set(self.required_keys) - set(data[0].keys())
                raise ValueError(f"Missing required keys: {missing}; present: {set(data[0].keys())}")
            return pl.from_dicts(data)
        elif isinstance(data, pl.DataFrame):
            if self.required_keys is not None and not self.required_keys.issubset(set(data.columns)):
                missing = set(self.required_keys) - set(data.columns)
                raise ValueError(f"Missing required columns: {missing}; present: {set(data.columns)}")
            return data
        else:
            raise ValueError("Data format not supported")

    def _transform_output(self, data):
        # Do not change order here. Rankers must output final order
        # (top row = highest priority; mapping to numeric priority happens in post()).
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
        

    def post(self, test: bool = True, handle: str | None = None, extra_uris_zero_prio: list[str] | None = None,):
        """Send the ranking to the server that generates new feeds
        Essentially a port of https://github.com/JBGruber/newsflows-bsky-feed-generator/blob/main/scripts/prioritise.r
        
        Parameters:
        test (bool): If True, only does a test run without changing the database
        """
        host = os.getenv('FEEDGEN_HOSTNAME', 'localhost:3020')
        scheme = 'http' if host.startswith('localhost') or host.startswith('127.0.0.1') else 'https'
        server = f"{scheme}://{host}"
        api_key = os.getenv("PRIORITIZE_API_KEY")
        headers = {"api-key": api_key} if api_key else {}
        params = {"test": str(test).lower()}

        if self.ranking is None:
            raise RuntimeError("You need to call .rank() first to rank the posts before you can post them")
        
        # Build priority list so that HIGHER numbers mean HIGHER priority.
        # Preserve current row order, then assign priorities starting at 1000:
        # top row → 1000, second → 999, ...
        df = self.ranking.select([pl.col('uri')]).with_row_index(name='idx')
        # Start at 1000 and clamp at minimum 1 to avoid negatives when many items
        df = df.with_columns((pl.lit(1000) - pl.col('idx')).alias('priority'))
        df = df.with_columns(
            pl.when(pl.col('priority') < 1).then(1).otherwise(pl.col('priority')).alias('priority')
        ).select(['priority','uri'])
        post_list = df.rows(named=True)
        # Payload size guard (warn only). Override via PRIORITIZE_MAX_ITEMS env.
        try:
            max_items = int(os.getenv('PRIORITIZE_MAX_ITEMS', '3000'))
        except Exception:
            max_items = 3000
        if len(post_list) > max_items:
            msg = f"Preparing to send {len(post_list)} items (> {max_items}). This may be slow or rejected by server."
            logger.warning(msg)
            print(f"[WARN] {msg}")
        # Optionally add extra URIs (typically from last run) with priority 0 to demote them
        if extra_uris_zero_prio:
            current_uris = {row['uri'] for row in post_list}
            extras = [
                {'priority': 0, 'uri': u}
                for u in extra_uris_zero_prio if u and u not in current_uris
            ]
            if extras:
                logger.debug(f"Appending {len(extras)} extra URIs with priority 0 (demote)")
                post_list.extend(extras)
        logger.debug(f"Sending this post_list:\n{post_list}")
        # Perform request and capture/print/save any server response
        try:
            resp = requests.post(f"{server}/api/prioritize", headers=headers, params=params, json=post_list, timeout=30)
        except requests.exceptions.Timeout as e:
            logger.error(f"POST to {server}/api/prioritize timed out: {e}")
            print(f"[ERROR] POST timed out: {e}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"POST to {server}/api/prioritize failed: {e}")
            print(f"[ERROR] POST failed: {e}")
            return False

        # Prepare readable filename components
        safe_handle = (handle or 'unknown').replace('.', '_').replace(' ', '_')
        ts_readable = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

        # Try to parse as JSON; fall back to text
        resp_json = None
        resp_text = None
        try:
            resp_json = resp.json()
            content_for_length = json.dumps(resp_json, ensure_ascii=False)
        except Exception:
            resp_text = resp.text or ''
            content_for_length = resp_text

        # Decide whether to print inline or save to file (long output)
        is_long = len(content_for_length) > 2000
        saved_path = None
        if is_long:
            try:
                os.makedirs('push_exports', exist_ok=True)
                ext = 'json' if resp_json is not None else 'txt'
                saved_path = os.path.join('push_exports', f"prioritize_response_{safe_handle}_{ts_readable}.{ext}")
                with open(saved_path, 'w', encoding='utf-8') as f:
                    if resp_json is not None:
                        json.dump(resp_json, f, ensure_ascii=False, indent=2)
                    else:
                        f.write(resp_text)
                print(f"Server response saved to: {saved_path}")
                logger.info(f"Server response saved to: {saved_path}")
            except Exception as e:
                logger.error(f"Failed to save server response: {e}")
                print(f"[WARN] Failed to save server response: {e}")
        else:
            # Short response: print to console and debug log
            if resp_json is not None:
                pretty = json.dumps(resp_json, ensure_ascii=False, indent=2)
                print(f"Server response (status {resp.status_code}):\n{pretty}")
                logger.debug(f"Server response JSON: {pretty}")
            else:
                print(f"Server response (status {resp.status_code}): {resp_text}")
                logger.debug(f"Server response text: {resp_text}")

        # Return success boolean and log details
        if resp.status_code == 200:
            logger.debug(resp)
            if saved_path:
                logger.info(f"Response stored at {saved_path}")
            return True
        else:
            logger.error(resp)
            # For non-200 responses, ensure details are available
            if not is_long and (resp_text or resp_json is not None):
                # Already printed above; also log at error level
                if resp_json is not None:
                    logger.error(json.dumps(resp_json, ensure_ascii=False))
                else:
                    logger.error(resp_text)
            elif saved_path:
                logger.error(f"Non-200 response details saved at {saved_path}")
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

        With descending=True, highest metric appears first (will be assigned the highest numeric priority).
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
        self.required_keys = {'uri', 'cid', 'like_count', 'news_description', 'news_title', 'news_uri', 'quote_count', 'reply_count', 'repost_count', 'text', 'createdAt'}
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
        # Meta counters to aid logging/export at pipeline level
        self.meta: dict[str, int] = {}

    
    def _cluster(self, data: pl.DataFrame, similarity: Literal['tfidf-cosine', 'count-cosine', 'sbert-cosine'] = 'tfidf-cosine'):
        """This function clusters the texts using the Leiden Algorithm by Traag, Waltman, & Van Eck (2019),
        See also suggestions by Trilling & Van Hoof (2020) for  news event clustering.
        It adds a column "cluster" and a column "clustersize" to the dataframe.
        """
        logger.debug("Creating cosine similarity matrix...")
        if similarity =='tfidf-cosine':
            vectorizer = TfidfVectorizer(stop_words=self.stopwords)
            texts = data['text'].fill_null('').to_list()
            bow = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(bow)
        elif similarity =='count-cosine':
            vectorizer = CountVectorizer(stop_words=self.stopwords)
            texts = data['text'].fill_null('').to_list()
            bow = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(bow)
        elif similarity == 'sbert-cosine':
            # import here to avoid forcing everyone to install this huge library
            from sentence_transformers import SentenceTransformer
            model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                sbert_model = SentenceTransformer(model_name, device=device)
            except Exception:
                sbert_model = SentenceTransformer(model_name)
            texts = data['text'].fill_null('').to_list()
            embeddings = sbert_model.encode(texts, show_progress_bar=True)
            sim_matrix = cosine_similarity(embeddings)
        else:
            raise NotImplementedError(f"Similarity {similarity} is not implemented")

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
        # Do NOT compute engagement-based stats here; they depend on the engagement window.
        # Stats (engagement count, rank, and sizes) are computed later in rank() via _add_cluster_stats
        # with the correct stats_subset and initial sizes.
        return data

    def _add_cluster_stats(self, df_with_clusters, stats_subset: pl.DataFrame | None = None):
        """Takes dataframe with cluster labels and adds cluster statistics.

        engagement_rank is computed so that 1 = highest engagement within the dataset,
        independent of the outer 'descending' flag.
        """
        src = stats_subset if stats_subset is not None else df_with_clusters
        # Aggregate per cluster on engagement window: sums per metric, size (engagement), and average createdAt (for tie-break)
        clusterstats = (
            src.group_by('cluster')
            .agg([
                pl.col('like_count').fill_null(0).sum().alias('like_count'),
                pl.col('reply_count').fill_null(0).sum().alias('reply_count'),
                pl.col('quote_count').fill_null(0).sum().alias('quote_count'),
                pl.col('repost_count').fill_null(0).sum().alias('repost_count'),
                pl.len().alias('size_engagement'),
                pl.col('createdAt_dt').mean().alias('avg_createdAt_dt'),
            ])
            .with_columns(
                sum=pl.sum_horizontal('reply_count','like_count','repost_count','quote_count')
            )
        )
        # Also compute initial cluster sizes on the clustering context
        initial_sizes = df_with_clusters.group_by('cluster').agg(pl.len().alias('size_initial'))
        # Deterministic ranking: higher engagement first; ties broken by newer average createdAt
        clusterstats = (
            clusterstats
            .sort(['sum','avg_createdAt_dt','cluster'], descending=[True, True, False])
            .with_row_index(name='rank_idx')
            .with_columns(engagement_rank=(pl.col('rank_idx') + 1))
            .drop('rank_idx')
            .rename({'sum':'engagement_count'})
            .join(initial_sizes, on='cluster', how='left')
        )
        # Prefix for clarity and join back
        clusterstats = clusterstats.rename(lambda x: f"cluster_{x}").rename({
            "cluster_cluster": "cluster",
            "cluster_size_engagement": "cluster_size_engagement",
            "cluster_size_initial": "cluster_size_initial",
        })
        df_with_clusters = df_with_clusters.join(clusterstats, on='cluster')
        return df_with_clusters
        
        
    def rank(self, data: list[dict] | pl.DataFrame) -> list[dict] | pl.DataFrame | list[str]:
        """Cluster → compute engagement/ranks → interleave by rank within push window.

        Steps:
        1) Build clusters on the clustering window (context only; no engagement yet).
        2) Compute engagement metrics and deterministic cluster ranks on the engagement window.
        3) Restrict to the push window FIRST; sort each cluster by recency.
        4) Interleave round‑robin across clusters ordered by cluster_engagement_rank.
        Returns the same structure as input (ids/dicts/dataframe) preserving the final order.
        """
        from datetime import datetime, timedelta, timezone
        df = self._transform_input(data)

        # Parse createdAt robustly: prefer epoch ns if present; otherwise parse strict UTC
        if 'createdAt_dt' not in df.columns:
            if 'createdAt_ns' in df.columns:
                df = df.with_columns(
                    pl.from_epoch(pl.col('createdAt_ns'), unit='ns', time_zone='UTC').alias('createdAt_dt')
                )
            else:
                df = df.with_columns(
                    pl.col('createdAt').str.to_datetime(strict=False, time_zone='UTC').alias('createdAt_dt')
                )

        now = datetime.now(timezone.utc)
        # Determine effective cluster window so every later subset is covered
        windows = [w for w in [self.cluster_window_days, self.engagement_window_days, self.push_window_days] if w is not None]
        effective_cluster_days = max(windows) if windows else None
        if effective_cluster_days is not None:
            cluster_cutoff = now - timedelta(days=int(effective_cluster_days))
            df_cluster_base = df.filter(pl.col('createdAt_dt') >= cluster_cutoff)
        else:
            df_cluster_base = df
        # Count posts that went into clustering context
        try:
            self.meta['cluster_posts'] = int(df_cluster_base.height)
        except Exception:
            self.meta['cluster_posts'] = 0

        if self.method == 'networkclustering-tfidf':
            df_with_clusters = self._cluster(df_cluster_base, similarity='tfidf-cosine')
        if self.method == 'networkclustering-count':
            df_with_clusters = self._cluster(df_cluster_base, similarity='count-cosine')
        if self.method == 'networkclustering-sbert':
            if len(df_cluster_base)>100:
                logger.warning(f"Do you really want to do this? You have {len(df_cluster_base)} texts; calculating sentence embeddings will be REALLY slow")
                logger.warning(f"Consider using another method, or submitting fewer documents")
            df_with_clusters = self._cluster(df_cluster_base, similarity='sbert-cosine')
        # Count clusters created over the clustered set (before any push-window filter)
        try:
            ncl = df_with_clusters.select(pl.col('cluster').n_unique().alias('n')).to_dicts()[0]['n']
            self.meta['clusters_created'] = int(ncl) if ncl is not None else 0
        except Exception:
            self.meta['clusters_created'] = 0
              
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
        try:
            self.meta['engagement_posts'] = int(stats_subset.height) if stats_subset is not None else int(df_with_clusters.height)
        except Exception:
            self.meta['engagement_posts'] = 0
        # Sum total engagement over the engagement window (likes+replies+quotes+reposts)
        try:
            src = stats_subset if stats_subset is not None else df_with_clusters
            tot = (
                src.select(
                    (
                        pl.col('like_count') + pl.col('reply_count') + pl.col('quote_count') + pl.col('repost_count')
                    ).sum().alias('total')
                ).to_dicts()[0]['total']
            )
            self.meta['engagement_total'] = int(tot) if tot is not None else 0
        except Exception:
            self.meta['engagement_total'] = 0
        df_with_clusters = self._add_cluster_stats(df_with_clusters, stats_subset=stats_subset)

        # Determine which rows are eligible for the push (apply push window BEFORE interleaving)
        if self.push_window_days is not None:
            push_cutoff = now - timedelta(days=int(self.push_window_days))
            eligible = df_with_clusters.filter(pl.col('createdAt_dt') >= push_cutoff)
        else:
            eligible = df_with_clusters

        # Build cluster order by engagement rank (1 first when descending=True)
        # Skip clusters that have no eligible rows
        try:
            cluster_order = (
                eligible
                .select(['cluster', 'cluster_engagement_rank'])
                .unique(maintain_order=True)
                .group_by('cluster')
                .agg(pl.col('cluster_engagement_rank').first())
                .sort('cluster_engagement_rank', descending=not self.descending)
                .select('cluster')
                .to_series()
                .to_list()
            )
        except Exception:
            logger.warning("Failed to compute cluster order; falling back to empty order", exc_info=True)
            cluster_order = []

        # For each cluster in order, collect its eligible rows sorted by recency (freshest first)
        per_cluster_rows = []
        for cid in cluster_order:
            try:
                rows = (
                    eligible
                    .filter(pl.col('cluster') == cid)
                    .sort('createdAt_dt', descending=True)
                    .rows(named=True)
                )
            except Exception:
                rows = []
            if rows:
                per_cluster_rows.append(rows)

        # Interleave round-robin across clusters in rank order
        interleaved = [item for bucket in zip_longest(*per_cluster_rows) for item in bucket if item is not None]
        final_ranking = pl.DataFrame(interleaved) if interleaved else eligible.head(0)
        # Add push-window cluster sizes for transparency
        try:
            push_sizes = eligible.group_by('cluster').agg(pl.len().alias('cluster_size_push'))
            final_ranking = final_ranking.join(push_sizes, on='cluster', how='left')
        except Exception:
            logger.warning("Failed to join push-window cluster sizes", exc_info=True)

        self.ranking = final_ranking
        try:
            self.meta['push_posts'] = int(final_ranking.height)
        except Exception:
            self.meta['push_posts'] = 0
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
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
    logger.setLevel(logging.DEBUG)
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
