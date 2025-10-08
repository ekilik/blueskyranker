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
from typing import Literal, Optional, Sequence, List, Dict, Any
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


def _from_epoch_ns(expr, tz: str = 'UTC'):
    """Compatibility wrapper to convert epoch ns expressions across Polars versions."""
    try:
        return pl.from_epoch(expr, time_unit='ns', time_zone=tz)
    except TypeError:
        try:
            out = pl.from_epoch(expr, unit='ns')
        except TypeError:
            out = expr.cast(pl.Datetime('ns'))
        if tz:
            try:
                out = out.dt.replace_time_zone(tz)
            except Exception:
                pass
        return out
    except AttributeError:
        out = expr.cast(pl.Datetime('ns'))
        if tz:
            try:
                out = out.dt.replace_time_zone(tz)
            except Exception:
                pass
        return out


def _parse_created_at_string(value: object) -> datetime | None:
    """Parse loosely formatted timestamps (handles '... UTC' and missing 'T')."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        # Normalise common variants emitted by R bridges: "YYYY-MM-DD HH:MM:SS UTC"
        if text.endswith(" UTC"):
            text = text[:-4] + "+00:00"
        if "T" not in text[:19] and len(text) >= 19:
            text = text[:19].replace(" ", "T", 1) + text[19:]
        if text.endswith("Z"):
            normalised = text[:-1] + "+00:00"
        else:
            normalised = text
        try:
            dt = datetime.fromisoformat(normalised)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


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
        host_env = (
            os.getenv('FEEDGEN_HOSTNAME')
            or os.getenv('FEEDGEN_LISTENHOST')
            or 'localhost:3020'
        )
        host_env = host_env.strip()
        if host_env.startswith(('http://', 'https://')):
            server = host_env.rstrip('/')
        else:
            if host_env.startswith(('localhost', '127.')):
                server = f"http://{host_env}"
            else:
                server = f"https://{host_env}:443"
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
    similarity_threshold defaults to 0.5 for SBERT and 0.2 for the TF–IDF / Count
    variants when not explicitly supplied.
    """
    def __init__(self, 
            returnformat: Literal["id","dicts","dataframe"],
            method: Literal['networkclustering-tfidf', 'networkclustering-count', 'networkclustering-sbert'],
            descending: bool,
            metric: Literal['like_count', 'quote_count',  'reply_count',  'repost_count', 'engagement'] = 'engagement',
            similarity_threshold: float | None = None,
            vectorizer_stopwords: str | list[str] | None = None,
            cluster_window_days: int | None = None,
            engagement_window_days: int | None = None,
            push_window_days: int | None = None,
            include_embedding_metrics: bool = False,
            cluster_insights: Sequence[str] | str | None = None,
            cluster_insight_options: Optional[Dict[str, Any]] | None = None,
        ):
        self.required_keys = {'uri', 'cid', 'like_count', 'news_description', 'news_title', 'news_uri', 'quote_count', 'reply_count', 'repost_count', 'text', 'createdAt'}
        self.returnformat = returnformat
        if metric != 'engagement':
            raise NotImplementedError
        self.metric = metric
        self.method = method
        self.descending = descending
        if similarity_threshold is None:
            method_key = (method or '').lower()
            similarity_threshold = 0.5 if method_key == 'networkclustering-sbert' else 0.2
        self.similarity_threshold = similarity_threshold
        self.stopwords = vectorizer_stopwords
        self.cluster_window_days = cluster_window_days
        self.engagement_window_days = engagement_window_days
        self.push_window_days = push_window_days
        self.include_embedding_metrics = bool(include_embedding_metrics)
        if cluster_insights is None:
            insights: List[str] = []
        elif isinstance(cluster_insights, str):
            insights = [cluster_insights]
        else:
            insights = list(cluster_insights)
        self.cluster_insights = {insight.strip().lower() for insight in insights if insight}
        self.cluster_insight_options: Dict[str, Any] = cluster_insight_options or {}
        self.ranking = None # will only be populated after .rank() is called (think of .fit() in scikit-learn)
        # Meta counters to aid logging/export at pipeline level
        self.meta: dict[str, int] = {}
        self._model_cache: Dict[tuple[str, str], Any] = {}
        self._keybert_cache: Dict[str, Any] = {}
        self._last_embeddings: Optional[np.ndarray] = None
        self._last_embedding_cid_to_idx: Dict[str, int] = {}
        self._last_sbert_model_name: Optional[str] = None

    def _ensure_text_column(self, df: pl.DataFrame) -> pl.DataFrame:
        """Guarantee that the text column is populated with meaningful content."""
        if "text" not in df.columns:
            return df
        text_expr = pl.col("text").cast(pl.Utf8)
        cleaned = text_expr.fill_null("").str.strip_chars()
        fallback_sources = []
        for column in ("news_title", "news_description", "news_uri", "uri", "cid"):
            if column in df.columns:
                expr = pl.col(column).cast(pl.Utf8).fill_null("").str.strip_chars()
                fallback_sources.append(pl.when(expr == "").then(None).otherwise(expr))
        if fallback_sources:
            fallback_expr = pl.coalesce(*fallback_sources)
        else:
            fallback_expr = pl.lit(None)
        df = df.with_columns(
            pl.when(cleaned == "")
            .then(fallback_expr)
            .otherwise(text_expr)
            .alias("text")
        )
        df = df.with_columns(
            pl.when(
                pl.col("text").cast(pl.Utf8).fill_null("").str.strip_chars() == ""
            )
            .then(
                pl.lit("placeholder_post_")
                + pl.arange(0, pl.len(), eager=False).cast(pl.Utf8)
            )
            .otherwise(pl.col("text"))
            .alias("text")
        )
        return df.with_columns(pl.col("text").cast(pl.Utf8))

    
    def _cluster(self, data: pl.DataFrame, similarity: Literal['tfidf-cosine', 'count-cosine', 'sbert-cosine'] = 'tfidf-cosine'):
        """This function clusters the texts using the Leiden Algorithm by Traag, Waltman, & Van Eck (2019),
        See also suggestions by Trilling & Van Hoof (2020) for  news event clustering.
        It adds a column "cluster" and a column "clustersize" to the dataframe.
        """
        logger.debug("Creating cosine similarity matrix...")
        self._last_embeddings = None
        self._last_embedding_cid_to_idx = {}
        self._last_sbert_model_name = None
        if similarity =='tfidf-cosine':
            vectorizer = TfidfVectorizer(stop_words=self.stopwords)
            self._last_sbert_model_name = model_name
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
            # model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
            model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            device = None
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                device = None
            try:
                sbert_model = self._load_sentence_transformer(model_name, device=device)
            except Exception:
                if device:
                    sbert_model = self._load_sentence_transformer(model_name, device=None)
                else:
                    raise
            texts = data['text'].fill_null('').to_list()
            embeddings = sbert_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
            if self.include_embedding_metrics:
                cid_list = data['cid'].to_list()
                self._last_embeddings = np.asarray(embeddings)
                self._last_embedding_cid_to_idx = {
                    cid: idx for idx, cid in enumerate(cid_list) if cid is not None
                }
            else:
                self._last_embeddings = None
                self._last_embedding_cid_to_idx = {}
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
                pl.col('createdAt_dt').min().alias('createdAt_first'),
                pl.col('createdAt_dt').max().alias('createdAt_last'),
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
        if "cluster_createdAt_first" in clusterstats.columns and "cluster_createdAt_last" in clusterstats.columns:
            try:
                duration = (pl.col("cluster_createdAt_last") - pl.col("cluster_createdAt_first"))
                clusterstats = clusterstats.with_columns(
                    (duration.dt.total_seconds() / 3600.0).alias("cluster_duration_hours")
                )
            except AttributeError:
                clusterstats = clusterstats.with_columns(
                    ((pl.col("cluster_createdAt_last") - pl.col("cluster_createdAt_first")).dt.nanoseconds() / 3_600_000_000_000).alias("cluster_duration_hours")
                )
        df_with_clusters = df_with_clusters.join(clusterstats, on='cluster')
        return df_with_clusters

    def _add_embedding_metrics(self, df_with_clusters: pl.DataFrame) -> pl.DataFrame:
        if not self.include_embedding_metrics or self.method != 'networkclustering-sbert':
            self._last_embeddings = None
            self._last_embedding_cid_to_idx = {}
            return df_with_clusters
        if self._last_embeddings is None or not len(self._last_embedding_cid_to_idx):
            logger.warning("Requested embedding metrics but embedding cache is empty; skipping distance computation.")
            return df_with_clusters
        embeddings = self._last_embeddings
        cid_to_idx = self._last_embedding_cid_to_idx
        cid_list = df_with_clusters['cid'].to_list()
        cluster_list = df_with_clusters['cluster'].to_list()
        cluster_vectors: Dict[Any, List[int]] = {}
        for cid, cluster in zip(cid_list, cluster_list):
            if cluster is None:
                continue
            embed_idx = cid_to_idx.get(cid)
            if embed_idx is None:
                continue
            cluster_vectors.setdefault(cluster, []).append(embed_idx)
        if not cluster_vectors:
            self._last_embeddings = None
            self._last_embedding_cid_to_idx = {}
            return df_with_clusters
        centroids: Dict[Any, np.ndarray] = {}
        for cluster, indices in cluster_vectors.items():
            if not indices:
                continue
            vecs = embeddings[indices]
            centroid = vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm:
                centroid = centroid / norm
            centroids[cluster] = centroid
        distances: List[Optional[float]] = []
        similarities: List[Optional[float]] = []
        for cid, cluster in zip(cid_list, cluster_list):
            embed_idx = cid_to_idx.get(cid)
            centroid = centroids.get(cluster)
            if embed_idx is None or centroid is None:
                distances.append(None)
                similarities.append(None)
                continue
            sim = float(np.dot(embeddings[embed_idx], centroid))
            similarities.append(sim)
            distances.append(1.0 - sim)
        df_with_clusters = df_with_clusters.with_columns([
            pl.Series("cluster_centroid_similarity", similarities),
            pl.Series("cluster_centroid_distance", distances),
        ])
        self._last_embeddings = None
        self._last_embedding_cid_to_idx = {}
        return df_with_clusters

    def _build_cluster_documents(self, df_with_clusters: pl.DataFrame, *, max_docs: int = 50, max_chars: int = 4000) -> Dict[Any, str]:
        cols = [col for col in ("text", "news_title", "news_description") if col in df_with_clusters.columns]
        if not cols:
            cols = ["text"]
        docs: Dict[Any, List[str]] = {}
        counts: Dict[Any, int] = {}
        rows = (
            df_with_clusters
            .select(["cluster", "createdAt_dt", *cols])
            .sort("createdAt_dt", descending=True, nulls_last=True)
            .to_dicts()
        )
        for row in rows:
            cluster = row.get("cluster")
            if cluster is None:
                continue
            if counts.get(cluster, 0) >= max_docs:
                continue
            pieces = []
            for col in cols:
                value = row.get(col)
                if value:
                    pieces.append(str(value))
            if not pieces:
                continue
            docs.setdefault(cluster, []).append(" ".join(pieces))
            counts[cluster] = counts.get(cluster, 0) + 1
        collapsed: Dict[Any, str] = {}
        for cluster, pieces in docs.items():
            doc = " ".join(pieces)
            if len(doc) > max_chars:
                doc = doc[:max_chars]
            collapsed[cluster] = doc
        return collapsed

    def _compute_distinct_words(self, cluster_docs: Dict[Any, str], top_n: int = 8) -> Dict[Any, List[str]]:
        if not cluster_docs:
            return {}
        texts = []
        clusters = []
        for cluster, doc in cluster_docs.items():
            text = (doc or "").strip()
            if not text:
                continue
            clusters.append(cluster)
            texts.append(text)
        if not texts:
            return {cluster: [] for cluster in cluster_docs}
        stopwords = self.stopwords if isinstance(self.stopwords, (list, str)) else None
        vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=5000)
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return {cluster: [] for cluster in cluster_docs}
        feature_names = vectorizer.get_feature_names_out()
        distinct: Dict[Any, List[str]] = {cluster: [] for cluster in cluster_docs}
        for idx, cluster in enumerate(clusters):
            row = matrix.getrow(idx)
            if row.nnz == 0:
                continue
            scores = row.toarray().ravel()
            top_indices = np.argsort(scores)[::-1]
            keywords = []
            for feature_idx in top_indices:
                if scores[feature_idx] <= 0:
                    continue
                keywords.append(feature_names[feature_idx])
                if len(keywords) >= top_n:
                    break
            distinct[cluster] = keywords
        return distinct

    def _compute_keybert_phrases(self, cluster_docs: Dict[Any, str]) -> Dict[Any, Dict[str, List[Any]]]:
        if not cluster_docs:
            return {}
        try:
            from keybert import KeyBERT
        except ImportError:
            logger.warning("KeyBERT not available; skipping keyphrase extraction.")
            return {}
        model_name = self._last_sbert_model_name or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        sentence_model = self._load_sentence_transformer(model_name, device=None)
        key = f"{model_name}"
        if key not in self._keybert_cache:
            self._keybert_cache[key] = KeyBERT(model=sentence_model)
        keybert_model = self._keybert_cache[key]
        keybert_config = self.cluster_insight_options.get('keybert', {})
        if not isinstance(keybert_config, dict):
            keybert_config = {}
        defaults = {
            "keyphrase_ngram_range": (1, 3),
            "top_n": 8,
            "use_mmr": True,
            "diversity": 0.5,
            "use_maxsum": False,
            "nr_candidates": 30,
            "stop_words": None,
        }
        options = defaults | {k: v for k, v in keybert_config.items() if k not in {"max_docs_per_cluster", "max_chars"}}
        doc_options = {
            "max_docs_per_cluster": int(keybert_config.get('max_docs_per_cluster', 50)),
            "max_chars": int(keybert_config.get('max_chars', 4000)),
        }
        top_n = int(options.get("top_n", 8))
        kw_kwargs = options.copy()
        maxsum = kw_kwargs.get("use_maxsum", False)
        if not maxsum:
            kw_kwargs.pop("nr_candidates", None)
        results: Dict[Any, Dict[str, List[Any]]] = {}
        for cluster, doc in cluster_docs.items():
            text = (doc or "").strip()
            if not text:
                results[cluster] = {"phrases": [], "scores": []}
                continue
            truncated = text[: doc_options["max_chars"]]
            try:
                keywords = keybert_model.extract_keywords(truncated, **kw_kwargs)
            except Exception as exc:
                logger.warning("KeyBERT failed for cluster %s: %s", cluster, exc)
                results[cluster] = {"phrases": [], "scores": []}
                continue
            phrases = [phrase for phrase, _ in keywords][:top_n]
            scores = [float(score) for _, score in keywords][:top_n]
            results[cluster] = {"phrases": phrases, "scores": scores}
        return results

    def _add_cluster_insights(self, df_with_clusters: pl.DataFrame) -> pl.DataFrame:
        if not self.cluster_insights:
            return df_with_clusters
        docs_options = self.cluster_insight_options.get('documents', {})
        if not isinstance(docs_options, dict):
            docs_options = {}
        keybert_doc_opts = self.cluster_insight_options.get('keybert', {})
        if not isinstance(keybert_doc_opts, dict):
            keybert_doc_opts = {}
        max_docs = int(docs_options.get('max_docs_per_cluster', keybert_doc_opts.get('max_docs_per_cluster', 50)))
        max_chars = int(docs_options.get('max_chars', keybert_doc_opts.get('max_chars', 4000)))
        cluster_docs = self._build_cluster_documents(df_with_clusters, max_docs=max_docs, max_chars=max_chars)
        clusters_present = {cluster for cluster in df_with_clusters['cluster'].to_list() if cluster is not None}
        for cluster in clusters_present:
            cluster_docs.setdefault(cluster, "")
        enrichments: List[pl.DataFrame] = []
        if 'distinct_words' in self.cluster_insights:
            distinct_opts = self.cluster_insight_options.get('distinct_words', {})
            if not isinstance(distinct_opts, dict):
                distinct_opts = {}
            top_n = int(distinct_opts.get('top_n', 8))
            distinct = self._compute_distinct_words(cluster_docs, top_n=top_n)
            enrichments.append(
                pl.DataFrame({
                    'cluster': list(distinct.keys()),
                    'cluster_keywords_distinct': list(distinct.values()),
                })
            )
        if 'keybert' in self.cluster_insights:
            keybert_results = self._compute_keybert_phrases(cluster_docs)
            if keybert_results:
                clusters = list(keybert_results.keys())
                phrases = [keybert_results[c]['phrases'] for c in clusters]
                scores = [keybert_results[c]['scores'] for c in clusters]
                titles = [phrases[i][0] if phrases[i] else None for i in range(len(phrases))]
                enrichments.append(
                    pl.DataFrame({
                        'cluster': clusters,
                        'cluster_keybert_phrases': phrases,
                        'cluster_keybert_scores': scores,
                        'cluster_keybert_label': titles,
                    })
                )
        if not enrichments:
            return df_with_clusters
        enrichment_df = enrichments[0]
        for extra in enrichments[1:]:
            enrichment_df = enrichment_df.join(extra, on='cluster', how='outer')
        df_with_clusters = df_with_clusters.join(enrichment_df, on='cluster', how='left')
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
        df = self._ensure_text_column(df)

        # Parse createdAt robustly: prefer epoch ns if present; otherwise parse strict UTC
        if 'createdAt_dt' not in df.columns:
            if 'createdAt_ns' in df.columns:
                df = df.with_columns(
                    _from_epoch_ns(pl.col('createdAt_ns'), tz='UTC').alias('createdAt_dt')
                )
            else:
                df = df.with_columns(
                    pl.col('createdAt').str.to_datetime(strict=False, time_zone='UTC').alias('createdAt_dt')
                )
        if 'createdAt_dt' in df.columns:
            try:
                null_count = int(df['createdAt_dt'].null_count())
            except Exception:
                null_count = None
            if (null_count is None or null_count == df.height) and 'createdAt' in df.columns:
                try:
                    df = df.with_columns(
                        pl.col('createdAt')
                        .map_elements(
                            _parse_created_at_string,
                            return_dtype=pl.Datetime(time_unit='us', time_zone='UTC')
                        )
                        .alias('createdAt_dt')
                    )
                except Exception:
                    pass

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

        if df_cluster_base.height == 0:
            logger.warning("No posts fall within the configured clustering window; returning empty ranking.")
            self.meta['clusters_created'] = 0
            self.meta['engagement_posts'] = 0
            self.meta['engagement_total'] = 0
            self.meta['push_posts'] = 0
            empty = df.head(0)
            self.ranking = empty
            return self._transform_output(empty)

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
        df_with_clusters = self._add_embedding_metrics(df_with_clusters)
        df_with_clusters = self._add_cluster_insights(df_with_clusters)

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

    def _load_sentence_transformer(self, model_name: str, device: str | None = None):
        """Load a SentenceTransformer model with optional offline overrides."""
        from sentence_transformers import SentenceTransformer
        cache_key = (model_name, device or "auto")
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        cache_override = os.getenv("SBERT_MODEL_PATH")
        cache_dir = None
        if cache_override:
            cache_dir = os.path.expanduser(cache_override)
        local_only = os.getenv("SBERT_LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes"}

        init_kwargs: dict[str, object] = {}
        if cache_dir:
            init_kwargs["cache_folder"] = cache_dir
        if device:
            init_kwargs["device"] = device
        if local_only:
            init_kwargs["local_files_only"] = True

        try:
            model = SentenceTransformer(model_name, **init_kwargs)
            self._model_cache[cache_key] = model
            return model
        except requests.exceptions.ConnectionError as exc:
            hint = (
                "Could not download SBERT model '{model}'. "
                "Set SBERT_MODEL_PATH to a directory that contains the model files "
                "or switch to method='networkclustering-tfidf'."
            ).format(model=model_name)
            raise RuntimeError(hint) from exc
        except Exception as exc:
            if cache_dir:
                hint = (
                    f"Failed to load SBERT model from '{cache_dir}'. "
                    "Ensure the directory contains a valid SentenceTransformer export "
                    "or remove SBERT_MODEL_PATH to allow default behaviour."
                )
                raise RuntimeError(hint) from exc
            raise


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
