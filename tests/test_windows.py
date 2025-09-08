#!/usr/bin/env python3
"""
Lightweight tests for TopicRanker time windows and the pipeline orchestration.

We stub out the heavy clustering step and the network push so tests run fast and offline.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import polars as pl
import sys, os as _os
# Ensure project root is on sys.path when running as a script
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from blueskyranker import ranker as ranker_mod
from blueskyranker.fetcher import ensure_db
from blueskyranker.pipeline import run_fetch_rank_push


def _nowiso(delta_days: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=delta_days)
    return dt.isoformat()


def make_df() -> pl.DataFrame:
    # Two obvious clusters based on keyword in text: 'apple' vs 'banana'
    rows = [
        {
            'uri': f'uri:{i}', 'cid': f'cid:{i}', 'text': txt,
            'news_title': txt.title(), 'news_description': None, 'news_uri': None,
            'like_count': lc, 'reply_count': rc, 'quote_count': qc, 'repost_count': pc,
            'createdAt': _nowiso(days)
        }
        for i, (txt, lc, rc, qc, pc, days) in enumerate([
            ("apple news one",   5, 1, 0, 0, 2.5),  # older than 2 days
            ("apple news two",   8, 0, 1, 0, 0.5),  # within 1 day
            ("banana news one",  3, 2, 0, 0, 0.8),  # within 1 day
            ("banana news two", 10, 1, 0, 1, 0.2),  # within 1 day
        ])
    ]
    return pl.from_dicts(rows)


def stub_cluster(self, data: pl.DataFrame, similarity='tfidf-cosine') -> pl.DataFrame:
    # Simple deterministic clustering: text containing 'apple' -> cluster 0; else -> cluster 1
    return data.with_columns(
        cluster=pl.when(pl.col('text').str.contains('apple')).then(pl.lit(0)).otherwise(pl.lit(1))
    )


def test_ranker_windows():
    df = make_df()

    # Monkeypatch clustering to avoid heavy deps
    orig_cluster = ranker_mod.TopicRanker._cluster
    ranker_mod.TopicRanker._cluster = stub_cluster

    try:
        tr = ranker_mod.TopicRanker(
            returnformat='dataframe',
            method='networkclustering-tfidf',
            descending=True,
            similarity_threshold=0.2,
            vectorizer_stopwords='english',
            cluster_window_days=7,
            engagement_window_days=1,
            push_window_days=1,
        )
        ranked = tr.rank(df)

        # All pushed items should be within the last 1 day
        assert (ranked.select(pl.col('createdAt_dt')).drop_nulls().height == ranked.height), "createdAt_dt should be present"
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        assert ranked.filter(pl.col('createdAt_dt') < cutoff).is_empty(), "Push window filter failed"

        # Engagement over 1 day: apple cluster has only one recent doc (8+0+1+0=9), banana has two (3+2+0+0 + 10+1+0+1 = 17)
        # With descending=True and engagement_window_days=1, banana cluster should outrank apple (rank 1 is highest internally).
        # We check that the first few rows are from banana cluster (cluster==1) after interleaving.
        top_clusters = ranked.select('cluster').head(2)['cluster'].to_list()
        assert 1 in top_clusters, "Expected banana cluster to be prioritized in top results"
    finally:
        ranker_mod.TopicRanker._cluster = orig_cluster


def test_pipeline_smoke(tmp_db='test_newsflows.db', log_path='test_push.log'):
    # Prepare a tiny SQLite DB with one handle
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    conn = ensure_db(tmp_db)
    handle = 'test.handle'
    df = make_df().with_columns(author_handle=pl.lit(handle))
    # Write rows into SQLite
    cols = ['uri','cid','author_handle','author_did','indexedAt','createdAt','text','reply_root_uri','reply_parent_uri','is_repost','like_count','repost_count','reply_count','quote_count','news_title','news_description','news_uri']
    rows = []
    for r in df.to_dicts():
        base = {c: None for c in cols}
        base.update({
            'uri': r['uri'],
            'cid': r['cid'],
            'author_handle': handle,
            'createdAt': r['createdAt'],
            'text': r['text'],
            'is_repost': 0,
            'like_count': r['like_count'],
            'repost_count': r['repost_count'],
            'reply_count': r['reply_count'],
            'quote_count': r['quote_count'],
            'news_title': r['news_title'],
            'news_description': r['news_description'],
            'news_uri': r['news_uri'],
        })
        rows.append(base)
    with conn:
        placeholders = ','.join(['?']*len(cols))
        conn.executemany(
            f"INSERT INTO posts ({','.join(cols)}) VALUES ({placeholders})",
            [tuple(r.get(c) for c in cols) for r in rows]
        )

    # Monkeypatch clustering and pushing
    orig_cluster = ranker_mod.TopicRanker._cluster
    orig_post = ranker_mod._BaseRanker.post
    # Stub fetcher to avoid network
    from blueskyranker import pipeline as pipeline_mod
    orig_fetch = pipeline_mod.Fetcher.fetch
    pipeline_mod.Fetcher.fetch = lambda self, **kwargs: {"result": "stubbed"}
    ranker_mod.TopicRanker._cluster = stub_cluster
    ranker_mod._BaseRanker.post = lambda self, test=True: True

    try:
        run_fetch_rank_push(
            handles=[handle],
            sqlite_path=tmp_db,
            fetch_max_age_days=7,
            method='networkclustering-tfidf',
            similarity_threshold=0.2,
            vectorizer_stopwords='english',
            cluster_window_days=7,
            engagement_window_days=1,
            push_window_days=1,
            test=True,
            log_path=log_path,
        )
        # Validate the log was written with the handle
        assert os.path.exists(log_path), "push log not created"
        content = open(log_path, 'r', encoding='utf-8').read()
        assert handle in content, "handle missing in push log"
    finally:
        ranker_mod.TopicRanker._cluster = orig_cluster
        ranker_mod._BaseRanker.post = orig_post
        pipeline_mod.Fetcher.fetch = orig_fetch


def test_pipeline_dry_run(tmp_db='test_newsflows.db'):
    # Prepare DB as in smoke test
    import os
    from blueskyranker.fetcher import ensure_db
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    conn = ensure_db(tmp_db)
    handle = 'test.handle'
    df = make_df().with_columns(author_handle=pl.lit(handle))
    cols = ['uri','cid','author_handle','author_did','indexedAt','createdAt','text','reply_root_uri','reply_parent_uri','is_repost','like_count','repost_count','reply_count','quote_count','news_title','news_description','news_uri']
    rows = []
    for r in df.to_dicts():
        base = {c: None for c in cols}
        base.update({
            'uri': r['uri'],
            'cid': r['cid'],
            'author_handle': handle,
            'createdAt': r['createdAt'],
            'text': r['text'],
            'is_repost': 0,
            'like_count': r['like_count'],
            'repost_count': r['repost_count'],
            'reply_count': r['reply_count'],
            'quote_count': r['quote_count'],
            'news_title': r['news_title'],
            'news_description': r['news_description'],
            'news_uri': r['news_uri'],
        })
        rows.append(base)
    with conn:
        placeholders = ','.join(['?']*len(cols))
        conn.executemany(
            f"INSERT INTO posts ({','.join(cols)}) VALUES ({placeholders})",
            [tuple(r.get(c) for c in cols) for r in rows]
        )

    # Monkeypatch clustering; ensure API post is NOT called
    orig_cluster = ranker_mod.TopicRanker._cluster
    ranker_mod.TopicRanker._cluster = stub_cluster
    from blueskyranker import pipeline as pipeline_mod
    orig_fetch = pipeline_mod.Fetcher.fetch
    pipeline_mod.Fetcher.fetch = lambda self, **kwargs: {"result": "stubbed"}

    called = {'post': False}
    def _no_post(self, test=True):
        called['post'] = True
        raise RuntimeError('post should not be called in dry-run')
    orig_post = ranker_mod._BaseRanker.post
    ranker_mod._BaseRanker.post = _no_post

    # Capture stdout
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run_fetch_rank_push(
                handles=[handle],
                sqlite_path=tmp_db,
                fetch_max_age_days=7,
                method='networkclustering-tfidf',
                similarity_threshold=0.2,
                vectorizer_stopwords='english',
                cluster_window_days=7,
                engagement_window_days=1,
                push_window_days=1,
                dry_run=True,
            )
        out = buf.getvalue()
        assert 'Dry Run:' in out and 'Priority preview' in out, 'Expected dry-run summary with preview'
        assert not called['post'], 'post() was called during dry-run'
    finally:
        ranker_mod.TopicRanker._cluster = orig_cluster
        ranker_mod._BaseRanker.post = orig_post
        pipeline_mod.Fetcher.fetch = orig_fetch


if __name__ == '__main__':
    print("Running window tests...")
    test_ranker_windows()
    print("OK: TopicRanker windows")
    test_pipeline_smoke()
    print("OK: Pipeline smoke test")
