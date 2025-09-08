#!/usr/bin/env python3
"""
End-to-end pipeline: fetch → rank (per handle) → push, with user-controlled
time windows for clustering, engagement ranking, and push.

Also provides a CLI.
"""
from __future__ import annotations

from typing import List, Optional
from pathlib import Path
from datetime import datetime, timezone, timedelta

import polars as pl

from .fetcher import Fetcher, ensure_db, load_posts_df
from .ranker import TopicRanker

import logging

logger = logging.getLogger('BSRlog')


def _ensure_push_logger(log_path: str = "push.log") -> logging.Logger:
    """Attach a FileHandler for push summaries if not yet attached."""
    logger = logging.getLogger('BSRpush')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fmt = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _keywords(texts: list[str], topk: int = 6) -> list[str]:
    import re
    counts = {}
    for t in texts:
        if not t:
            continue
        for w in re.findall(r"[A-Za-zÀ-ÿ]+", str(t).lower()):
            if len(w) < 4:
                continue
            counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:topk]]


def run_fetch_rank_push(
    handles: List[str],
    *,
    # Fetcher options
    xrpc_base: str = "https://public.api.bsky.app/xrpc",
    include_pins: bool = False,
    cutoff_check_every: int = 1,
    sqlite_path: Optional[str] = None,
    fetch_max_age_days: Optional[int] = None,
    # Ranker options
    method: str = 'networkclustering-tfidf',
    similarity_threshold: float = 0.2,
    vectorizer_stopwords: Optional[str | list[str]] = None,
    cluster_window_days: Optional[int] = None,
    engagement_window_days: Optional[int] = None,
    push_window_days: Optional[int] = 1,
    descending: bool = True,
    # Push options
    test: bool = True,
    dry_run: bool = False,
    log_path: str = 'push.log',
) -> None:
    """Fetch posts, rank per handle, and push ranked posts to the API per handle.

    - fetch_max_age_days: if None, uses max(cluster_window_days, engagement_window_days, push_window_days).
    - push_window_days defaults to 1 (24h), as commonly desired.
    """
    # Determine how far back we need to fetch to cover all windows
    windows = [w for w in [cluster_window_days, engagement_window_days, push_window_days] if w is not None]
    effective_fetch_days = fetch_max_age_days if fetch_max_age_days is not None else (max(windows) if windows else 7)

    pushlog = _ensure_push_logger(log_path)

    fetcher = Fetcher(xrpc_base)
    fetcher.fetch(
        handles=handles,
        max_age_days=effective_fetch_days,
        cutoff_check_every=cutoff_check_every,
        include_pins=include_pins,
        sqlite_path=sqlite_path,
    )

    db_path = sqlite_path or 'newsflows.db'
    conn = ensure_db(db_path)

    for h in handles:
        df = load_posts_df(conn, handle=h, order_by='createdAt', descending=False)
        if df.is_empty():
            logger.info(f"No rows for {h}; skipping push")
            continue

        ranker = TopicRanker(
            returnformat='dataframe',
            method=method,
            descending=descending,
            similarity_threshold=similarity_threshold,
            vectorizer_stopwords=vectorizer_stopwords,
            cluster_window_days=cluster_window_days,
            engagement_window_days=engagement_window_days,
            push_window_days=push_window_days,
        )

        ranked = ranker.rank(df)
        if ranked.is_empty():
            logger.info(f"Ranking for {h} returned 0 rows (push window too narrow?); skipping push")
            continue

        # Prepare summary for the actually pushed subset
        pushed = ranked  # final_ranking already restricted to push_window
        # Compute simple cluster stats
        agg = (
            pushed.group_by('cluster')
            .agg([
                pl.len().alias('size'),
                (pl.col('like_count') + pl.col('reply_count') + pl.col('quote_count') + pl.col('repost_count')).sum().alias('engagement'),
                pl.col('news_title').drop_nulls().head(60).alias('titles'),
                pl.col('text').drop_nulls().head(60).alias('texts'),
            ])
            .sort(['engagement','size'], descending=[True, True])
        )

        top_lines: List[str] = []
        for row in agg.head(5).iter_rows(named=True):
            titles = list(row['titles']) if row['titles'] is not None else []
            texts = list(row['texts']) if row['texts'] is not None else []
            kws = _keywords(titles + texts, topk=6)
            top_lines.append(
                f"cluster={row['cluster']} size={int(row['size'])} engagement={int(row['engagement'])} keywords=\"{' '.join(kws)}\""
            )

        windows_str = f"cluster={cluster_window_days or '-'}d, engagement={engagement_window_days or '-'}d, push={push_window_days or '-'}d"
        if dry_run:
            # Display an intelligible summary and a small priority preview
            print(f"\n=== Dry Run: handle={h} posts={pushed.height} method={method} threshold={similarity_threshold} windows=({windows_str})")
            for line in top_lines:
                print(f"  {line}")
            preview = pushed.select([
                pl.arange(0, pl.len()).alias('priority'),
                pl.col('cluster'),
                pl.col('createdAt'),
                pl.col('uri'),
                pl.col('news_title'),
            ]).head(15)
            print("  Priority preview (top 15):")
            for rec in preview.iter_rows(named=True):
                title = (rec['news_title'] or '')
                if isinstance(title, str) and len(title) > 80:
                    title = title[:77] + '...'
                print(f"    {rec['priority']:>3}  c={rec['cluster']:<4}  {rec['createdAt']}  {rec['uri']}  | {title}")
            # Also log the summary for auditing purposes
            pushlog.info(
                f"[DRY] handle={h} posts={pushed.height} method={method} threshold={similarity_threshold} windows=({windows_str})\n  "
                + "\n  ".join(top_lines)
            )
        else:
            # Push
            ok = ranker.post(test=test)
            status = 'OK' if ok else 'FAIL'
            pushlog.info(
                f"[{status}] handle={h} posts={pushed.height} method={method} threshold={similarity_threshold} windows=({windows_str})\n  "
                + "\n  ".join(top_lines)
            )


def main():
    import argparse

    p = argparse.ArgumentParser(description="Fetch → Rank (per handle) → Push with time windows")
    p.add_argument('--handles', nargs='+', required=True, help='Bluesky handles to include')
    p.add_argument('--xrpc-base', default='https://public.api.bsky.app/xrpc')
    p.add_argument('--include-pins', dest='include_pins', action='store_true')
    p.add_argument('--no-include-pins', dest='include_pins', action='store_false')
    p.set_defaults(include_pins=False)
    p.add_argument('--cutoff-check-every', type=int, default=1)
    p.add_argument('--sqlite-path', default='newsflows.db')
    p.add_argument('--fetch-max-age-days', type=int, default=None)

    p.add_argument('--method', default='networkclustering-tfidf', choices=['networkclustering-tfidf','networkclustering-count','networkclustering-sbert'])
    p.add_argument('--similarity-threshold', type=float, default=0.2)
    p.add_argument('--stopwords', default=None, help="'english' or comma-separated list; leave empty for None")
    p.add_argument('--cluster-window-days', type=int, default=None)
    p.add_argument('--engagement-window-days', type=int, default=None)
    p.add_argument('--push-window-days', type=int, default=1)
    p.add_argument('--descending', action='store_true', default=True)
    p.add_argument('--ascending', dest='descending', action='store_false')
    p.add_argument('--test', action='store_true', default=True, help='Do not persist priorities on server')
    p.add_argument('--no-test', dest='test', action='store_false')
    p.add_argument('--dry-run', action='store_true', default=False, help='Print summary and priority preview instead of calling the API')
    p.add_argument('--log-path', default='push.log')

    args = p.parse_args()

    stopwords = None
    if args.stopwords:
        stopwords = args.stopwords if args.stopwords == 'english' else [s.strip() for s in args.stopwords.split(',') if s.strip()]

    run_fetch_rank_push(
        handles=args.handles,
        xrpc_base=args.xrpc_base,
        include_pins=args.include_pins,
        cutoff_check_every=args.cutoff_check_every,
        sqlite_path=args.sqlite_path,
        fetch_max_age_days=args.fetch_max_age_days,
        method=args.method,
        similarity_threshold=args.similarity_threshold,
        vectorizer_stopwords=stopwords,
        cluster_window_days=args.cluster_window_days,
        engagement_window_days=args.engagement_window_days,
        push_window_days=args.push_window_days,
        descending=args.descending,
        test=args.test,
        dry_run=args.dry_run,
        log_path=args.log_path,
    )


if __name__ == '__main__':
    main()
