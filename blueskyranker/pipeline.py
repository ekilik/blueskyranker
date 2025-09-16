#!/usr/bin/env python3
"""
End-to-end pipeline: fetch → rank (per handle) → push, with user-controlled
time windows for clustering, engagement ranking, and push.

Also provides a CLI.
"""
from __future__ import annotations

from typing import List, Optional
import json
import os
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
    refresh_window: bool = True,
    # Ranker options
    method: str = 'networkclustering-tfidf',
    similarity_threshold: float = 0.2,
    vectorizer_stopwords: Optional[str | list[str]] = None,
    cluster_window_days: Optional[int] = None,
    engagement_window_days: Optional[int] = None,
    push_window_days: Optional[int] = 1,
    descending: bool = True,
    demote_last: bool = True,
    demote_window_hours: int = 48,
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
        refresh_window=refresh_window,
    )

    db_path = sqlite_path or 'newsflows.db'
    conn = ensure_db(db_path)

    # Helper: load URIs from the most recent export for a handle
    def _load_last_uris_for_handle(handle: str) -> set[str]:
        try:
            safe = handle.replace('.', '_')
            export_dir = Path('push_exports')
            if not export_dir.exists():
                return set()
            candidates = [p for p in export_dir.glob(f"push_{safe}_*.json") if p.is_file()]
            if not candidates:
                return set()
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            with latest.open('r', encoding='utf-8') as f:
                data = json.load(f)
            items = data.get('items') or []
            uris = [it.get('uri') for it in items if isinstance(it, dict) and it.get('uri')]
            return set(uris)
        except Exception:
            return set()

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
        # Determine last run URIs for this handle and demote any that are not in the current push set
        try:
            current_uris = set(pushed.select(pl.col('uri')).to_series().to_list())
        except Exception:
            current_uris = set()
        # New behavior: consider all posts from the past demote_window_hours and demote those not in current set
        if demote_last:
            try:
                now_utc = datetime.now(timezone.utc)
                cutoff_dt = now_utc - timedelta(hours=int(demote_window_hours))
                df_all = df
                if 'createdAt_dt' not in df_all.columns:
                    df_all = df_all.with_columns(
                        pl.col('createdAt').str.to_datetime(strict=False, time_zone='UTC').alias('createdAt_dt')
                    )
                window_uris = set(
                    df_all.filter(pl.col('createdAt_dt') >= cutoff_dt)
                    .select(pl.col('uri')).to_series().to_list()
                )
            except Exception:
                window_uris = set()
            demote_uris = list(window_uris - current_uris) if window_uris else []
        else:
            demote_uris = []
        demoted_count = len(demote_uris)
        # Compute simple cluster stats
        # Compute cluster engagement on the pushed subset using the same definition
        # as the ranker (sum per metric with nulls treated as 0, then sum horizontally).
        agg = (
            pushed.group_by('cluster')
            .agg([
                pl.len().alias('size'),
                pl.col('like_count').fill_null(0).sum().alias('like_sum'),
                pl.col('reply_count').fill_null(0).sum().alias('reply_sum'),
                pl.col('quote_count').fill_null(0).sum().alias('quote_sum'),
                pl.col('repost_count').fill_null(0).sum().alias('repost_sum'),
                pl.col('cluster_engagement_rank').min().alias('cluster_engagement_rank'),
                pl.col('news_title').drop_nulls().head(60).alias('titles'),
                pl.col('text').drop_nulls().head(60).alias('texts'),
            ])
            .with_columns(
                engagement=pl.sum_horizontal('like_sum','reply_sum','quote_sum','repost_sum')
            )
            .drop(['like_sum','reply_sum','quote_sum','repost_sum'])
            .sort(['engagement','size'], descending=[True, True])
        )

        # Build top cluster summary data for cleaner logging
        top_clusters: List[dict] = []
        top_cluster_ids: List[int] = []
        for row in agg.head(5).iter_rows(named=True):
            titles = list(row['titles']) if row['titles'] is not None else []
            texts = list(row['texts']) if row['texts'] is not None else []
            kws = _keywords(titles + texts, topk=6)
            cid = row['cluster']
            try:
                cid = int(cid)
            except Exception:
                pass
            top_clusters.append({
                'cluster': cid,
                'size': int(row['size']) if row['size'] is not None else 0,
                'engagement': int(row['engagement']) if row['engagement'] is not None else 0,
                'rank': int(row['cluster_engagement_rank']) if row.get('cluster_engagement_rank') is not None else None,
                'keywords': kws,
            })
            try:
                top_cluster_ids.append(int(row['cluster']))
            except Exception:
                pass

        # Prepare helpers and parsed datetime for recency computations
        def _truncate(s: Optional[str], n: int = 180) -> str:
            if s is None:
                return ''
            s = str(s).replace('\n', ' ').replace('\r', ' ')
            return s if len(s) <= n else (s[: n - 1] + '…')

        try:
            pushed_with_dt = pushed.with_columns(
                pl.col('createdAt').str.to_datetime(strict=False, time_zone='UTC').alias('createdAt_dt')
            )
        except Exception:
            pushed_with_dt = pushed.with_columns(pl.col('createdAt').alias('createdAt_dt'))

        # For each top cluster, collect the 5 most recent posts (structured)
        per_cluster_recent_lines: List[str] = []  # for console preview only
        per_cluster_recent: dict[int, list[dict]] = {}
        for cid in top_cluster_ids:
            try:
                cid_int = int(cid)
            except Exception:
                cid_int = cid
            subset = (
                pushed_with_dt
                .filter(pl.col('cluster') == cid_int)
                .sort('createdAt_dt', descending=True)
                .select(['createdAt','cluster','uri','news_uri','text','news_title','news_description'])
                .head(5)
                .to_dicts()
            )
            if not subset:
                continue
            per_cluster_recent_lines.append(f"Cluster {cid_int} — most recent posts (5):")
            per_cluster_recent[cid_int] = []
            for i, r in enumerate(subset, start=1):
                # For console preview (single line)
                per_cluster_recent_lines.append(
                    " | ".join([
                        f"  #{i}",
                        f"{r.get('createdAt','')}",
                        f"post={r.get('uri','')}",
                        f"news={r.get('news_uri','')}",
                        f"text=\"{_truncate(r.get('text'))}\"",
                        f"title=\"{_truncate(r.get('news_title'))}\"",
                        f"desc=\"{_truncate(r.get('news_description'))}\"",
                    ])
                )
                # For log (multi-line fields)
                per_cluster_recent[cid_int].append({
                    'createdAt': r.get('createdAt',''),
                    'uri': r.get('uri',''),
                    'news_uri': r.get('news_uri',''),
                    'text': _truncate(r.get('text')),
                    'news_title': _truncate(r.get('news_title')),
                    'news_description': _truncate(r.get('news_description')),
                })

        windows_str = f"cluster={cluster_window_days or '-'}d, engagement={engagement_window_days or '-'}d, push={push_window_days or '-'}d"

        # Build a comprehensive JSON export of the push order and metadata
        def _export_json(pushed_df: pl.DataFrame, demoted_count: int) -> str:
            os.makedirs('push_exports', exist_ok=True)
            # Compute priorities (higher number = higher priority).
            # Start at 1000 for the first item, then decrease by 1.
            push_with_prio = pushed_df.with_columns(
                prio=(pl.lit(1000) - pl.arange(0, pl.len()))
            )
            # Select fields and convert to list of dicts in the final order
            cols = [
                'prio','uri','cluster','createdAt','news_uri','text','news_title','news_description',
                'like_count','reply_count','quote_count','repost_count',
            ]
            # Some cluster stats may or may not be present; include when available
            opt_cols = ['cluster_size','cluster_engagement_count','cluster_engagement_rank']
            existing_opt = [c for c in opt_cols if c in push_with_prio.columns]
            rows = push_with_prio.select(
                [pl.col('prio').alias('priority')] + [pl.col(c) for c in cols[1:]] + [pl.col(c) for c in existing_opt]
            ).to_dicts()

            # Collect meta counters from ranker if available
            counts = {
                'cluster_posts': int(getattr(ranker, 'meta', {}).get('cluster_posts', 0)),
                'engagement_posts': int(getattr(ranker, 'meta', {}).get('engagement_posts', 0)),
                'push_posts': int(getattr(ranker, 'meta', {}).get('push_posts', pushed_df.height)),
                'clusters_created': int(getattr(ranker, 'meta', {}).get('clusters_created', 0)),
                'engagement_total': int(getattr(ranker, 'meta', {}).get('engagement_total', 0)),
                'demoted': int(demoted_count),
            }

            export = {
                'run': {
                    'handle': h,
                    'method': method,
                    'similarity_threshold': similarity_threshold,
                    'windows': {
                        'cluster_days': cluster_window_days,
                        'engagement_days': engagement_window_days,
                        'push_days': push_window_days,
                    },
                    'desc': 'higher numbers = higher priority',
                },
                'counts': counts,
                'top_clusters': [
                    {
                        'cluster': tc['cluster'],
                        'size': tc['size'],
                        'engagement': tc['engagement'],
                        'keywords': tc['keywords'],
                    }
                    for tc in top_clusters
                ],
                'items': rows,
            }
            # Human-readable timestamp for filename (UTC) e.g., 2025-09-15T12-49-43Z
            ts_readable = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')
            safe_handle = h.replace('.', '_')
            out_path = os.path.join('push_exports', f"push_{safe_handle}_{ts_readable}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(export, f, ensure_ascii=False, indent=2)
            return out_path
        if dry_run:
            # Display an intelligible summary and a small priority preview
            print(f"\n=== Dry Run: handle={h} posts={pushed.height} method={method} threshold={similarity_threshold} windows=({windows_str})")
            for tc in top_clusters:
                kws = ' '.join(tc['keywords']) if tc['keywords'] else ''
                print(f"  cluster={tc['cluster']} size={tc['size']} engagement={tc['engagement']} keywords=\"{kws}\"")
            if per_cluster_recent_lines:
                print("  Top cluster posts (5 each):")
                for line in per_cluster_recent_lines:
                    print(f"    {line}")
            preview = pushed.select([
                # Higher numbers = higher priority (start at 1000)
                (pl.lit(1000) - pl.arange(0, pl.len())).alias('priority'),
                pl.col('cluster'),
                pl.col('createdAt'),
                pl.col('uri'),
                pl.col('news_title'),
            ]).with_columns(
                pl.when(pl.col('priority') < 1).then(1).otherwise(pl.col('priority')).alias('priority')
            ).head(15)
            print("  Priority preview (top 15):")
            for rec in preview.iter_rows(named=True):
                title = (rec['news_title'] or '')
                if isinstance(title, str) and len(title) > 80:
                    title = title[:77] + '...'
                print(f"    {rec['priority']:>3}  c={rec['cluster']:<4}  {rec['createdAt']}  {rec['uri']}  | {title}")
            # Write JSON export for auditing the exact push order
            export_path = _export_json(pushed, demoted_count)
            print(f"  JSON export: {export_path}")
            # Also log the summary for auditing purposes
            # Build a cleaner, human-readable log block
            lines: List[str] = []
            lines.append(f"[DRY] handle={h} posts={pushed.height} method={method} threshold={similarity_threshold} windows=({windows_str})")
            # Counts summary for readability
            counts = {
                'cluster_posts': int(getattr(ranker, 'meta', {}).get('cluster_posts', 0)),
                'engagement_posts': int(getattr(ranker, 'meta', {}).get('engagement_posts', 0)),
                'push_posts': int(getattr(ranker, 'meta', {}).get('push_posts', pushed.height)),
                'clusters_created': int(getattr(ranker, 'meta', {}).get('clusters_created', 0)),
                'engagement_total': int(getattr(ranker, 'meta', {}).get('engagement_total', 0)),
                'demoted': int(demoted_count),
            }
            lines.append("Counts:")
            lines.append(f"  - posts for clustering: {counts['cluster_posts']}")
            lines.append(f"  - posts for engagement: {counts['engagement_posts']}")
            lines.append(f"  - posts for push      : {counts['push_posts']}")
            lines.append(f"  - clusters created     : {counts['clusters_created']}")
            lines.append(f"  - engagement total     : {counts['engagement_total']}")
            lines.append(f"  - demoted last-run URIs: {counts['demoted']}")
            lines.append("Top Clusters:")
            for tc in top_clusters:
                lines.append(f"  - cluster {tc['cluster']} (size={tc['size']}, engagement={tc['engagement']})")
                if tc['keywords']:
                    lines.append(f"    keywords: {' '.join(tc['keywords'])}")
            if per_cluster_recent:
                lines.append("Top Cluster Posts (5 each):")
                for cid in top_cluster_ids:
                    if cid not in per_cluster_recent:
                        continue
                    lines.append(f"  - Cluster {cid}:")
                    for i, r in enumerate(per_cluster_recent[cid], start=1):
                        lines.append(f"    {i}. {r['createdAt']}")
                        lines.append(f"       post: {r['uri']}")
                        lines.append(f"       news: {r['news_uri']}")
                        lines.append(f"       title: \"{r['news_title']}\"")
                        lines.append(f"       text: \"{r['text']}\"")
                        lines.append(f"       desc: \"{r['news_description']}\"")
            lines.append(f"JSON export: {export_path}")
            pushlog.info("\n".join(lines))
        else:
            # Push
            # Determine last run URIs for this handle and demote any that are not in the current push set
            try:
                current_uris = set(pushed.select(pl.col('uri')).to_series().to_list())
            except Exception:
                current_uris = set()
            last_uris = _load_last_uris_for_handle(h)
            demote_uris = list(last_uris - current_uris) if last_uris else []

            ok = ranker.post(test=test, handle=h, extra_uris_zero_prio=demote_uris)
            status = 'OK' if ok else 'FAIL'
            # Build a cleaner, human-readable log block
            lines: List[str] = []
            lines.append(f"[{status}] handle={h} posts={pushed.height} method={method} threshold={similarity_threshold} windows=({windows_str})")
            # Counts summary for readability
            counts = {
                'cluster_posts': int(getattr(ranker, 'meta', {}).get('cluster_posts', 0)),
                'engagement_posts': int(getattr(ranker, 'meta', {}).get('engagement_posts', 0)),
                'push_posts': int(getattr(ranker, 'meta', {}).get('push_posts', pushed.height)),
                'clusters_created': int(getattr(ranker, 'meta', {}).get('clusters_created', 0)),
                'engagement_total': int(getattr(ranker, 'meta', {}).get('engagement_total', 0)),
                'demoted': int(demoted_count),
            }
            lines.append("Counts:")
            lines.append(f"  - posts for clustering: {counts['cluster_posts']}")
            lines.append(f"  - posts for engagement: {counts['engagement_posts']}")
            lines.append(f"  - posts for push      : {counts['push_posts']}")
            lines.append(f"  - clusters created     : {counts['clusters_created']}")
            lines.append(f"  - engagement total     : {counts['engagement_total']}")
            lines.append(f"  - demoted last-run URIs: {counts['demoted']}")
            lines.append("Top Clusters:")
            for tc in top_clusters:
                lines.append(f"  - cluster {tc['cluster']} (size={tc['size']}, engagement={tc['engagement']})")
                if tc['keywords']:
                    lines.append(f"    keywords: {' '.join(tc['keywords'])}")
            if per_cluster_recent:
                lines.append("Top Cluster Posts (5 each):")
                for cid in top_cluster_ids:
                    if cid not in per_cluster_recent:
                        continue
                    lines.append(f"  - Cluster {cid}:")
                    for i, r in enumerate(per_cluster_recent[cid], start=1):
                        lines.append(f"    {i}. {r['createdAt']}")
                        lines.append(f"       post: {r['uri']}")
                        lines.append(f"       news: {r['news_uri']}")
                        lines.append(f"       title: \"{r['news_title']}\"")
                        lines.append(f"       text: \"{r['text']}\"")
                        lines.append(f"       desc: \"{r['news_description']}\"")
            # Write JSON export with the final order sent to server
            export_path = _export_json(pushed, demoted_count)
            lines.append(f"JSON export: {export_path}")
            pushlog.info("\n".join(lines))


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
    p.add_argument('--refresh-window', action='store_true', default=True, help='Re-fetch the entire effective window to refresh engagement metrics (default True)')
    p.add_argument('--no-refresh-window', dest='refresh_window', action='store_false')

    p.add_argument('--method', default='networkclustering-tfidf', choices=['networkclustering-tfidf','networkclustering-count','networkclustering-sbert'])
    p.add_argument('--similarity-threshold', type=float, default=0.2)
    p.add_argument('--stopwords', default=None, help="'english' or comma-separated list; leave empty for None")
    p.add_argument('--cluster-window-days', type=int, default=None)
    p.add_argument('--engagement-window-days', type=int, default=None)
    p.add_argument('--push-window-days', type=int, default=1)
    p.add_argument('--descending', action='store_true', default=True)
    p.add_argument('--ascending', dest='descending', action='store_false')
    p.add_argument('--demote-last', action='store_true', default=True, help='Also send last run\'s URIs not in the current set with priority 0')
    p.add_argument('--no-demote-last', dest='demote_last', action='store_false')
    p.add_argument('--demote-window-hours', type=int, default=48, help='Time window (in hours) to consider for demotion if not in current prioritisation (default: 48)')
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
        refresh_window=args.refresh_window,
        method=args.method,
        similarity_threshold=args.similarity_threshold,
        vectorizer_stopwords=stopwords,
        cluster_window_days=args.cluster_window_days,
        engagement_window_days=args.engagement_window_days,
        push_window_days=args.push_window_days,
        descending=args.descending,
        demote_last=args.demote_last,
        demote_window_hours=args.demote_window_hours,
        test=args.test,
        dry_run=args.dry_run,
        log_path=args.log_path,
    )


if __name__ == '__main__':
    main()
