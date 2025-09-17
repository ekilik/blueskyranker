#!/usr/bin/env python3
"""
End‑to‑end pipeline: fetch → rank (per handle) → push, with user‑controlled
time windows and robust, auditable behavior.

Highlights:
- Two‑phase fetch: refresh engagement window (and push window if larger) so
  engagement counts used for ranking are current; then extend to the clustering
  window incrementally to provide graph context without refreshing older rows.
- Ranking uses the TopicRanker which applies deterministic tie‑breaks and interleaves
  the most recent posts from the most engaged clusters first (within the push window).
- JSON export contains a complete trace (windows, counts, top clusters, items).
  Items include UTC timestamps and a human‑readable local timestamp for validation.
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
        # Use a plain FileHandler (no rotation) so logs are never auto-deleted.
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fmt = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        # Avoid duplicate logs bubbling up to root
        logger.propagate = False
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
    handles: Optional[List[str]] = None,
    *,
    # Fetcher options
    xrpc_base: str = "https://public.api.bsky.app/xrpc",
    include_pins: bool = False,
    cutoff_check_every: int = 1,
    sqlite_path: Optional[str] = None,
    fetch_max_age_days: Optional[int] = None,
    # Ranker options
    method: str = 'networkclustering-sbert',
    similarity_threshold: float = 0.2,
    vectorizer_stopwords: Optional[str | list[str]] = None,
    cluster_window_days: Optional[int] = 7,
    engagement_window_days: Optional[int] = 1,
    push_window_days: Optional[int] = 1,
    descending: bool = True,
    demote_last: bool = True,
    demote_window_hours: int = 48,
    # Push options
    test: bool = True,
    dry_run: bool = False,
    log_path: str = 'push.log',
) -> None:
    """Fetch posts, rank per handle, and optionally push ranked posts.

    The function performs a two‑phase fetch by default:
    1) Refresh engagement/push window to update engagement (likes/replies/quotes/reposts)
       that actually matters for ranking.
    2) Extend to the clustering window incrementally to provide sufficient context
       for topic clustering without refreshing older rows.

    Ranking and ordering are window‑aware and deterministic. Exports and logs provide
    a full audit trail of the data sent (or previewed in dry‑run mode).
    """
    # Apply default handles if not provided
    if handles is None:
        handles = [
            "news-flows-nl.bsky.social",
            "news-flows-ir.bsky.social",
            "news-flows-cz.bsky.social",
            "news-flows-fr.bsky.social",
        ]

    # Determine windows
    win_cluster = cluster_window_days
    win_eng = engagement_window_days
    win_push = push_window_days
    # For engagement refresh, use the max of engagement/push (whichever is broader and matters for ranking)
    refresh_days = max([w for w in [win_eng, win_push] if w is not None], default=None)
    # Cluster context window: fallback to refresh_days if cluster window not provided
    cluster_days = win_cluster if win_cluster is not None else refresh_days
    # If user overrides max age via CLI, cap both phases accordingly
    if fetch_max_age_days is not None:
        if refresh_days is None:
            refresh_days = fetch_max_age_days
        else:
            refresh_days = min(refresh_days, fetch_max_age_days)
        if cluster_days is None:
            cluster_days = fetch_max_age_days
        else:
            cluster_days = min(cluster_days, fetch_max_age_days)
    # Sensible defaults if everything is None
    if refresh_days is None and cluster_days is None:
        refresh_days = 7
        cluster_days = 7

    pushlog = _ensure_push_logger(log_path)

    fetcher = Fetcher(xrpc_base)
    # Phase A: refresh engagement window (and push) to update engagement counts used for ranking
    if refresh_days is not None:
        fetcher.fetch(
            handles=handles,
            max_age_days=int(refresh_days),
            cutoff_check_every=cutoff_check_every,
            include_pins=include_pins,
            sqlite_path=sqlite_path,
            refresh_window=True,
        )
    # Phase B: ensure clustering context is available out to cluster_days without re-refreshing older rows
    if cluster_days is not None and (refresh_days is None or int(cluster_days) > int(refresh_days)):
        fetcher.fetch(
            handles=handles,
            max_age_days=int(cluster_days),
            cutoff_check_every=cutoff_check_every,
            include_pins=include_pins,
            sqlite_path=sqlite_path,
            refresh_window=False,
        )

    db_path = sqlite_path or 'newsflows.db'
    conn = ensure_db(db_path)

    # Migration: ensure createdAt_ns exists and is populated for robust time handling
    try:
        cur = conn.execute("PRAGMA table_info(posts)")
        cols = [r[1] for r in cur.fetchall()]
        if 'createdAt_ns' not in cols:
            conn.execute("ALTER TABLE posts ADD COLUMN createdAt_ns INTEGER")
        # Populate any NULL createdAt_ns values
        cur = conn.execute("SELECT uri, createdAt FROM posts WHERE createdAt_ns IS NULL OR createdAt_ns = ''")
        rows = cur.fetchall()
        if rows:
            def _to_ns(s: str) -> int | None:
                if not s:
                    return None
                t = s
                if t.endswith('Z'):
                    t = t[:-1] + '+00:00'
                try:
                    dt = datetime.fromisoformat(t)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return int(dt.timestamp() * 1_000_000_000)
                except Exception:
                    return None
            for uri, created in rows:
                ns = _to_ns(created)
                if ns is not None:
                    conn.execute("UPDATE posts SET createdAt_ns=? WHERE uri=?", (ns, uri))
            conn.commit()
    except Exception:
        logger.warning("createdAt_ns migration failed", exc_info=True)

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

        # Compute createdAt_dt once and reuse downstream
        if 'createdAt_dt' not in df.columns:
            try:
                df = df.with_columns(
                    pl.col('createdAt').str.to_datetime(strict=False, time_zone='UTC').alias('createdAt_dt')
                )
            except Exception:
                df = df.with_columns(pl.col('createdAt').alias('createdAt_dt'))

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
                window_uris = set(
                    df_all.filter(pl.col('createdAt_dt') >= cutoff_dt)
                    .select(pl.col('uri')).to_series().to_list()
                )
            except Exception:
                logger.warning("Failed computing demotion window URIs", exc_info=True)
                window_uris = set()
            demote_uris = list(window_uris - current_uris) if window_uris else []
        else:
            demote_uris = []
        demoted_count = len(demote_uris)
        
        # Compute simple cluster stats
        # Build top cluster summary derived from the ranker's fields to ensure consistency
        agg = (
            pushed.group_by('cluster')
            .agg([
                pl.len().alias('size_push'),
                pl.col('cluster_engagement_count').max().alias('engagement'),
                pl.col('cluster_engagement_rank').min().alias('rank'),
                pl.col('cluster_size_initial').max().alias('size_initial'),
                pl.col('cluster_size_engagement').max().alias('size_engagement'),
                pl.col('cluster_size_push').max().alias('size_push_confirm'),
                pl.col('news_title').drop_nulls().head(60).alias('titles'),
                pl.col('text').drop_nulls().head(60).alias('texts'),
            ])
            .with_columns(
                size_push=pl.coalesce([pl.col('size_push_confirm'), pl.col('size_push')])
            )
            .drop(['size_push_confirm'])
            .sort(['rank'], descending=[False])
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
                'size_initial': int(row.get('size_initial') or 0),
                'size_engagement': int(row.get('size_engagement') or 0),
                'size_push': int(row.get('size_push') or 0),
                'engagement': int(row['engagement']) if row['engagement'] is not None else 0,
                'rank': int(row['rank']) if row.get('rank') is not None else None,
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
            # Add local-time string for createdAt (Europe/Amsterdam) for validation
            tz = "Europe/Amsterdam"
            if 'createdAt_dt' in push_with_prio.columns:
                push_with_prio = push_with_prio.with_columns(
                    pl.col('createdAt_dt').dt.convert_time_zone(tz).dt.strftime('%Y-%m-%d %H:%M:%S %z').alias('createdAt_local')
                )
            else:
                # Fallback: parse createdAt and convert
                push_with_prio = push_with_prio.with_columns(
                    pl.col('createdAt').str.to_datetime(strict=False, time_zone='UTC').dt.convert_time_zone(tz).dt.strftime('%Y-%m-%d %H:%M:%S %z').alias('createdAt_local')
                )
            # Select fields and convert to list of dicts in the final order
            cols = [
                'prio','uri','cluster','createdAt','createdAt_local','news_uri','text','news_title','news_description',
                'like_count','reply_count','quote_count','repost_count',
            ]
            # Some cluster stats may or may not be present; include when available
            opt_cols = ['cluster_size_initial','cluster_size_engagement','cluster_size_push','cluster_engagement_count','cluster_engagement_rank']
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
                        'rank': tc.get('rank'),
                        'engagement': tc.get('engagement', 0),
                        'size_initial': tc.get('size_initial', 0),
                        'size_engagement': tc.get('size_engagement', 0),
                        'size_push': tc.get('size_push', 0),
                        'keywords': tc.get('keywords', []),
                    }
                    for tc in top_clusters
                ],
                'items': rows,
            }
            # Human-readable timestamp for filename based on the newest local item time
            try:
                ts_local = push_with_prio.select(pl.col('createdAt_local').max().alias('ts')).to_dicts()[0]['ts']
            except Exception:
                ts_local = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S +0000')
            # Convert "YYYY-MM-DD HH:MM:SS +HHMM" → "YYYY-MM-DDTHH-MM-SS+HHMM" for filename safety
            ts_readable = ts_local.replace(' ', 'T', 1).replace(':','-')
            safe_handle = h.replace('.', '_')
            out_path = os.path.join('push_exports', f"push_{safe_handle}_{ts_readable}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(export, f, ensure_ascii=False, indent=2)
            return out_path
        if dry_run:
            # Display an intelligible summary and a small priority preview
            print(f"\n=== Dry Run: handle={h} posts={pushed.height} method={method} threshold={similarity_threshold} windows=({windows_str})")
            for tc in top_clusters:
                kws = ' '.join(tc.get('keywords', [])) if tc.get('keywords') else ''
                print(f"  cluster={tc['cluster']} rank={tc.get('rank')} size_push={tc.get('size_push',0)} engagement={tc.get('engagement',0)} keywords=\"{kws}\"")
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
            lines.append(f"  - demoted (time-window): {counts['demoted']}")
            lines.append("Top Clusters:")
            for tc in top_clusters:
                lines.append(f"  - cluster {tc['cluster']} (rank={tc.get('rank')}, size_push={tc.get('size_push',0)}, engagement={tc.get('engagement',0)})")
                if tc.get('keywords'):
                    lines.append(f"    keywords: {' '.join(tc.get('keywords', []))}")
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
            # Push using time‑window demotion computed above
            try:
                if demote_uris:
                    ok = ranker.post(test=test, handle=h, extra_uris_zero_prio=demote_uris)
                else:
                    # Keep compatibility with test stubs that only accept `test`
                    ok = ranker.post(test=test)
            except TypeError:
                logger.warning("ranker.post() signature mismatch; calling with test only")
                ok = ranker.post(test=test)
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
            lines.append(f"  - demoted (time-window): {counts['demoted']}")
            lines.append("Top Clusters:")
            for tc in top_clusters:
                lines.append(f"  - cluster {tc['cluster']} (rank={tc.get('rank')}, size_push={tc.get('size_push',0)}, engagement={tc.get('engagement',0)})")
                if tc.get('keywords'):
                    lines.append(f"    keywords: {' '.join(tc.get('keywords', []))}")
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
    p.add_argument('--handles', nargs='+', default=[
        'news-flows-nl.bsky.social',
        'news-flows-ir.bsky.social',
        'news-flows-cz.bsky.social',
        'news-flows-fr.bsky.social',
    ], help='Bluesky handles to include')
    p.add_argument('--xrpc-base', default='https://public.api.bsky.app/xrpc')
    p.add_argument('--include-pins', dest='include_pins', action='store_true')
    p.add_argument('--no-include-pins', dest='include_pins', action='store_false')
    p.set_defaults(include_pins=False)
    p.add_argument('--cutoff-check-every', type=int, default=1)
    p.add_argument('--sqlite-path', default='newsflows.db')
    p.add_argument('--fetch-max-age-days', type=int, default=None)

    p.add_argument('--method', default='networkclustering-sbert', choices=['networkclustering-tfidf','networkclustering-count','networkclustering-sbert'])
    p.add_argument('--similarity-threshold', type=float, default=0.2)
    p.add_argument('--stopwords', default=None, help="'english' or comma-separated list; leave empty for None")
    p.add_argument('--cluster-window-days', type=int, default=7)
    p.add_argument('--engagement-window-days', type=int, default=1)
    p.add_argument('--push-window-days', type=int, default=1)
    p.add_argument('--descending', action='store_true', default=True)
    p.add_argument('--ascending', dest='descending', action='store_false')
    p.add_argument('--demote-last', action='store_true', default=True, help='Demote (priority 0) posts from the last N hours that are not in the current prioritisation')
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
