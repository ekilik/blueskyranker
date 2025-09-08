#!/usr/bin/env python3
"""
Generate a per-handle cluster report from SQLite data.

For each handle, loads posts from the DB, selects the top-N by engagement,
clusters with TopicRanker (TF‑IDF / Count / SBERT), and writes a Markdown
report with topic keywords, cluster stats, and a few example headlines.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import re

import polars as pl

from .fetcher import ensure_db, load_posts_df
from .ranker import TopicRanker


def _simple_keywords(texts: List[str], topk: int = 8) -> List[str]:
    counts = {}
    for t in texts:
        if not t:
            continue
        for w in re.findall(r"[A-Za-zÀ-ÿ]+", str(t).lower()):
            if len(w) < 4:
                continue
            counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:topk]]


def generate_cluster_report(
    db_path: str = "newsflows.db",
    output_path: str = "cluster_report.md",
    handles: Optional[List[str]] = None,
    method: str = "networkclustering-tfidf",
    sample_max: int = 600,
    similarity_threshold: float = 0.2,
    vectorizer_stopwords: Optional[str | List[str]] = None,
) -> str:
    """Generate cluster report Markdown and write to output_path. Returns path."""
    conn = ensure_db(db_path)
    if not handles:
        # infer handles present in DB
        import sqlite3
        cur = conn.execute("SELECT DISTINCT author_handle FROM posts ORDER BY author_handle")
        handles = [r[0] for r in cur.fetchall() if r[0]]

    sections: List[str] = []
    for h in handles:
        df = load_posts_df(conn, handle=h)
        if df.is_empty():
            continue
        df = df.with_columns(
            engagement=(pl.col('like_count') + pl.col('reply_count') + pl.col('quote_count') + pl.col('repost_count')).cast(pl.Float64)
        ).sort('engagement', descending=True).head(sample_max)

        rows = df.to_dicts()
        ranker = TopicRanker(
            returnformat='dataframe', method=method, descending=True,
            similarity_threshold=similarity_threshold,
            vectorizer_stopwords=vectorizer_stopwords,
        )
        ranked = ranker.rank(rows)

        # Aggregate by cluster; rely on cluster_* stats from TopicRanker
        agg = (
            ranked.group_by('cluster')
            .agg([
                pl.col('cluster_size').first().alias('size'),
                pl.col('cluster_engagement_count').first().alias('sum_engagement_cluster'),
                pl.col('cluster_engagement_rank').first().alias('engagement_rank'),
                pl.col('engagement').sum().alias('sum_engagement_sample'),
                pl.col('news_title').drop_nulls().head(80).alias('titles'),
                pl.col('text').drop_nulls().head(80).alias('texts'),
                pl.struct(['uri','news_title','engagement']).sort_by('engagement', descending=True).head(5).alias('top_posts')
            ])
            .sort(['sum_engagement_cluster','size'], descending=[True, True])
            .head(8)
        )

        lines = [f"## {h}"]
        for row in agg.iter_rows(named=True):
            titles = list(row['titles']) if row['titles'] is not None else []
            texts = list(row['texts']) if row['texts'] is not None else []
            kws = _simple_keywords(titles + texts, topk=8)
            lines.append(f"\n- Topic: {' '.join(kws) if kws else '(keywords n/a)'}")
            lines.append(f"  - Cluster ID: {row['cluster']}")
            lines.append(f"  - Posts: {int(row['size']) if row['size'] is not None else 0}")
            lines.append(f"  - Engagement (cluster sum): {int(row['sum_engagement_cluster']) if row['sum_engagement_cluster'] else 0}")
            lines.append(f"  - Engagement (sample sum): {int(row['sum_engagement_sample']) if row['sum_engagement_sample'] else 0}")
            if row['top_posts']:
                lines.append("  - Top posts:")
                for tp in row['top_posts']:
                    lines.append(f"    - {tp['engagement']}: {tp['news_title'] or '(no title)'}")
        sections.append("\n".join(lines))

    md = [
        "# Cluster Report",
        "This report lists the top topic clusters per handle, including a loose topic label (keywords), number of posts, and engagement.",
        "",
    ]
    md.extend(sections)
    Path(output_path).write_text("\n\n".join(md), encoding='utf-8')
    return output_path


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate per-handle cluster report from SQLite")
    p.add_argument('--db', default='newsflows.db', help='Path to SQLite database')
    p.add_argument('--output', default='cluster_report.md', help='Output Markdown path')
    p.add_argument('--method', default='networkclustering-tfidf', choices=['networkclustering-tfidf','networkclustering-count','networkclustering-sbert'])
    p.add_argument('--sample-max', type=int, default=600)
    p.add_argument('--similarity-threshold', type=float, default=0.2)
    p.add_argument('--stopwords', default=None, help="'english' or comma-separated list; leave empty for None")
    p.add_argument('--handles', nargs='*', default=None, help='Optional list of handles to include')
    args = p.parse_args()

    stopwords = None
    if args.stopwords:
        stopwords = args.stopwords if args.stopwords == 'english' else [s.strip() for s in args.stopwords.split(',') if s.strip()]

    generate_cluster_report(
        db_path=args.db,
        output_path=args.output,
        handles=args.handles,
        method=args.method,
        sample_max=args.sample_max,
        similarity_threshold=args.similarity_threshold,
        vectorizer_stopwords=stopwords,
    )
    print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()

