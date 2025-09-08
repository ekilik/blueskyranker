#!/usr/bin/env python3
"""
Create a small SQLite sample DB from the bundled example CSV so users can
work with the new SQLite-first flow without needing to hit the network.

Usage (Python):
    from blueskyranker.sample_db import ensure_sample_db
    ensure_sample_db('newsflows_sample.db')

Then load posts via:
    from blueskyranker.fetcher import ensure_db, load_posts_df
    conn = ensure_db('newsflows_sample.db')
    df = load_posts_df(conn)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import polars as pl

from .fetcher import ensure_db, upsert_rows, CSV_HEADERS


EXAMPLE_CSV = Path(__file__).with_name('example_news.csv')


def ensure_sample_db(db_path: str = 'newsflows_sample.db') -> str:
    """Create (or refresh) a small SQLite DB with rows from example_news.csv.

    Returns the path to the SQLite file. Overwrites existing rows by URI.
    """
    if not EXAMPLE_CSV.exists():
        raise FileNotFoundError(f"Missing example CSV at {EXAMPLE_CSV}")

    df = pl.read_csv(str(EXAMPLE_CSV))

    # Normalize to SQLite schema columns; fill missing with None
    rows: List[Dict[str, Any]] = []
    for rec in df.to_dicts():
        out: Dict[str, Any] = {k: rec.get(k) for k in CSV_HEADERS}
        # Provide some defaults if absent in the example CSV
        out.setdefault('author_handle', 'sample.handle')
        out.setdefault('is_repost', 0)
        out.setdefault('like_count', 0)
        out.setdefault('repost_count', 0)
        out.setdefault('reply_count', 0)
        out.setdefault('quote_count', 0)
        rows.append(out)

    conn = ensure_db(db_path)
    upsert_rows(conn, rows)
    return db_path


def main():
    import argparse
    p = argparse.ArgumentParser(description='Create a sample SQLite DB from example_news.csv')
    p.add_argument('--db', default='newsflows_sample.db', help='Path to write the SQLite DB')
    args = p.parse_args()
    path = ensure_sample_db(args.db)
    print(f"Wrote sample DB: {path}")


if __name__ == '__main__':
    main()

