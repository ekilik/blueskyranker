#!/usr/bin/env python3
"""
Bluesky Author Feed Scraper (via public AppView, using atproto)

Fetches PUBLIC posts for one or more Bluesky handles with engagement metrics and
embedded news link metadata. Shows progress bars, supports a max-age cutoff,
appends to per-handle CSVs (de-duped by URI), re-fetches the last-saved day for
safety, and updates a combined CSV. Designed for repeat, incremental runs.

Usage:
    python fetch_bsky_author_feeds_progress.py \
        --handles <handle1> <handle2> ... \
        [--max-age-days N] [--cutoff-check-every K] \
        [--include-pins | --no-include-pins] [--xrpc-base URL]

Parameters:
    --handles               Space-separated Bluesky handles to fetch.
                            Default: news-flows-nl.bsky.social news-flows-ir.bsky.social
                                    news-flows-cz.bsky.social news-flows-fr.bsky.social
    --xrpc-base             XRPC base URL.
                            Default: https://public.api.bsky.app/xrpc
    --max-age-days          Only keep posts with createdAt within last N days;
                            also used with incremental logic (see below).
                            Default: 7
    --cutoff-check-every    Pages between early-stop checks against cutoff.
                            Default: 1
    --include-pins          Include pinned posts.
                            Default is to exclude.
    --no-include-pins       Explicitly exclude pinned posts.
                            Default is to include.

Outputs:
    - Per-handle CSVs: {handle_with_dots_replaced}_author_feed.csv (append-only, de-duped by uri).
    - Combined CSV:    all_handles_author_feed.csv (only rows newly added THIS run).

Example usage for the past 7 days:
    python fetch_bsky_author_feeds_progress.py

Requirements:
    pip install atproto tqdm
"""

import csv
import os
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from atproto import Client, models
from tqdm import tqdm

import logging

logger = logging.getLogger('BSRlog')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    #datefmt='%Y-%m-%d %H:%M:%S')
    datefmt='%H:%M:%S')
logger.setLevel(logging.DEBUG)

APPVIEW_XRPC = "https://public.api.bsky.app/xrpc"
PAGE_LIMIT = 100
SLEEP_SEC = 0.20  # polite pacing

CSV_HEADERS = [
    "uri","cid","author_handle","author_did","indexedAt","createdAt","text",
    "reply_root_uri","reply_parent_uri","is_repost",
    "like_count","repost_count","reply_count","quote_count",
    "news_title","news_description","news_uri"
]

def iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # handle trailing Z or timezone-less strings
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def ensure_headers(path: str) -> bool:
    """Create file with header if it doesn't exist. Return True if newly created."""
    if os.path.exists(path):
        return False
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()
    return True

def load_existing_uris(path: str) -> Set[str]:
    """Return set of URIs already present in an existing CSV (if it exists)."""
    if not os.path.exists(path):
        return set()
    uris: Set[str] = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uri = row.get("uri")
            if uri:
                uris.add(uri)
    return uris

def latest_created_at_in_csv(path: str) -> Optional[datetime]:
    """Return the latest (max) createdAt datetime found in the CSV, if any."""
    if not os.path.exists(path):
        return None
    latest = None
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = row.get("createdAt")
            if not ts:
                continue
            dt = iso_to_dt(ts)
            if dt and (latest is None or dt > latest):
                latest = dt
    return latest

def append_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    """Append given rows to CSV (assumes header exists)."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        for r in rows:
            writer.writerow({h: r.get(h) for h in CSV_HEADERS})

def extract_news_embed(item_post: Any, record: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Try both hydrated embed (post.embed.external) and record embed (record.embed.external).
    Return (title, description, uri).
    """
    def safe_get_external(from_obj: Any):
        if not from_obj:
            return None
        # Hydrated view path: post.embed.external (AppBskyEmbedExternal.ViewExternal)
        ext = getattr(from_obj, "external", None)
        if ext and getattr(ext, "title", None) is not None:
            title = getattr(ext, "title", None)
            desc = getattr(ext, "description", None)
            uri  = getattr(ext, "uri", None) or getattr(ext, "url", None)
            return (title, desc, uri)
        return None

    # Try hydrated (view) embed on post
    view_embed = getattr(item_post, "embed", None)
    got = safe_get_external(view_embed)
    if got:
        return got

    # Try record embed
    rec_embed = getattr(record, "embed", None) if record else None
    got = safe_get_external(rec_embed)
    if got:
        return got

    # Some implementations keep external directly (rare)
    if view_embed and (getattr(view_embed, "title", None) and (getattr(view_embed, "uri", None) or getattr(view_embed, "url", None))):
        return (
            getattr(view_embed, "title", None),
            getattr(view_embed, "description", None),
            getattr(view_embed, "uri", None) or getattr(view_embed, "url", None),
        )

    return (None, None, None)

def flatten_item(item: models.AppBskyFeedDefs.FeedViewPost) -> Dict[str, Any]:
    post = item.post
    record = getattr(post, "record", None)
    author = getattr(post, "author", None)
    reply = getattr(item, "reply", None)
    reason = getattr(item, "reason", None)

    def get_uri(x):
        return getattr(x, "uri", None) if x else None

    reply_root_uri = get_uri(getattr(reply, "root", None)) if reply else None
    reply_parent_uri = get_uri(getattr(reply, "parent", None)) if reply else None

    news_title, news_description, news_uri = extract_news_embed(post, record)

    return {
        "uri": getattr(post, "uri", None),
        "cid": getattr(post, "cid", None),
        "author_handle": getattr(author, "handle", None) if author else None,
        "author_did": getattr(author, "did", None) if author else None,
        "indexedAt": getattr(post, "indexed_at", None),
        "createdAt": getattr(record, "created_at", None) if record else None,
        "text": getattr(record, "text", None) if record else None,
        "reply_root_uri": reply_root_uri,
        "reply_parent_uri": reply_parent_uri,
        "is_repost": 1 if (reason and getattr(reason, "$type", "") == "app.bsky.feed.defs#reasonRepost") else 0,
        "like_count": getattr(post, "like_count", 0) or 0,
        "repost_count": getattr(post, "repost_count", 0) or 0,
        "reply_count": getattr(post, "reply_count", 0) or 0,
        "quote_count": getattr(post, "quote_count", 0) or 0,
        "news_title": news_title,
        "news_description": news_description,
        "news_uri": news_uri,
    }

def fetch_author_feed_all(
    client: Client,
    actor: str,
    posts_pbar: tqdm,
    include_pins: bool,
    cutoff_dt: Optional[datetime],
    cutoff_check_every: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch all pages for one handle, honoring cutoff_dt (createdAt >= cutoff_dt).
    Update global posts progress bar. Return (flattened_rows, stats_dict).

    Early-stop decision is evaluated every `cutoff_check_every` pages. Rows outside
    the cutoff are never written.
    """
    handle_start = time.perf_counter()
    pages = 0
    total_posts = 0
    rows: List[Dict[str, Any]] = []

    originals = 0
    reposts = 0
    replies = 0
    sum_likes = 0
    sum_reposts = 0
    sum_replies = 0
    sum_quotes = 0
    first_dt = None
    last_dt = None

    cursor: Optional[str] = None
    page_bar = tqdm(desc=f"[{actor}] pages", unit="page", dynamic_ncols=True, leave=False)

    def within_window(created_at: Optional[str]) -> bool:
        if cutoff_dt is None:
            return True
        dt = iso_to_dt(created_at)
        return (dt is not None) and (dt >= cutoff_dt)

    while True:
        params = models.AppBskyFeedGetAuthorFeed.Params(
            actor=actor, limit=PAGE_LIMIT, cursor=cursor, include_pins=include_pins
        )
        resp = client.app.bsky.feed.get_author_feed(params)
        feed = resp.feed or []

        # If we got nothing, stop
        if not feed:
            break

        # Track last seen to show in bar
        last_created_seen = None

        page_keep: List[models.AppBskyFeedDefs.FeedViewPost] = []
        for item in feed:
            created_at = getattr(getattr(item.post, "record", None), "created_at", None)
            last_created_seen = created_at or last_created_seen
            if within_window(created_at):
                page_keep.append(item)

        # Always keep in-window posts on this page
        for item in page_keep:
            flat = flatten_item(item)
            rows.append(flat)
            total_posts += 1
            posts_pbar.update(1)

            sum_likes += flat["like_count"]
            sum_reposts += flat["repost_count"]
            sum_replies += flat["reply_count"]
            sum_quotes += flat["quote_count"]

            if flat["is_repost"]:
                reposts += 1
            else:
                if flat["reply_parent_uri"]:
                    replies += 1
                else:
                    originals += 1

            dt = iso_to_dt(flat["createdAt"])
            if dt:
                if not first_dt or dt < first_dt:
                    first_dt = dt
                if not last_dt or dt > last_dt:
                    last_dt = dt

        cursor = resp.cursor
        pages += 1

        # Page bar description with last seen timestamp and window status
        status_suffix = ""
        if last_created_seen:
            dt_seen = iso_to_dt(last_created_seen)
            if dt_seen:
                w = (cutoff_dt is None) or (dt_seen >= cutoff_dt)
                status_suffix = f" — last: {dt_seen.isoformat()} — within window: {'yes' if w else 'no'}"
        page_bar.set_description(f"[{actor}] pages{status_suffix}")
        page_bar.update(1)

        # Early-stop decision: if nothing on the page was in-window AND it's time to check
        all_outside = (cutoff_dt is not None and not page_keep)
        should_check_now = (pages % max(1, cutoff_check_every) == 0)

        if all_outside and should_check_now:
            break

        if not cursor:
            break
        time.sleep(SLEEP_SEC)

    page_bar.close()
    elapsed = time.perf_counter() - handle_start
    rate = total_posts / elapsed if elapsed > 0 else 0.0

    stats = {
        "handle": actor,
        "pages": pages,
        "posts": total_posts,
        "originals": originals,
        "reposts": reposts,
        "replies": replies,
        "sum_likes": sum_likes,
        "sum_reposts": sum_reposts,
        "sum_replies": sum_replies,
        "sum_quotes": sum_quotes,
        "avg_likes": (sum_likes / total_posts) if total_posts else 0.0,
        "avg_reposts": (sum_reposts / total_posts) if total_posts else 0.0,
        "avg_replies": (sum_replies / total_posts) if total_posts else 0.0,
        "avg_quotes": (sum_quotes / total_posts) if total_posts else 0.0,
        "first_post": dt_to_iso(first_dt),
        "last_post": dt_to_iso(last_dt),
        "elapsed_sec": elapsed,
        "rate_posts_per_sec": rate,
    }
    return rows, stats

def final_report(per_handle_stats: List[Dict[str, Any]]) -> None:
    # Aggregate totals
    lines = []
    grand = defaultdict(float)
    for s in per_handle_stats:
        for k, v in s.items():
            if k in {"handle","first_post","last_post"}:
                continue
            if isinstance(v, (int, float)):
                grand[k] += v

    lines.append("\n" + "="*72)
    lines.append("FINAL REPORT")
    lines.append("="*72)
    for s in per_handle_stats:
        lines.append(f"\nHandle: {s['handle']}")
        lines.append(f"  Pages fetched         : {s['pages']}")
        lines.append(f"  Posts fetched         : {s['posts']}")
        lines.append(f"    - originals         : {s['originals']}")
        lines.append(f"    - replies           : {s['replies']}")
        lines.append(f"    - reposts           : {s['reposts']}")
        lines.append(f"  Engagement (sums)")
        lines.append(f"    - likes             : {s['sum_likes']}")
        lines.append(f"    - reposts           : {s['sum_reposts']}")
        lines.append(f"    - replies           : {s['sum_replies']}")
        lines.append(f"    - quotes            : {s['sum_quotes']}")
        lines.append(f"  Engagement (averages per post)")
        lines.append(f"    - likes             : {s['avg_likes']:.2f}")
        lines.append(f"    - reposts           : {s['avg_reposts']:.2f}")
        lines.append(f"    - replies           : {s['avg_replies']:.2f}")
        lines.append(f"    - quotes            : {s['avg_quotes']:.2f}")
        lines.append(f"  Time range            : {s['first_post']}  →  {s['last_post']}")
        lines.append(f"  Time taken            : {s['elapsed_sec']:.2f}s")
        lines.append(f"  Effective rate        : {s['rate_posts_per_sec']:.2f} posts/sec")

    total_posts = int(grand["posts"])
    lines.append("\n" + "-"*72)
    lines.append("All handles combined")
    lines.append("-"*72)
    lines.append(f"  Total pages           : {int(grand['pages'])}")
    lines.append(f"  Total posts           : {total_posts}")
    lines.append(f"    - originals         : {int(grand['originals'])}")
    lines.append(f"    - replies           : {int(grand['replies'])}")
    lines.append(f"    - reposts           : {int(grand['reposts'])}")
    lines.append(f"  Engagement (sums)")
    lines.append(f"    - likes             : {int(grand['sum_likes'])}")
    lines.append(f"    - reposts           : {int(grand['sum_reposts'])}")
    lines.append(f"    - replies           : {int(grand['sum_replies'])}")
    lines.append(f"    - quotes            : {int(grand['sum_quotes'])}")
    if total_posts:
        lines.append(f"  Engagement (averages per post)")
        lines.append(f"    - likes             : {grand['sum_likes']/total_posts:.2f}")
        lines.append(f"    - reposts           : {grand['sum_reposts']/total_posts:.2f}")
        lines.append(f"    - replies           : {grand['sum_replies']/total_posts:.2f}")
        lines.append(f"    - quotes            : {grand['sum_quotes']/total_posts:.2f}")
    
    return "\n".join(lines)




class Fetcher():
    """Fetcher to get public Bluesky posts with max-age cutoff, incremental updates, and embed extraction."""
    def __init__(self, xrpc_base=APPVIEW_XRPC):
        self.client = Client(xrpc_base)

    def fetch(self, handles=[
            "news-flows-nl.bsky.social",
            "news-flows-ir.bsky.social",
            "news-flows-cz.bsky.social",
            "news-flows-fr.bsky.social",
        ],
        max_age_days=7,
        cutoff_check_every=1,
        include_pins=False):
        """Fetch public Bluesky posts.

        Parameters:

        handles (list of strings): Bluesky handles to fetch
        max_age_days (int): Only keep posts whose createdAt is within the last N days; pagination stops early when older content is reached.
        cutoff_check_every (int)": How many pages between early-stop checks against the cutoff (default: 1 = check every page).
        include_pins (bool): Include pinned posts at the top
        """

        posts_pbar = tqdm(desc="Posts fetched (all handles)", unit="post", dynamic_ncols=True)
        combined_new_rows: List[Dict[str, Any]] = []
        per_handle_stats: List[Dict[str, Any]] = []

        handles_pbar = tqdm(handles, desc="Handles", unit="handle", dynamic_ncols=True)
        start_all = time.perf_counter()

        for handle in handles_pbar:
            out_path = f"{handle.replace('.', '_')}_author_feed.csv"

            # Ensure header exists before reading
            _ = ensure_headers(out_path)

            # Incremental cutoff per handle:
            # Use later of (now - N days) and latest saved createdAt in CSV (if present).
            cutoff_dt: Optional[datetime] = None
            if max_age_days is not None:
                rel_cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
                latest_saved = latest_created_at_in_csv(out_path)
                cutoff_dt = max(latest_saved, rel_cutoff) if latest_saved else rel_cutoff

            # Fetch rows for the handle
            handle_rows, stats = fetch_author_feed_all(
                client=self.client,
                actor=handle,
                posts_pbar=posts_pbar,
                include_pins=include_pins,
                cutoff_dt=cutoff_dt,
                cutoff_check_every=cutoff_check_every,
            )

            # Prepare append with de-dup
            existing_uris = load_existing_uris(out_path)
            new_rows = [r for r in handle_rows if r.get("uri") and r["uri"] not in existing_uris]

            # Append only new rows
            append_rows(out_path, new_rows)

            # Track combined (only newly added this run)
            combined_new_rows.extend(new_rows)

            # Report: how many NEW and their time frame
            if new_rows:
                dts = [iso_to_dt(r.get("createdAt")) for r in new_rows if r.get("createdAt")]
                dts = [d for d in dts if d is not None]
                new_min = min(dts).isoformat() if dts else None
                new_max = max(dts).isoformat() if dts else None
                handles_pbar.write(
                    f"✅ DONE {handle}: added {len(new_rows)} new posts "
                    f"({new_min} → {new_max}) → {out_path}"
                )
            else:
                handles_pbar.write(
                    f"✅ DONE {handle}: no new posts to append → {out_path}"
                )

            # Keep scrape stats (before de-dup)
            per_handle_stats.append(stats)

        elapsed_all = time.perf_counter() - start_all
        posts_pbar.close()

        # Combined CSV: append only newly added rows across all handles in this run.
        combined_path = "all_handles_author_feed.csv"
        ensure_headers(combined_path)
        existing_combined_uris = load_existing_uris(combined_path)
        combined_to_append = [r for r in combined_new_rows if r["uri"] not in existing_combined_uris]
        append_rows(combined_path, combined_to_append)

        logger.debug(
            f"\nWrote/updated combined CSV with +{len(combined_to_append)} new rows → {combined_path} "
            f"({elapsed_all:.2f}s total)."
        )

        # Final detailed report for the run (scraped rows prior to de-dup)
        return final_report(per_handle_stats)



def main():
    parser = argparse.ArgumentParser(description="Fetch public Bluesky posts with progress, max-age cutoff, incremental updates, and embed extraction.")
    parser.add_argument(
        "--handles",
        nargs="+",
        default=[
            "news-flows-nl.bsky.social",
            "news-flows-ir.bsky.social",
            "news-flows-cz.bsky.social",
            "news-flows-fr.bsky.social",
        ],
        help="Bluesky handles to fetch (space-separated)."
    )
    parser.add_argument(
        "--xrpc-base",
        default=APPVIEW_XRPC,
        help="XRPC base URL (default: AppView public API)."
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="Only keep posts whose createdAt is within the last N days; pagination stops early when older content is reached."
    )
    parser.add_argument(
        "--cutoff-check-every",
        type=int,
        default=1,
        help="How many pages between early-stop checks against the cutoff (default: 1 = check every page)."
    )
    parser.add_argument(
        "--include-pins",
        dest="include_pins",
        action="store_true",
        help="Include pinned posts at the top (default: False)."
    )
    parser.add_argument(
        "--no-include-pins",
        dest="include_pins",
        action="store_false",
        help="Exclude pinned posts (default)."
    )
    parser.set_defaults(include_pins=False)
    args = parser.parse_args()

    fetcher = Fetcher(args.xrpc_base)
    results = fetcher.fetch(handles = args.handles, max_age_days=args.max_age_days, cutoff_check_every=args.cutoff_check_every, include_pins=args.include_pins)
    print(results)

   

if __name__ == "__main__":
    main()
