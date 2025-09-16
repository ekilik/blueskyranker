#!/usr/bin/env python3
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / 'example.ipynb'

md = (
    "### Pipeline updates (priority and demotion)\n\n"
    "- Priority assignment now starts at 1000 for the first item and decreases by 1 (1000, 999, 998, …). The minimum is clamped at 1. Items explicitly demoted are sent with priority 0.\n"
    "- Demotion: by default, all posts from the last 48 hours that are not in the current prioritisation are sent with priority 0. Configure via `--demote-window-hours`.\n"
    "- Export filenames use a human‑readable UTC timestamp: `push_{handle}_{YYYY-MM-DDTHH-mm-ssZ}.json`.\n"
    "- Server responses: short responses print to stdout; long responses are saved to `push_exports/prioritize_response_{handle}_{YYYY-MM-DDTHH-mm-ssZ}.{json|txt}`.\n\n"
    "Example CLI:\n\n"
    "```") + (
    "\npython -m blueskyranker.pipeline \\\n+  --handles news-flows-nl.bsky.social news-flows-fr.bsky.social \\\n+  --method networkclustering-tfidf \\\n+  --similarity-threshold 0.2 \\\n+  --cluster-window-days 7 \\\n+  --engagement-window-days 1 \\\n+  --push-window-days 2 \\\n+  --demote-last \\\n+  --demote-window-hours 48 \\\n+  --log-path push.log \\\n+  --no-test\n") + (
    "```\n\n"
    "Programmatic call:\n\n"
    "```python\n"
    "from blueskyranker.pipeline import run_fetch_rank_push\n"
    "run_fetch_rank_push(\n"
    "    handles=['news-flows-nl.bsky.social'],\n"
    "    method='networkclustering-tfidf', similarity_threshold=0.2,\n"
    "    cluster_window_days=7, engagement_window_days=1, push_window_days=2,\n"
    "    demote_last=True, demote_window_hours=48,\n"
    "    include_pins=False, test=True, log_path='push.log')\n"
    "```\n"
)

cell = {"cell_type": "markdown", "metadata": {}, "source": md}

# Additional cell documenting ordering logic
md2 = (
    "### Ordering logic (time windows)\n\n"
    "- Clustering window: clusters are built from posts in this window (e.g., 7 days).\n"
    "- Engagement window: cluster engagement is computed here to derive `cluster_engagement_rank` (1 = most engaged).\n"
    "- Push window: only posts in this window are eligible for the final feed.\n\n"
    "Order of posts:\n\n"
    "1) Filter to the push window.\n\n"
    "2) Order clusters by engagement rank (most engaged first).\n\n"
    "3) Within each cluster, sort by recency (newest first).\n\n"
    "4) Interleave round‑robin across clusters in rank order (1, 2, 3, … then repeat).\n\n"
    "Result: the first post is the most‑recent item from the most‑engaged cluster that has posts in the push window.\n"
)
cell2 = {"cell_type": "markdown", "metadata": {}, "source": md2}

def main():
    data = json.loads(NB_PATH.read_text(encoding='utf-8'))
    data.setdefault('cells', []).append(cell)
    data['cells'].append(cell2)
    NB_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Appended markdown cell to {NB_PATH}")

if __name__ == '__main__':
    main()
