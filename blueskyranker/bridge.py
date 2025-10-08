#!/usr/bin/env python3
"""
Bridge helpers used by the R reticulate layer.

These functions keep the query logic inside Python so that ranking always runs
against a fresh SQLite query instead of reusing Polars objects that travelled
through reticulate. The helpers deliberately return plain Python data
structures (lists/dicts) to make conversion on the R side predictable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sqlite3

from .fetcher import ensure_db, iso_to_dt, DEFAULT_SQLITE_PATH


COLUMN_ALIASES: Dict[str, str] = {
    "handle": "author_handle",
}


@dataclass
class TimeFilters:
    min_ns: Optional[int] = None
    max_ns: Optional[int] = None
    max_comparison: str = "le"  # 'le' | 'lt'

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "TimeFilters":
        def _iso_to_ns(value: Optional[str]) -> Optional[int]:
            if not value:
                return None
            dt = iso_to_dt(value)
            if dt is None:
                # try parsing manually; assume UTC if naive
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return int(dt.timestamp() * 1_000_000_000)

        min_iso = spec.get("min_created_at")
        max_iso = spec.get("max_created_at")
        max_cmp = spec.get("max_comparison") or "le"
        return cls(
            min_ns=_iso_to_ns(min_iso),
            max_ns=_iso_to_ns(max_iso),
            max_comparison="lt" if max_cmp == "lt" else "le",
        )

    def has_filters(self) -> bool:
        return self.min_ns is not None or self.max_ns is not None


def _map_column(column: str) -> str:
    return COLUMN_ALIASES.get(column, column)


def _normalise_spec(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Copy the user-supplied spec and fill defaults."""
    base: Dict[str, Any] = dict(spec or {})
    base.setdefault("sqlite_path", DEFAULT_SQLITE_PATH)
    base.setdefault("order_by", "createdAt")
    base["descending"] = bool(base.get("descending", False))
    # Normalise empty strings/NULL-like values to None for known keys
    for key in ("handle", "min_created_at", "max_created_at"):
        if base.get(key) in ("", None):
            base[key] = None
    if base.get("max_comparison") not in ("lt", "le"):
        base["max_comparison"] = "le"
    return base


def _build_time_conditions(filters: TimeFilters) -> Tuple[List[str], List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    if filters.min_ns is not None:
        clauses.append("(createdAt_ns IS NULL OR createdAt_ns >= ?)")
        params.append(filters.min_ns)
    if filters.max_ns is not None:
        op = "<" if filters.max_comparison == "lt" else "<="
        clauses.append(f"(createdAt_ns IS NULL OR createdAt_ns {op} ?)")
        params.append(filters.max_ns)
    return clauses, params


def _build_where_clause(
    spec: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    *,
    include_time_filters: bool = True,
) -> Tuple[List[str], List[Any], TimeFilters]:
    clauses: List[str] = []
    params: List[Any] = []
    overrides = overrides or {}

    # Merge base + overrides for equality filters
    def merge_value(key: str) -> Optional[Any]:
        return overrides.get(key) if overrides.get(key) not in (None, "") else spec.get(key)

    # Handle/author filters
    handle_value = merge_value("handle")
    if handle_value:
        column = _map_column("handle")
        clauses.append(f"{column} = ?")
        params.append(handle_value)

    # Additional equality overrides (e.g., author_handle, language)
    for key, value in overrides.items():
        if key in {"handle", "min_created_at", "max_created_at", "limit", "max_comparison"}:
            continue
        if value in (None, ""):
            continue
        column = _map_column(key)
        clauses.append(f"{column} = ?")
        params.append(value)

    # Time filters
    time_spec = dict(spec)
    time_spec["min_created_at"] = merge_value("min_created_at")
    time_spec["max_created_at"] = merge_value("max_created_at")
    time_spec["max_comparison"] = overrides.get("max_comparison") or spec.get("max_comparison")
    filters = TimeFilters.from_spec(time_spec)
    if include_time_filters:
        time_clauses, time_params = _build_time_conditions(filters)
        clauses.extend(time_clauses)
        params.extend(time_params)

    return clauses, params, filters


def _prepare_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure downstream requirements: non-empty text and explicit handle."""
    result = dict(row)
    author_handle = result.get("author_handle")
    if "handle" not in result or result.get("handle") in (None, ""):
        result["handle"] = author_handle

    text = result.get("text") or ""
    if isinstance(text, str):
        text = text.strip()
    else:
        text = str(text or "").strip()

    if not text:
        for key in ("news_title", "news_description", "news_uri", "uri"):
            value = result.get(key)
            if value:
                text_candidate = str(value).strip()
                if text_candidate:
                    text = text_candidate
                    break
    if not text:
        seed = result.get("cid") or result.get("uri") or result.get("author_handle") or "unknown"
        text = f"placeholder_post_{abs(hash(str(seed))) % 1_000_000}"
    result["text"] = text
    return result


def distinct_group_values(spec: Dict[str, Any], columns: Sequence[str]) -> List[Dict[str, Any]]:
    """Return distinct combinations for the requested columns under the spec filters."""
    spec = _normalise_spec(spec)
    if isinstance(columns, str):
        columns = [columns]
    if not columns:
        return []

    conn = ensure_db(spec["sqlite_path"])
    select_parts = []
    order_parts = []
    for col in columns:
        mapped = _map_column(col)
        if mapped == col:
            select_parts.append(mapped)
            order_parts.append(mapped)
        else:
            select_parts.append(f"{mapped} AS {col}")
            order_parts.append(col)

    base_sql = f"SELECT DISTINCT {', '.join(select_parts)} FROM posts"
    where, params, _ = _build_where_clause(spec)
    if where:
        base_sql += " WHERE " + " AND ".join(where)
    base_sql += f" ORDER BY {', '.join(order_parts)}"

    cur = conn.execute(base_sql, params)
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        record = {col: row[idx] for idx, col in enumerate(columns)}
        out.append(record)
    return out


def load_posts_for_ranking(
    spec: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    *,
    fallback_to_full: bool = False,
) -> Dict[str, Any]:
    """
    Load posts applying the base spec plus optional overrides (e.g., for per-group queries).

    Returns a dict with:
      - rows: list of dicts (ready for TopicRanker)
      - fallback_used: True if a second query without time filters was required
    """
    spec = _normalise_spec(spec)
    overrides = overrides or {}

    def _run_query(include_time_filters: bool) -> Tuple[List[Dict[str, Any]], TimeFilters]:
        where, params, filters = _build_where_clause(spec, overrides, include_time_filters=include_time_filters)
        order_column = _map_column(overrides.get("order_by") or spec.get("order_by") or "createdAt")
        descending = bool(overrides.get("descending", spec["descending"]))
        limit_value = overrides.get("limit", spec.get("limit"))

        sql = f"SELECT * FROM posts"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order_column} {'DESC' if descending else 'ASC'}"
        if limit_value:
            sql += f" LIMIT {int(limit_value)}"

        conn = ensure_db(spec["sqlite_path"])
        cur = conn.execute(sql, params)
        colnames = [desc[0] for desc in cur.description]
        raw_rows = [dict(zip(colnames, values)) for values in cur.fetchall()]
        prepared = [_prepare_row(row) for row in raw_rows]
        return prepared, filters

    rows, filters = _run_query(include_time_filters=True)
    fallback_used = False
    if fallback_to_full and not rows and filters.has_filters():
        rows, _ = _run_query(include_time_filters=False)
        fallback_used = bool(rows)
    return {"rows": rows, "fallback_used": fallback_used}


__all__ = ["distinct_group_values", "load_posts_for_ranking"]
