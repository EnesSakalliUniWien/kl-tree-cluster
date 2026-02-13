"""Time formatting helpers shared by benchmark entrypoints."""

from __future__ import annotations

from datetime import datetime, timezone


def format_timestamp_utc(dt: datetime | None = None) -> str:
    """Return a filesystem-safe UTC timestamp like 20250101_235959Z."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return dt.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


__all__ = ["format_timestamp_utc"]
