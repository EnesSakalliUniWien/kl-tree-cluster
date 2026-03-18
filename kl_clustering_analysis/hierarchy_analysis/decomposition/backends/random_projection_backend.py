"""Deterministic per-test seeding backend.

Provides a reproducible seed derivation function used by both Gate 2 (edge)
and Gate 3 (sibling) statistical tests to ensure deterministic results
without relying on global RNG state.
"""

from __future__ import annotations

import hashlib


def derive_projection_seed_backend(base_seed: int | None, test_id: str) -> int:
    """Derive deterministic per-test seed from base seed and test id."""
    if not test_id:
        raise ValueError("test_id must be a non-empty string.")
    base = "none" if base_seed is None else str(int(base_seed))
    seed_payload = f"{base}|{test_id}".encode("utf-8")
    seed_digest = hashlib.blake2b(seed_payload, digest_size=8).digest()
    return int.from_bytes(seed_digest, byteorder="big", signed=False) & 0xFFFFFFFF


__all__ = [
    "derive_projection_seed_backend",
]
