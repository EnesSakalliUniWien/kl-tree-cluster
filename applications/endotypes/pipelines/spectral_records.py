"""Shared row fields for endotype spectral-block outputs."""

from __future__ import annotations

import math
from typing import Any


def spectral_block_record(
    block: Any,
    *,
    block_energy: float,
    diagnostics: dict[str, Any],
) -> dict[str, object]:
    """Return the common metadata fields for one spectral block."""

    return {
        "block_id": int(block.block_id),
        "block_name": block.block_name,
        "block_type": block.block_type,
        "block_start": int(block.block_start),
        "block_end": int(block.block_end),
        "subspace_dimensions": int(block.block_end - block.block_start + 1),
        "block_energy_fraction": block_energy,
        "segmentation_bic": diagnostics.get("bic", math.nan),
        "common_mode_isolated": diagnostics.get("common_mode_isolated", False),
    }
