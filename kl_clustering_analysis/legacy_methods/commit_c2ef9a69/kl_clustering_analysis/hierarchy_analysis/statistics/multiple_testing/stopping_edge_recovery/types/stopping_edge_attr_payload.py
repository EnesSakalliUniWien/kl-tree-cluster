from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StoppingEdgeAttrPayload:
    child_node_ids: tuple[str, ...]
    stopping_edge_p_values: np.ndarray
    distances_to_stopping_edge: np.ndarray
    signal_p_values: np.ndarray
    distances_to_signal: np.ndarray
