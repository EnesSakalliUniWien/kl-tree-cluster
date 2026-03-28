from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .models import SignalNeighborInfo, StoppingEdgeInfo

STOPPING_EDGE_INFO_ATTR_KEY = "_stopping_edge_info"
_REQUIRED_ATTR_FIELDS = (
    "child_node_ids",
    "stopping_edge_p_values",
    "distances_to_stopping_edge",
    "signal_p_values",
    "distances_to_signal",
)


def _malformed_payload_error(message: str) -> ValueError:
    return ValueError(f"Malformed {STOPPING_EDGE_INFO_ATTR_KEY} attrs payload: {message}")


def _coerce_float_array(
    *,
    field_name: str,
    raw_values: object,
    expected_length: int,
) -> np.ndarray:
    values = np.asarray(raw_values, dtype=float)
    if values.ndim != 1:
        raise _malformed_payload_error(f"{field_name!r} must be 1-D.")
    if len(values) != expected_length:
        raise _malformed_payload_error(
            f"{field_name!r} has length {len(values)}, expected {expected_length}."
        )
    return values.copy()


@dataclass(frozen=True)
class StoppingEdgeAttrPayload:
    child_node_ids: tuple[str, ...]
    stopping_edge_p_values: np.ndarray
    distances_to_stopping_edge: np.ndarray
    signal_p_values: np.ndarray
    distances_to_signal: np.ndarray


def build_stopping_edge_attrs(
    *,
    child_node_ids: list[str],
    stopping_edge_info_by_child: Mapping[str, StoppingEdgeInfo],
    signal_neighbor_info_by_child: Mapping[str, SignalNeighborInfo],
) -> dict[str, object]:
    child_ids = tuple(child_node_ids)
    stopping_edge_p_values = np.full(len(child_ids), np.nan)
    distances_to_stopping_edge = np.full(len(child_ids), np.nan)
    signal_p_values = np.full(len(child_ids), np.nan)
    distances_to_signal = np.full(len(child_ids), np.nan)

    for index, child_id in enumerate(child_ids):
        child_stopping_edge_info = stopping_edge_info_by_child.get(child_id)
        if child_stopping_edge_info is not None:
            stopping_edge_p_values[index] = child_stopping_edge_info.stopping_edge_p_value
            distances_to_stopping_edge[index] = child_stopping_edge_info.distance_to_stopping_edge

        child_signal_neighbor_info = signal_neighbor_info_by_child.get(child_id)
        if child_signal_neighbor_info is not None:
            signal_p_values[index] = child_signal_neighbor_info.sig_p_value
            distances_to_signal[index] = child_signal_neighbor_info.distance_to_sig

    return {
        "child_node_ids": list(child_ids),
        "stopping_edge_p_values": stopping_edge_p_values,
        "distances_to_stopping_edge": distances_to_stopping_edge,
        "signal_p_values": signal_p_values,
        "distances_to_signal": distances_to_signal,
    }


def parse_stopping_edge_attrs(
    attrs: Mapping[str, object],
) -> StoppingEdgeAttrPayload | None:
    attrs_value = attrs.get(STOPPING_EDGE_INFO_ATTR_KEY)
    if attrs_value is None:
        return None
    if not isinstance(attrs_value, Mapping):
        raise _malformed_payload_error("expected a mapping.")

    missing_fields = [
        field_name for field_name in _REQUIRED_ATTR_FIELDS if field_name not in attrs_value
    ]
    if missing_fields:
        raise _malformed_payload_error(f"missing fields {missing_fields!r}.")

    child_node_ids = tuple(str(child_node_id) for child_node_id in attrs_value["child_node_ids"])
    expected_length = len(child_node_ids)
    return StoppingEdgeAttrPayload(
        child_node_ids=child_node_ids,
        stopping_edge_p_values=_coerce_float_array(
            field_name="stopping_edge_p_values",
            raw_values=attrs_value["stopping_edge_p_values"],
            expected_length=expected_length,
        ),
        distances_to_stopping_edge=_coerce_float_array(
            field_name="distances_to_stopping_edge",
            raw_values=attrs_value["distances_to_stopping_edge"],
            expected_length=expected_length,
        ),
        signal_p_values=_coerce_float_array(
            field_name="signal_p_values",
            raw_values=attrs_value["signal_p_values"],
            expected_length=expected_length,
        ),
        distances_to_signal=_coerce_float_array(
            field_name="distances_to_signal",
            raw_values=attrs_value["distances_to_signal"],
            expected_length=expected_length,
        ),
    )


__all__ = [
    "STOPPING_EDGE_INFO_ATTR_KEY",
    "StoppingEdgeAttrPayload",
    "build_stopping_edge_attrs",
    "parse_stopping_edge_attrs",
]
