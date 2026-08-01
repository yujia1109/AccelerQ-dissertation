"""Small, dependency-free helpers for ADAPT generator selection."""

from __future__ import annotations

from collections.abc import Mapping


def select_precise_gradient_index(
    precise_gradients: Mapping[int, complex],
    previous_index: int | None,
    check_duplicate: bool,
) -> tuple[int | None, int | None]:
    """Return (selected index, newly ignored duplicate index).

    A ``None`` selected index means that no generator remains eligible. This
    is a normal terminal state after zero-angle candidates have been ignored.
    """

    ranked = sorted(
        precise_gradients,
        key=lambda index: abs(precise_gradients[index]),
        reverse=True,
    )
    if not ranked:
        return None, None
    if (
        check_duplicate
        and previous_index is not None
        and len(ranked) >= 2
        and ranked[0] == previous_index
    ):
        return ranked[1], ranked[0]
    return ranked[0], None
