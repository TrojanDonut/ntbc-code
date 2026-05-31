"""Wall-clock timing helpers used in the eval pipeline.

We always do a small warmup (to amortize Python and torch lazy init) followed
by N timed repeats; the report uses the median to reduce noise.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable


def median_time(fn: Callable[[], None], *, repeats: int = 5, warmup: int = 1) -> float:
    """Return the median wall-clock time (seconds) of ``fn`` over ``repeats`` runs."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)
