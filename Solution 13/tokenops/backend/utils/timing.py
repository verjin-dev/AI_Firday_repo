"""@timed decorator — records elapsed_ms onto result objects where possible."""
from __future__ import annotations

import functools
import time
from typing import Any, Callable


def timed(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = int((time.perf_counter() - t0) * 1000)
        if isinstance(result, dict) and "elapsed_ms" not in result:
            result["elapsed_ms"] = elapsed
        elif hasattr(result, "elapsed_ms") and getattr(result, "elapsed_ms", None) in (None, 0):
            try:
                setattr(result, "elapsed_ms", elapsed)
            except Exception:
                pass
        return result

    return wrapper


class Stopwatch:
    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._t0) * 1000)
