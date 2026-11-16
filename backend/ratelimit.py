"""A small in-memory sliding-window rate limiter (per process)."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Hashable


class SlidingWindowLimiter:
    def __init__(
        self,
        limit: int,
        window: float = 60.0,
        *,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window = window
        self.max_keys = max_keys
        self._clock = clock
        self._hits: OrderedDict[Hashable, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, key: Hashable) -> float | None:
        """Record a request. Returns None if allowed, else seconds until a slot frees up."""
        if self.limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = deque()
                self._hits[key] = hits
            self._hits.move_to_end(key)
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(0.0, hits[0] + self.window - now)
            hits.append(now)
            while len(self._hits) > self.max_keys:
                self._hits.popitem(last=False)
        return None
