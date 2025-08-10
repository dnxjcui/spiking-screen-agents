"""
Generic streaming interfaces for neuromorphic event data.

Defines a thread-safe event buffer and a base receiver interface so
different transport protocols (e.g., UDP, TCP, file replay) can plug in.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from typing import Deque, Dict, List, Optional


DOT_FADE_DURATION_SECONDS: float = 0.1


class EventBuffer:
    """Thread-safe container for neuromorphic event data.

    Event schema (dict per event):
      - 'timestamp': int (microseconds)
      - 'x': int
      - 'y': int
      - 'polarity': int (signed 8-bit semantics; negative for OFF)
      - 'received_time': float (seconds, time.time())
    """

    def __init__(self, max_events: int = 100_000) -> None:
        self.events: Deque[Dict] = deque(maxlen=max_events)
        self.lock = threading.Lock()
        self.stats = {"packets": 0, "events": 0, "bytes": 0}

    def add_events(self, events: List[Dict]) -> None:
        with self.lock:
            current_time = time.time()
            for e in events:
                e.setdefault("received_time", current_time)
                self.events.append(e)
            self.stats["events"] += len(events)

    def get_recent_events(self, time_window: float = DOT_FADE_DURATION_SECONDS) -> List[Dict]:
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - time_window
            return [e for e in self.events if e.get("received_time", 0.0) >= cutoff_time]

    def clear(self) -> None:
        with self.lock:
            self.events.clear()


class BaseEventReceiver:
    """Base interface for event receivers.

    Concrete implementations must populate `event_buffer` and update its stats.
    """

    def __init__(self) -> None:
        self.event_buffer = EventBuffer()
        self.running: bool = False

    def start(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def is_running(self) -> bool:
        return self.running

