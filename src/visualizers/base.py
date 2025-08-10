from __future__ import annotations

from typing import Protocol, List, Dict


class EventSource(Protocol):
    """Minimal protocol any event source must provide to visualizers."""

    def get_recent_events(self, time_window: float) -> List[Dict]:
        ...


class BaseVisualizer:
    """Base visualizer interface."""

    def update(self, source: EventSource) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def render(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

