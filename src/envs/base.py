from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, Any


@dataclass
class EnvConfig:
    env_id: str
    seed: int = 0
    render_mode: str = "rgb_array"


class ActionMapper(Protocol):
    def to_env(self, policy_action: int) -> int: ...
    def n_actions(self) -> int: ...


class EnvWrapper(Protocol):
    """Protocol for environment wrappers used by training/inference."""

    def reset(self, seed: int | None = None) -> Tuple[Any, dict]: ...
    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]: ...
    def render(self) -> Any: ...
    def close(self) -> None: ...

