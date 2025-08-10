from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .base import EnvConfig
from .pong_env import PongEnvWrapper


@dataclass
class EnvSpec:
    name: str
    id: str


def make_env(name: str, seed: int = 0) -> Tuple[object, object]:
    """Factory for environments.

    Currently supports only Pong. Returns (env_wrapper, action_mapper).
    """
    if name.lower() == "pong":
        cfg = EnvConfig(env_id="ALE/Pong-v5", seed=seed)
        env = PongEnvWrapper(cfg)
        return env, env.action_mapper
    raise ValueError(f"Unknown environment '{name}'.")

