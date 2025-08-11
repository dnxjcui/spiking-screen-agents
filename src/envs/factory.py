from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .base import EnvConfig
from .pong_env import PongEnvWrapper
from .cartpole_env import CartPoleEnvWrapper


@dataclass
class EnvSpec:
    name: str
    id: str


def make_env(name: str, seed: int = 0, render_mode: str = "rgb_array") -> Tuple[object, object]:
    """Factory for environments.

    Currently supports Pong and CartPole. Returns (env_wrapper, action_mapper).
    """
    if name.lower() == "pong":
        cfg = EnvConfig(env_id="ALE/Pong-v5", seed=seed, render_mode=render_mode)
        env = PongEnvWrapper(cfg)
        return env, env.action_mapper
    elif name.lower() == "cartpole":
        cfg = EnvConfig(env_id="CartPole-v1", seed=seed, render_mode=render_mode)
        env = CartPoleEnvWrapper(cfg)
        return env, env.action_mapper
    raise ValueError(f"Unknown environment '{name}'. Available: pong, cartpole")

