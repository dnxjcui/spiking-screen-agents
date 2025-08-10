from __future__ import annotations

import gymnasium as gym

from typing import Tuple, Any

from .base import EnvConfig
from .action_mapper import ActionMapper as _ActionMapper
import ale_py



class PongEnvWrapper:
    """Thin wrapper around Gymnasium ALE/Pong with dynamic action mapping."""

    def __init__(self, cfg: EnvConfig) -> None:
        try:
            gym.register_envs(ale_py)
            self.env = gym.make(cfg.env_id, render_mode=cfg.render_mode)
        except Exception as e:
            print(f"Failed to create '{cfg.env_id}': {e}")
            raise
        self.action_mapper = _ActionMapper(self.env)

    def reset(self, seed: int | None = None) -> Tuple[Any, dict]:
        return self.env.reset(seed=seed)

    def step(self, action: int):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

