from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Tuple, Any

from .base import EnvConfig
from .action_mapper import ActionMapper as _ActionMapper


class CartPoleActionMapper:
    """Simple action mapper for CartPole: left/right actions."""
    
    def __init__(self, env):
        # CartPole has 2 actions: 0 (left) and 1 (right)
        self.action_list = [0, 1]
    
    def to_env(self, policy_action: int) -> int:
        return self.action_list[int(policy_action)]
    
    def n_actions(self) -> int:
        return 2


class CartPoleEnvWrapper:
    """Wrapper for CartPole-v1 environment with state-based observations."""
    
    def __init__(self, cfg: EnvConfig) -> None:
        try:
            self.env = gym.make(cfg.env_id, render_mode=cfg.render_mode)
        except Exception as e:
            print(f"Failed to create '{cfg.env_id}': {e}")
            raise
        self.action_mapper = CartPoleActionMapper(self.env)
    
    def reset(self, seed: int | None = None) -> Tuple[Any, dict]:
        return self.env.reset(seed=seed)
    
    def step(self, action: int):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self) -> None:
        self.env.close()