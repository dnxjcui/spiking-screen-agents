from __future__ import annotations

from typing import Literal

from .snn_policy import SNNPolicy


ModelName = Literal["snn"]


def make_model(name: ModelName, *, in_ch: int, n_actions: int, beta: float, device: str):
    if name == "snn":
        return SNNPolicy(in_ch=in_ch, n_actions=n_actions, beta=beta, device=device)
    raise ValueError(f"Unknown model '{name}'")

