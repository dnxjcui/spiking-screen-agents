from __future__ import annotations

from typing import Literal

from .snn_policy import SNNPolicy
from .snn_actor_critic import SNNActorCritic
from .snn_state_policy import SNNStateActorCritic
from .traditional_models import TraditionalActorCritic, TraditionalStateActorCritic, SimpleStateActorCritic
from .dqn_models import TraditionalDQN, SpikingDQN, VisionDQN


ModelName = Literal["snn", "snn_ac", "snn_state_ac", "traditional_ac", "traditional_state_ac", "simple_state_ac", 
                   "dqn", "snn_dqn", "vision_dqn"]


def make_model(name: ModelName, *, in_ch: int = None, state_dim: int = None, n_actions: int, beta: float = 0.95, device: str):
    # Spiking Neural Network models
    if name == "snn":
        if in_ch is None:
            raise ValueError("in_ch required for snn model")
        return SNNPolicy(in_ch=in_ch, n_actions=n_actions, beta=beta, device=device)
    if name == "snn_ac":
        if in_ch is None:
            raise ValueError("in_ch required for snn_ac model")
        return SNNActorCritic(in_ch=in_ch, n_actions=n_actions, beta=beta, device=device)
    if name == "snn_state_ac":
        if state_dim is None:
            raise ValueError("state_dim required for snn_state_ac model")
        return SNNStateActorCritic(state_dim=state_dim, n_actions=n_actions, beta=beta, device=device)
    
    # Traditional Neural Network models  
    if name == "traditional_ac":
        if in_ch is None:
            raise ValueError("in_ch required for traditional_ac model")
        return TraditionalActorCritic(in_ch=in_ch, n_actions=n_actions, device=device)
    if name == "traditional_state_ac":
        if state_dim is None:
            raise ValueError("state_dim required for traditional_state_ac model")
        return TraditionalStateActorCritic(state_dim=state_dim, n_actions=n_actions, device=device)
    if name == "simple_state_ac":
        if state_dim is None:
            raise ValueError("state_dim required for simple_state_ac model")
        return SimpleStateActorCritic(state_dim=state_dim, n_actions=n_actions, device=device)
    
    # Deep Q-Network models
    if name == "dqn":
        if state_dim is None:
            raise ValueError("state_dim required for dqn model")
        return TraditionalDQN(state_dim=state_dim, action_dim=n_actions, device=device)
    if name == "snn_dqn":
        if state_dim is None:
            raise ValueError("state_dim required for snn_dqn model")
        return SpikingDQN(state_dim=state_dim, action_dim=n_actions, beta=beta, device=device)
    if name == "vision_dqn":
        if in_ch is None:
            raise ValueError("in_ch required for vision_dqn model")
        return VisionDQN(in_channels=in_ch, action_dim=n_actions, device=device)
    
    raise ValueError(f"Unknown model '{name}'")

