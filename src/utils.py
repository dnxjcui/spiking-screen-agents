import torch
import numpy as np
from dataclasses import dataclass

# ---------------------------
# Config
# ---------------------------

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Frame/Event encoding
    width: int = 84
    height: int = 84
    diff_threshold: int = 15           # intensity threshold (0-255) for ON/OFF events
    stride: int = 2                    # downsample stride before resize (for speed); 1 = no downsample
    # SNN
    beta: float = 0.95                 # LIF decay
    lr: float = 1e-3
    gamma: float = 0.99                # reward discount
    entropy_coef: float = 0.01         # encourage exploration
    value_coef: float = 0.5            # not used (pure REINFORCE); keep for future A2C
    max_grad_norm: float = 1.0
    # Training
    max_episodes: int = 50             # try 50 to start; raise to 1000+ for real training
    render: bool = False
    save_path: str = "pong_snn.pt"
    seed: int = 123
    # ALE
    env_id: str = "ALE/Pong-v5"
    frame_skip: int = 4                # typical Atari frame-skip
    noop_on_reset: bool = True

# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

