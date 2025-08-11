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
    lr: float = 3e-4                   # learning rate 
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


@dataclass 
class DQNConfig:
    """DQN-specific hyperparameters based on reference implementation."""
    # DQN hyperparameters
    learning_rate: float = 1e-3        # learning rate (alpha)
    discount_factor: float = 0.99      # discount rate (gamma)  
    network_sync_rate: int = 1000      # steps before syncing target network
    replay_memory_size: int = 50000    # replay buffer size
    mini_batch_size: int = 32          # batch size for training
    epsilon_init: float = 1.0          # initial exploration rate
    epsilon_decay: float = 0.9995      # epsilon decay per step
    epsilon_min: float = 0.01          # minimum exploration rate
    stop_on_reward: float = 500.0      # stop training when reaching this reward
    hidden_dim: int = 256              # hidden layer size
    enable_double_dqn: bool = True     # use double DQN
    enable_dueling_dqn: bool = True    # use dueling DQN
    # Training
    max_episodes: int = 1000           # maximum episodes to train
    save_frequency: int = 100          # save model every N episodes
    log_frequency: int = 10            # log progress every N episodes


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters based on reference implementation."""
    # PPO hyperparameters
    learning_rate: float = 3e-4        # learning rate
    gamma: float = 0.99                # discount factor
    lmbda: float = 0.95                # GAE lambda parameter
    eps_clip: float = 0.2              # PPO clip parameter
    k_epoch: int = 4                   # number of epochs per update
    v_coef: float = 0.5                # value loss coefficient
    entropy_coef: float = 0.01         # entropy coefficient
    memory_size: int = 2048            # trajectory memory size
    update_freq: int = 1               # update frequency (episodes)
    plot_every: int = 100              # plot results every N episodes
    # Training
    train_cartpole: bool = True        # training on cartpole
    max_episodes: int = 1000           # maximum episodes to train
    log_frequency: int = 10            # log progress every N episodes

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

