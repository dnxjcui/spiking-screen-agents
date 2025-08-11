from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TraditionalActorCritic(nn.Module):
    """Traditional (non-spiking) Actor-Critic for vision-based tasks like Pong.
    
    Uses standard CNN layers with ReLU activations.
    Input: (B, C=2, H=84, W=84)
    Outputs: policy logits (B, A), value (B, 1)
    """
    
    def __init__(self, in_ch: int = 2, n_actions: int = 3, device: str = "cpu"):
        super().__init__()
        self.device = device
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        # Fully connected layers
        self.fc_in = 64 * 11 * 11
        self.fc = nn.Linear(self.fc_in, 128)
        
        # Output heads
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)
        
        self.to(self.device)
    
    def reset_state(self) -> None:
        """No-op for traditional models (no hidden state)."""
        pass
    
    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        
        # Conv layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten and fully connected
        x = x.flatten(1)
        x = F.relu(self.fc(x))
        
        # Output heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value


class TraditionalStateActorCritic(nn.Module):
    """Traditional (non-spiking) Actor-Critic for state-based tasks like CartPole.
    
    Uses standard fully-connected layers with ReLU activations.
    Input: (B, state_dim) - raw state values
    Outputs: policy logits (B, A), value (B, 1)
    """
    
    def __init__(self, state_dim: int = 4, n_actions: int = 2, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        
        # Shared network layers (similar to reference DQN but adapted for actor-critic)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Separate output heads 
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
        self.to(self.device)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def reset_state(self) -> None:
        """No-op for traditional models (no hidden state)."""
        pass
    
    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        
        # Shared feature extraction
        features = self.shared(x)
        
        # Output heads
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value


class SimpleStateActorCritic(nn.Module):
    """Very simple traditional Actor-Critic for debugging.
    
    Minimal architecture to test if the issue is model complexity.
    """
    
    def __init__(self, state_dim: int = 4, n_actions: int = 2, device: str = "cpu"):
        super().__init__()
        self.device = device
        
        # Simple but appropriately sized network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Output heads
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
        self.to(self.device)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def reset_state(self) -> None:
        """No-op for traditional models (no hidden state)."""
        pass
    
    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        
        # Simple forward pass
        features = self.shared(x)
        
        # Output heads
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value