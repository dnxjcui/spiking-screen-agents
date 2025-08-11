from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNStateActorCritic(nn.Module):
    """Simple SNN Actor-Critic for state-based environments like CartPole.
    
    Takes raw state values (e.g., 4 values for CartPole) and processes them through
    spiking layers to produce policy and value outputs.
    
    Input: (B, state_dim) - raw state values
    Outputs: policy logits (B, A), value (B, 1)
    """
    
    def __init__(self, state_dim: int = 4, n_actions: int = 2, beta: float = 0.95, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        
        # State normalization layer - important for SNN stability
        self.state_norm = nn.LayerNorm(state_dim)
        
        # Use fast sigmoid surrogate gradient
        spike_grad = surrogate.fast_sigmoid()
        
        # Input layer - converts continuous state to spikes
        # We use a small fully connected layer to transform the input
        self.input_fc = nn.Linear(state_dim, 64)
        
        # First spiking layer
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        # Second hidden layer
        self.fc2 = nn.Linear(64, 128)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        # Third hidden layer
        self.fc3 = nn.Linear(128, 64)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        # Output heads
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)
        
        self.to(self.device)
    
    def reset_state(self) -> None:
        """Reset the membrane potentials of all LIF neurons."""
        self.lif1.reset_mem()
        self.lif2.reset_mem()  
        self.lif3.reset_mem()
    
    def forward(self, x: torch.Tensor):
        """Forward pass through the spiking network.
        
        Args:
            x: State tensor of shape (B, state_dim)
            
        Returns:
            tuple: (policy_logits, value) both of shape (B, n_actions) and (B, 1)
        """
        x = x.to(self.device)
        
        # Normalize the input state for better SNN stability
        x = self.state_norm(x)
        
        # Transform input state to higher dimension
        z = self.input_fc(x)
        
        # First spiking layer
        spk1, _ = self.lif1(z)
        
        # Second layer
        z = self.fc2(spk1)
        spk2, _ = self.lif2(z)
        
        # Third layer  
        z = self.fc3(spk2)
        spk3, _ = self.lif3(z)
        
        # Output (non-spiking for stability)
        policy_logits = self.policy_head(spk3)
        value = self.value_head(spk3)
        
        return policy_logits, value