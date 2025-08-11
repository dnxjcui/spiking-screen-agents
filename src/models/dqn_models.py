"""
Deep Q-Network models for both traditional and spiking implementations.
Based on the reference implementation but adapted for our architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


class TraditionalDQN(nn.Module):
    """Traditional DQN for state-based environments like CartPole."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, enable_dueling: bool = True, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.enable_dueling = enable_dueling
        
        # First layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        if self.enable_dueling:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)
            
            # Advantage stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            # Simple Q-value output
            self.output = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        self.to(self.device)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def reset_state(self) -> None:
        """No-op for traditional models (no hidden state)."""
        pass
    
    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        
        if self.enable_dueling:
            # Value stream
            v = F.relu(self.fc_value(x))
            V = self.value(v)
            
            # Advantage stream  
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)
            
            # Combine using dueling formula: Q = V + A - mean(A)
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)
        
        return Q


class SpikingDQN(nn.Module):
    """Spiking Neural Network DQN for state-based environments."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, beta: float = 0.95, enable_dueling: bool = True, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.enable_dueling = enable_dueling
        
        # Surrogate gradient function
        spike_grad = surrogate.fast_sigmoid()
        
        # Input processing
        self.input_fc = nn.Linear(state_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        if self.enable_dueling:
            # Value stream
            self.value_fc = nn.Linear(hidden_dim, 64)
            self.value_lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
            self.value_out = nn.Linear(64, 1)
            
            # Advantage stream
            self.adv_fc = nn.Linear(hidden_dim, 64)
            self.adv_lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
            self.adv_out = nn.Linear(64, action_dim)
        else:
            # Simple output
            self.output_fc = nn.Linear(hidden_dim, 64)
            self.output_lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
            self.output = nn.Linear(64, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        self.to(self.device)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def reset_state(self) -> None:
        """Reset membrane potentials of all LIF neurons."""
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        if self.enable_dueling:
            self.value_lif.reset_mem()
            self.adv_lif.reset_mem()
        else:
            self.output_lif.reset_mem()
    
    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        
        # Input processing
        z = self.input_fc(x)
        spk1, _ = self.lif1(z)
        
        # Hidden layer
        z = self.fc2(spk1)
        spk2, _ = self.lif2(z)
        
        if self.enable_dueling:
            # Value stream
            v_z = self.value_fc(spk2)
            v_spk, _ = self.value_lif(v_z)
            V = self.value_out(v_spk)
            
            # Advantage stream
            a_z = self.adv_fc(spk2)
            a_spk, _ = self.adv_lif(a_z)
            A = self.adv_out(a_spk)
            
            # Combine using dueling formula
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            # Simple output
            o_z = self.output_fc(spk2)
            o_spk, _ = self.output_lif(o_z)
            Q = self.output(o_spk)
        
        return Q


class VisionDQN(nn.Module):
    """DQN for vision-based environments like Pong (traditional CNN)."""
    
    def __init__(self, in_channels: int, action_dim: int, enable_dueling: bool = True, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.enable_dueling = enable_dueling
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size (assuming 84x84 input)
        self.feature_size = 64 * 11 * 11
        
        if self.enable_dueling:
            # Value stream
            self.value_fc = nn.Linear(self.feature_size, 256)
            self.value_out = nn.Linear(256, 1)
            
            # Advantage stream
            self.adv_fc = nn.Linear(self.feature_size, 256)
            self.adv_out = nn.Linear(256, action_dim)
        else:
            # Simple output
            self.fc = nn.Linear(self.feature_size, 512)
            self.output = nn.Linear(512, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        self.to(self.device)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def reset_state(self) -> None:
        """No-op for traditional models (no hidden state)."""
        pass
    
    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        
        # Convolutional features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        if self.enable_dueling:
            # Value stream
            V = F.relu(self.value_fc(x))
            V = self.value_out(V)
            
            # Advantage stream
            A = F.relu(self.adv_fc(x))
            A = self.adv_out(A)
            
            # Combine using dueling formula
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            # Simple output
            x = F.relu(self.fc(x))
            Q = self.output(x)
        
        return Q