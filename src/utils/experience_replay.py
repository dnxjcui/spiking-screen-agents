"""
Experience Replay Memory for DQN training.
Based on the reference implementation.
"""
from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple, Any


class ReplayMemory:
    """Circular buffer to store and sample experiences for DQN training."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay memory.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
    
    def append(self, experience: Tuple[Any, ...]) -> None:
        """
        Add an experience to memory.
        
        Args:
            experience: Tuple of (state, action, next_state, reward, done)
        """
        self.memory.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        """Return current memory size."""
        return len(self.memory)
    
    def is_full(self) -> bool:
        """Check if memory is at capacity."""
        return len(self.memory) >= self.capacity