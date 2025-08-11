#!/usr/bin/env python3
"""
Deep Q-Network (DQN) training script for both traditional and spiking neural networks.
Based on the reference implementation but adapted for our environment and model architecture.
"""
from __future__ import annotations

import argparse
import random
import os
from datetime import datetime, timedelta

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt

from src.utils import Config, DQNConfig, set_seed
from src.envs.factory import make_env
from src.models import make_model
from src.encoders.event_encoder import EventEncoder
from src.utils.experience_replay import ReplayMemory

# Use Agg backend for matplotlib (no display required)
matplotlib.use('Agg')

# Create runs directory for saving results
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

DATE_FORMAT = "%m-%d %H:%M:%S"


class DQNAgent:
    """DQN Agent that can work with different environments and model types."""
    
    def __init__(self, env_name: str, model_type: str, config: DQNConfig, device: str = "cpu", save_path: str = None):
        self.env_name = env_name
        self.model_type = model_type
        self.config = config
        self.device = device
        
        # Create environment
        self.env, self.action_map = make_env(env_name, seed=123)
        self.n_actions = self.action_map.n_actions()
        
        # Determine if we need vision processing or state-based
        self.use_vision = model_type in ["vision_dqn"]
        
        if self.use_vision:
            # Vision-based setup
            self.encoder = EventEncoder(84, 84, 15, 2)
            self.state_dim = None
            self.input_channels = 2
        else:
            # State-based setup
            self.encoder = None
            self.state_dim = 4  # CartPole has 4 state dimensions
            self.input_channels = None
        
        # Create models
        self.policy_net = self._create_model()
        self.target_net = self._create_model()
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize training components
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayMemory(config.replay_memory_size)
        
        # Training state
        self.step_count = 0
        self.epsilon = config.epsilon_init
        
        # Logging
        self.rewards_per_episode = []
        self.epsilon_history = []
        self.best_reward = -float('inf')
        
        # File paths
        self.log_file = os.path.join(RUNS_DIR, f'{env_name}_{model_type}.log')
        self.model_file = save_path if save_path else os.path.join(RUNS_DIR, f'{env_name}_{model_type}.pt')
        self.graph_file = os.path.join(RUNS_DIR, f'{env_name}_{model_type}.png')
    
    def _create_model(self):
        """Create the neural network model."""
        if self.use_vision:
            return make_model(self.model_type, in_ch=self.input_channels, n_actions=self.n_actions, device=self.device)
        else:
            return make_model(self.model_type, state_dim=self.state_dim, n_actions=self.n_actions, device=self.device)
    
    def _preprocess_state(self, obs, info=None):
        """Convert environment observation to model input."""
        if self.use_vision:
            # Vision-based processing
            if isinstance(obs, np.ndarray) and len(obs.shape) >= 2:
                frame = obs
            else:
                frame = self.env.render()
                if frame is None:
                    # Fallback to a default frame
                    frame = np.zeros((84, 84, 3), dtype=np.uint8)
            
            encoded = self.encoder.encode(frame)
            return encoded.unsqueeze(0)  # Add batch dimension
        else:
            # State-based processing  
            return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Random action
            return torch.tensor(random.randrange(self.n_actions), dtype=torch.long, device=self.device)
        else:
            # Best action from policy
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax(dim=1)[0]
    
    def optimize_model(self):
        """Perform one optimization step on the policy network."""
        if len(self.memory) < self.config.mini_batch_size:
            return
        
        # Sample batch from memory
        batch = self.memory.sample(self.config.mini_batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device).squeeze()
        next_states = torch.cat(next_states).to(self.device) 
        rewards = torch.cat(rewards).to(self.device).squeeze()
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Next Q values from target network
        with torch.no_grad():
            if self.config.enable_double_dqn:
                # Double DQN: use policy network to select action, target network to evaluate
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states).max(dim=1)[0]
        
        # Target Q values
        target_q_values = rewards + (self.config.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def save_graph(self):
        """Save training progress graphs."""
        if len(self.rewards_per_episode) == 0:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot average rewards (rolling mean over 100 episodes)
        if len(self.rewards_per_episode) > 0:
            rewards = np.array(self.rewards_per_episode)
            if len(rewards) >= 100:
                rolling_mean = np.convolve(rewards, np.ones(100)/100, mode='valid')
                ax1.plot(rolling_mean)
            else:
                ax1.plot(rewards)
            ax1.set_ylabel('Mean Rewards (100 episodes)')
            ax1.set_xlabel('Episodes')
            ax1.set_title(f'{self.env_name} - {self.model_type}')
        
        # Plot epsilon decay
        if len(self.epsilon_history) > 0:
            ax2.plot(self.epsilon_history)
            ax2.set_ylabel('Epsilon')
            ax2.set_xlabel('Training Steps')
            ax2.set_title('Exploration Rate')
        
        plt.tight_layout()
        plt.savefig(self.graph_file)
        plt.close()
    
    def train(self):
        """Main training loop."""
        print(f"Starting DQN training: {self.env_name} with {self.model_type}")
        print(f"Device: {self.device}")
        print(f"Actions: {self.n_actions}")
        print(f"Vision-based: {self.use_vision}")
        
        start_time = datetime.now()
        last_graph_update = start_time
        
        # Log training start
        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training started - {self.env_name} with {self.model_type}"
        print(log_message)
        with open(self.log_file, 'w') as f:
            f.write(log_message + '\n')
        
        episode = 0
        while episode < self.config.max_episodes:
            # Reset environment and model state
            obs, info = self.env.reset()
            if self.encoder:
                self.encoder.reset()
            if hasattr(self.policy_net, 'reset_state'):
                self.policy_net.reset_state()
            if hasattr(self.target_net, 'reset_state'):
                self.target_net.reset_state()
            
            state = self._preprocess_state(obs, info)
            episode_reward = 0.0
            step = 0
            
            while True:
                # Select action
                action = self.select_action(state, training=True)
                
                # Execute action
                env_action = self.action_map.to_env(action.item())
                next_obs, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated
                
                # Process next state
                next_state = self._preprocess_state(next_obs, info) if not done else None
                
                # Store experience
                self.memory.append((
                    state,
                    torch.tensor([action.item()], dtype=torch.long, device=self.device),
                    next_state if next_state is not None else torch.zeros_like(state),
                    torch.tensor([reward], dtype=torch.float32, device=self.device),
                    done
                ))
                
                # Move to next state
                state = next_state
                episode_reward += reward
                step += 1
                self.step_count += 1
                
                # Optimize model
                self.optimize_model()
                
                # Decay epsilon
                self.epsilon = max(self.epsilon * self.config.epsilon_decay, self.config.epsilon_min)
                self.epsilon_history.append(self.epsilon)
                
                # Sync target network
                if self.step_count % self.config.network_sync_rate == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Check termination
                if done or episode_reward >= self.config.stop_on_reward:
                    break
            
            # Store episode reward
            self.rewards_per_episode.append(episode_reward)
            episode += 1
            
            # Save best model
            if episode_reward > self.best_reward:
                old_best = self.best_reward
                self.best_reward = episode_reward
                improvement = ((episode_reward - old_best) / abs(old_best) * 100) if old_best != -float('inf') and old_best != 0 else 0
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:.1f} (+{improvement:.1f}%) at episode {episode}"
                print(log_message)
                with open(self.log_file, 'a') as f:
                    f.write(log_message + '\n')
                
                # Save model
                torch.save({
                    'policy_state_dict': self.policy_net.state_dict(),
                    'target_state_dict': self.target_net.state_dict(),
                    'config': self.config,
                    'episode': episode,
                    'best_reward': self.best_reward,
                    'env_name': self.env_name,
                    'model_type': self.model_type,
                    'use_vision': self.use_vision
                }, self.model_file)
            
            # Logging
            if episode % self.config.log_frequency == 0:
                avg_reward = np.mean(self.rewards_per_episode[-100:]) if len(self.rewards_per_episode) >= 100 else np.mean(self.rewards_per_episode)
                print(f"Episode {episode:4d} | Reward: {episode_reward:6.1f} | Avg: {avg_reward:6.1f} | Epsilon: {self.epsilon:.3f} | Steps: {step}")
            
            # Update graphs periodically
            current_time = datetime.now()
            if current_time - last_graph_update > timedelta(seconds=10):
                self.save_graph()
                last_graph_update = current_time
        
        # Final save
        self.save_graph()
        print(f"Training completed! Best reward: {self.best_reward:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DQN models on CartPole and Pong")
    parser.add_argument("--env", type=str, default="cartpole", choices=["cartpole", "pong"],
                        help="Environment to train on")
    parser.add_argument("--model", type=str, default="dqn", 
                        choices=["dqn", "snn_dqn", "vision_dqn"],
                        help="Model type to use")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes to train (overrides config default)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save trained model")
    args = parser.parse_args()
    
    # Set up device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create DQN config
    config = DQNConfig()
    if args.episodes:
        config.max_episodes = args.episodes
    if args.lr:
        config.learning_rate = args.lr
    
    # Adjust config for CartPole
    if args.env == "cartpole":
        config.stop_on_reward = 500.0  # CartPole max reward
        config.replay_memory_size = 10000  # Smaller buffer for simpler env
        config.network_sync_rate = 500     # More frequent updates
        config.epsilon_decay = 0.995       # Faster decay for CartPole
        config.max_episodes = min(config.max_episodes, 500)  # CartPole solves quickly
    
    # Set random seed
    set_seed(123)
    
    print("="*60)
    print(f"DQN Training Configuration:")
    print(f"Environment: {args.env}")
    print(f"Model: {args.model}")
    print(f"Episodes: {config.max_episodes}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Memory Size: {config.replay_memory_size}")
    print(f"Batch Size: {config.mini_batch_size}")
    print(f"Epsilon: {config.epsilon_init} -> {config.epsilon_min} (decay: {config.epsilon_decay})")
    print(f"Double DQN: {config.enable_double_dqn}")
    print(f"Dueling DQN: {config.enable_dueling_dqn}")
    print("="*60)
    
    # Create and run agent
    agent = DQNAgent(args.env, args.model, config, device, args.save)
    agent.train()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

