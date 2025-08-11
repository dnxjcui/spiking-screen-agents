#!/usr/bin/env python3
"""
PPO training script based on reference implementation but adapted for our SNN and traditional models.
Supports both vision-based (Pong) and state-based (CartPole) environments.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.utils import Config, PPOConfig, set_seed
from src.envs.factory import make_env
from src.models import make_model
from src.encoders.event_encoder import EventEncoder

# Use Agg backend for matplotlib (no display required)
import matplotlib
matplotlib.use('Agg')

# Create runs directory for saving results
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

DATE_FORMAT = "%m-%d %H:%M:%S"


class PPOAgent:
    """PPO Agent following reference implementation pattern."""
    
    def __init__(self, env_name: str, model_type: str, config: PPOConfig, device: str = "cpu", save_path: str = None):
        super().__init__()
        self.env_name = env_name
        self.model_type = model_type
        self.config = config
        self.device = device
        
        # Create environment
        self.env, self.action_map = make_env(env_name, seed=123)
        self.action_size = self.action_map.n_actions()
        
        # Determine if we need vision processing or state-based
        self.use_vision = model_type in ["snn_ac", "traditional_ac"]
        
        if self.use_vision:
            # Vision-based setup (Pong)
            self.encoder = EventEncoder(84, 84, 15, 2)
            self.input_size = None
            self.policy_network = make_model(model_type, in_ch=2, n_actions=self.action_size, device=device)
        else:
            # State-based setup (CartPole)
            self.encoder = None
            self.input_size = 4  # CartPole has 4 state dimensions
            self.policy_network = make_model(model_type, state_dim=self.input_size, n_actions=self.action_size, device=device)
        
        # Training components
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config.k_epoch, gamma=0.999)
        self.criterion = nn.MSELoss()
        
        # Memory for storing trajectory
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': [], 'count': 0,
            'advantage': [], 'td_target': torch.FloatTensor([])
        }
        
        # Training state
        self.loss = 0
        self.reward_history = []
        self.avg_reward = []
        self.solved = False
        
        # File paths
        self.model_file = save_path if save_path else os.path.join(RUNS_DIR, f'{env_name}_{model_type}_ppo.pt')
        self.graph_file = os.path.join(RUNS_DIR, f'{env_name}_{model_type}_ppo.png')
    
    def _preprocess_state(self, obs, info=None):
        """Convert environment observation to model input."""
        if self.use_vision:
            # Vision-based processing
            if isinstance(obs, np.ndarray) and len(obs.shape) >= 2:
                frame = obs
            else:
                frame = self.env.render()
                if frame is None:
                    frame = np.zeros((84, 84, 3), dtype=np.uint8)
            
            encoded = self.encoder.encode(frame)
            return encoded.numpy()  # Convert to numpy for consistency with reference
        else:
            # State-based processing  
            return obs
    
    def new_random_game(self):
        """Initialize a new episode with random first action (following reference pattern)."""
        obs, info = self.env.reset()
        if self.encoder:
            self.encoder.reset()
        if hasattr(self.policy_network, 'reset_state'):
            self.policy_network.reset_state()
        
        # Take one random action as in reference  
        action = self.env.env.action_space.sample()
        env_action = self.action_map.to_env(action)
        next_obs, reward, terminated, truncated, info = self.env.step(env_action)
        terminal = terminated or truncated
        
        state = self._preprocess_state(next_obs, info)
        return state, reward, action, terminal
    
    def train(self):
        """Main training loop following reference implementation pattern."""
        print(f"Starting PPO training: {self.env_name} with {self.model_type}")
        print(f"Device: {self.device}")
        print(f"Actions: {self.action_size}")
        print(f"Vision-based: {self.use_vision}")
        
        episode = 0
        step = 0
        
        # Main training loop
        while not self.solved:
            start_step = step
            episode += 1
            episode_length = 0
            
            # Get initial state
            state, reward, action, terminal = self.new_random_game()
            current_state = state
            total_episode_reward = 1
            
            # Episode loop
            while not self.solved:
                step += 1
                episode_length += 1
                
                # Choose action using policy network
                if self.use_vision:
                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
                else:
                    state_tensor = torch.FloatTensor(current_state).to(self.device)
                
                # Get action probabilities from policy network
                if hasattr(self.policy_network, 'pi'):
                    # Reference-style network with separate pi and v methods
                    prob_a = self.policy_network.pi(state_tensor)
                else:
                    # Our actor-critic networks
                    logits, _ = self.policy_network(state_tensor)
                    prob_a = torch.softmax(logits, dim=-1)
                
                # Sample action
                action = torch.distributions.Categorical(prob_a).sample().item()
                
                # Execute action in environment
                env_action = self.action_map.to_env(action)
                next_obs, reward, terminated, truncated, info = self.env.step(env_action)
                terminal = terminated or truncated
                new_state = self._preprocess_state(next_obs, info) if not terminal else current_state
                
                # Reward shaping as in reference
                reward = -1 if terminal else reward
                
                # Store experience
                self.add_memory(current_state, action, reward/10.0, new_state, terminal, prob_a[action].item())
                
                current_state = new_state
                total_episode_reward += reward
                
                if terminal:
                    episode_length = step - start_step
                    self.reward_history.append(total_episode_reward)
                    self.avg_reward.append(sum(self.reward_history[-10:])/10.0)
                    
                    self.finish_path(episode_length)
                    
                    # Check if solved (CartPole criterion)
                    if len(self.reward_history) > 100 and sum(self.reward_history[-100:-1]) / 100 >= 195:
                        self.solved = True
                    
                    print('episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                          'loss: %.4f, lr: %.4f' % (episode, step, episode_length, total_episode_reward, self.loss,
                                                    self.scheduler.get_lr()[0]))
                    
                    self.env.reset()
                    break
            
            # Update network periodically
            if episode % self.config.update_freq == 0:
                for _ in range(self.config.k_epoch):
                    self.update_network()
            
            # Plot results periodically
            if episode % self.config.plot_every == 0:
                self.plot_graph()
            
            # Early stopping for demonstration
            if episode >= self.config.max_episodes:
                break
        
        print(f"Training completed! Episodes: {episode}")
        
        # Save final model
        save_dict = {
            "model_state": self.policy_network.state_dict(), 
            "config": self.config.__dict__,
            "env_name": self.env_name,
            "use_vision": self.use_vision,
            "model_type": self.model_type,
            "episode": episode,
            "reward_history": self.reward_history
        }
        torch.save(save_dict, self.model_file)
        self.plot_graph()
    
    def update_network(self):
        """Update policy network using PPO loss (following reference implementation)."""
        if len(self.memory['state']) == 0:
            return
            
        # Convert memory to tensors
        if self.use_vision:
            states = torch.FloatTensor(np.array(self.memory['state'])).to(self.device)
        else:
            states = torch.FloatTensor(np.array(self.memory['state'])).to(self.device)
        actions = torch.tensor(self.memory['action']).to(self.device)
        old_probs_a = torch.FloatTensor(self.memory['action_prob']).to(self.device)
        advantages = torch.FloatTensor(self.memory['advantage']).to(self.device)
        td_targets = self.memory['td_target'].to(self.device)
        
        # Reset network state for fresh computation
        if hasattr(self.policy_network, 'reset_state'):
            self.policy_network.reset_state()
        
        # Get current policy probabilities
        if hasattr(self.policy_network, 'pi'):
            # Reference-style network
            pi = self.policy_network.pi(states)
            pred_v = self.policy_network.v(states)
        else:
            # Our actor-critic networks
            logits, values = self.policy_network(states)
            pi = torch.softmax(logits, dim=-1)
            pred_v = values
        
        # Get action probabilities
        new_probs_a = torch.gather(pi, 1, actions.unsqueeze(1)).squeeze()
        
        # Compute ratio
        ratio = torch.exp(torch.log(new_probs_a + 1e-8) - torch.log(old_probs_a + 1e-8))
        
        # PPO surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantages
        
        # Value loss
        v_loss = 0.5 * (pred_v.squeeze() - td_targets).pow(2)
        
        # Entropy for exploration
        entropy = torch.distributions.Categorical(pi).entropy()
        
        # Total loss
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = v_loss.mean()
        entropy_loss = -entropy.mean()
        
        self.loss = policy_loss + self.config.v_coef * value_loss + self.config.entropy_coef * entropy_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    
    def add_memory(self, s, a, r, next_s, t, prob):
        """Add experience to memory (following reference implementation)."""
        if self.memory['count'] < self.config.memory_size:
            self.memory['count'] += 1
        else:
            # Remove oldest entries
            for key in ['state', 'action', 'reward', 'next_state', 'terminal', 'action_prob', 'advantage']:
                if len(self.memory[key]) > 0:
                    self.memory[key] = self.memory[key][1:]
            if len(self.memory['td_target']) > 0:
                self.memory['td_target'] = self.memory['td_target'][1:]

        self.memory['state'].append(s)
        self.memory['action'].append(a)
        self.memory['reward'].append(r)
        self.memory['next_state'].append(next_s)
        self.memory['terminal'].append(1 - t)  # Convert done to terminal (1 - done)
        self.memory['action_prob'].append(prob)
    
    def finish_path(self, length):
        """Compute advantages and TD targets (following reference implementation)."""
        state = self.memory['state'][-length:]
        reward = self.memory['reward'][-length:]
        next_state = self.memory['next_state'][-length:]
        terminal = self.memory['terminal'][-length:]
        
        # Convert to tensors
        if self.use_vision:
            state_tensor = torch.FloatTensor(np.array(state)).to(self.device)
            next_state_tensor = torch.FloatTensor(np.array(next_state)).to(self.device)
        else:
            state_tensor = torch.FloatTensor(np.array(state)).to(self.device)
            next_state_tensor = torch.FloatTensor(np.array(next_state)).to(self.device)
        
        reward_tensor = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        terminal_tensor = torch.FloatTensor(terminal).unsqueeze(1).to(self.device)
        
        # Get value estimates
        if hasattr(self.policy_network, 'v'):
            # Reference-style network
            values = self.policy_network.v(state_tensor)
            next_values = self.policy_network.v(next_state_tensor)
        else:
            # Our actor-critic networks
            _, values = self.policy_network(state_tensor)
            _, next_values = self.policy_network(next_state_tensor)
        
        # Compute TD targets
        td_target = reward_tensor + self.config.gamma * next_values * terminal_tensor
        
        # Compute TD errors (delta)
        delta = (td_target - values).detach().cpu().numpy()
        
        # Compute GAE advantages
        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.config.gamma * self.config.lmbda * adv + d[0]
            advantages.append(adv)
        advantages.reverse()
        
        # Store TD targets and advantages
        if self.memory['td_target'].shape == torch.Size([0]) or len(self.memory['td_target']) == 0:
            self.memory['td_target'] = td_target.squeeze().data
        else:
            self.memory['td_target'] = torch.cat((self.memory['td_target'], td_target.squeeze().data), dim=0)
        
        self.memory['advantage'].extend(advantages)
    
    def plot_graph(self):
        """Plot training progress (following reference implementation)."""
        if len(self.reward_history) == 0:
            return
            
        df = pd.DataFrame({
            'x': range(len(self.reward_history)), 
            'Reward': self.reward_history, 
            'Average': self.avg_reward[:len(self.reward_history)]
        })
        
        plt.figure(figsize=(10, 6))
        plt.style.use('default')  # Changed from seaborn-darkgrid which is deprecated
        
        plt.plot(df['x'], df['Reward'], marker='', linewidth=0.8, alpha=0.9, label='Reward')
        plt.plot(df['x'], df['Average'], marker='', color='red', linewidth=1, alpha=0.9, label='Average (10 episodes)')
        
        plt.legend(loc='upper left')
        plt.title(f"PPO Training - {self.env_name} ({self.model_type})", fontsize=14)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.graph_file, dpi=150, bbox_inches='tight')
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PPO models on CartPole and Pong")
    parser.add_argument("--env", type=str, default="cartpole", choices=["cartpole", "pong"],
                        help="Environment to train on")
    parser.add_argument("--model", type=str, default="traditional_state_ac", 
                        choices=["snn_ac", "snn_state_ac", "traditional_ac", "traditional_state_ac"],
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
    
    # Create PPO config
    config = PPOConfig()
    if args.episodes:
        config.max_episodes = args.episodes
    if args.lr:
        config.learning_rate = args.lr
    
    # Adjust config for environment
    if args.env == "cartpole":
        config.memory_size = 1024  # Smaller memory for CartPole
        config.k_epoch = 8         # More epochs per update
        config.max_episodes = min(config.max_episodes, 2000)  # CartPole can take longer to solve
    
    # Auto-select model if needed
    if args.model == "auto":
        if args.env == "pong":
            args.model = "snn_ac"
        else:  # cartpole
            args.model = "snn_state_ac"
    
    # Set random seed
    set_seed(123)
    
    print("="*60)
    print(f"PPO Training Configuration:")
    print(f"Environment: {args.env}")
    print(f"Model: {args.model}")
    print(f"Episodes: {config.max_episodes}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Memory Size: {config.memory_size}")
    print(f"Update Frequency: {config.update_freq} episodes")
    print(f"Epochs per Update: {config.k_epoch}")
    print(f"Clip Epsilon: {config.eps_clip}")
    print("="*60)
    
    # Create and run agent
    agent = PPOAgent(args.env, args.model, config, device, args.save)
    agent.train()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())