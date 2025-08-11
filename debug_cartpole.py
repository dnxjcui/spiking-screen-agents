#!/usr/bin/env python3
"""
Debug script to analyze CartPole model behavior and diagnose training issues.
"""
from __future__ import annotations

import torch
import numpy as np
from torch.distributions import Categorical

from src.utils import Config
from src.envs.factory import make_env
from src.models import make_model


def analyze_model_behavior(model_path: str, n_episodes: int = 3):
    """Analyze what the model is actually doing step by step."""
    
    print(f"=== Debugging CartPole Model: {model_path} ===\n")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model_cfg = Config(**checkpoint['cfg'])
    
    # Create environment and model
    env, action_map = make_env("cartpole", seed=42, render_mode="rgb_array")
    model = make_model("snn_state_ac", state_dim=4, n_actions=2, beta=model_cfg.beta, device=model_cfg.device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"Model loaded. Beta: {model_cfg.beta}")
    print(f"Actions: 0=LEFT, 1=RIGHT\n")
    
    action_counts = {0: 0, 1: 0}
    all_rewards = []
    all_lengths = []
    
    for episode in range(1, n_episodes + 1):
        print(f"--- Episode {episode} ---")
        obs, info = env.reset(seed=42 + episode)
        model.reset_state()
        
        episode_reward = 0
        steps = 0
        done = False
        
        action_history = []
        state_history = []
        prob_history = []
        
        while not done and steps < 500:
            # Process state
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            state_history.append(obs.copy())
            
            with torch.no_grad():
                logits, value = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                dist = Categorical(logits=logits)
                action_idx = dist.sample()
                
                prob_history.append(probs.numpy())
                action_history.append(action_idx.item())
                action_counts[action_idx.item()] += 1
            
            # Take action
            env_action = action_map.to_env(action_idx.item())
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Print detailed info for first few steps
            if steps <= 5 or done:
                state_str = f"[{obs[0]:6.3f}, {obs[1]:6.3f}, {obs[2]:6.3f}, {obs[3]:6.3f}]"
                prob_str = f"[{probs[0]:.3f}, {probs[1]:.3f}]"
                action_name = "LEFT" if action_idx.item() == 0 else "RIGHT"
                print(f"  Step {steps:2d}: State={state_str} Action={action_name} Probs={prob_str} Value={value.item():.3f}")
        
        all_rewards.append(episode_reward)
        all_lengths.append(steps)
        
        print(f"  Episode {episode} ended: {steps} steps, reward={episode_reward}")
        print(f"  Final state: [{obs[0]:6.3f}, {obs[1]:6.3f}, {obs[2]:6.3f}, {obs[3]:6.3f}]")
        
        # Analyze action patterns for this episode
        unique_actions, action_counts_ep = np.unique(action_history, return_counts=True)
        print(f"  Action distribution: {dict(zip(unique_actions, action_counts_ep))}")
        
        # Check if it's always choosing the same action
        if len(unique_actions) == 1:
            print(f"  ⚠️  WARNING: Only choosing action {unique_actions[0]} ({'LEFT' if unique_actions[0] == 0 else 'RIGHT'})")
        
        # Check probability distribution
        avg_probs = np.mean(prob_history, axis=0)
        print(f"  Average action probabilities: LEFT={avg_probs[0]:.3f}, RIGHT={avg_probs[1]:.3f}")
        
        if max(avg_probs) > 0.95:
            print(f"  ⚠️  WARNING: Very high confidence in one action ({max(avg_probs):.3f})")
        elif max(avg_probs) < 0.6:
            print(f"  ⚠️  WARNING: Very low confidence (max prob: {max(avg_probs):.3f})")
        
        print()
    
    # Overall analysis
    print("=== OVERALL ANALYSIS ===")
    print(f"Average episode length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print(f"Average episode reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    
    total_actions = sum(action_counts.values())
    print(f"Action distribution across all episodes:")
    print(f"  LEFT:  {action_counts[0]:3d} ({action_counts[0]/total_actions*100:.1f}%)")
    print(f"  RIGHT: {action_counts[1]:3d} ({action_counts[1]/total_actions*100:.1f}%)")
    
    if action_counts[0] == 0 or action_counts[1] == 0:
        print("  ❌ PROBLEM: Model only uses one action!")
    elif abs(action_counts[0] - action_counts[1]) / total_actions > 0.8:
        print("  ⚠️  Model heavily biased toward one action")
    else:
        print("  ✅ Model uses both actions")
    
    # Performance assessment
    if np.mean(all_lengths) < 20:
        print("  ❌ PROBLEM: Very poor performance (< 20 steps average)")
        print("     Possible issues: Model not trained, wrong architecture, or bad hyperparameters")
    elif np.mean(all_lengths) < 50:
        print("  ⚠️  Poor performance (< 50 steps average)")
        print("     Suggestion: Train longer or adjust hyperparameters")
    elif np.mean(all_lengths) < 200:
        print("  🤔 Moderate performance")
        print("     Suggestion: Continue training")
    else:
        print("  ✅ Good performance!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debug_cartpole.py <model_path>")
        sys.exit(1)
    
    analyze_model_behavior(sys.argv[1])