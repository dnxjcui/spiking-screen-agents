#!/usr/bin/env python3
"""
Evaluation script to test a trained SNN model on Pong with visual rendering.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.distributions import Categorical

from src.utils import Config, set_seed
from src.envs.factory import make_env
from src.models import make_model
from src.encoders.event_encoder import EventEncoder


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SNN model on Pong")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum steps per episode")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (argmax)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load model checkpoint
    print(f"Loading model from {args.model}")
    # Add safe globals for config classes to handle PyTorch 2.6+ weights_only default
    from src.utils.configs import Config, DQNConfig, PPOConfig
    torch.serialization.add_safe_globals([Config, DQNConfig, PPOConfig])
    checkpoint = torch.load(args.model, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'cfg' in checkpoint:
        model_cfg = Config(**checkpoint['cfg']) if isinstance(checkpoint['cfg'], dict) else checkpoint['cfg']
    elif 'config' in checkpoint:
        # DQN/PPO checkpoint format - convert to Config or use defaults
        config_obj = checkpoint['config']
        if hasattr(config_obj, '__dict__'):
            # It's a config object, use default Config for evaluation
            model_cfg = Config()
        elif isinstance(config_obj, dict):
            # Filter config_obj to only include Config parameters
            config_params = {k: v for k, v in config_obj.items() if hasattr(Config(), k)}
            model_cfg = Config(**config_params)
        else:
            model_cfg = Config()
    else:
        # Default config if not saved
        model_cfg = Config()
    
    # Extract environment and model type from checkpoint
    env_name = checkpoint.get('env_name', 'pong')  # Default to pong for backward compatibility
    use_vision = checkpoint.get('use_vision', True)  # Default to vision for backward compatibility
    model_type = checkpoint.get('model_type', 'snn_ac' if use_vision else 'snn_state_ac')  # Backward compatibility
    
    # Handle DQN models - infer from state dict keys if not explicitly saved
    if 'policy_state_dict' in checkpoint:
        state_dict_keys = list(checkpoint['policy_state_dict'].keys())
        
        # Check if this is a DQN model based on layer names
        is_dueling_dqn = any('fc_advantages' in key or 'advantages' in key for key in state_dict_keys)
        has_conv_layers = any('conv' in key for key in state_dict_keys)
        has_fc_only = any('fc' in key for key in state_dict_keys) and not has_conv_layers
        
        if is_dueling_dqn or has_fc_only:
            # This is definitely a DQN model
            if has_conv_layers:
                model_type = 'vision_dqn'
                use_vision = True
                env_name = 'pong'
            else:
                model_type = 'dqn' 
                use_vision = False
                env_name = 'cartpole'
    
    set_seed(args.seed)
    
    # Create environment based on saved model type
    print(f"Creating {env_name} environment...")
    render_mode = "human" if env_name == "cartpole" else "human"  # CartPole renders better with human mode
    env, action_map = make_env(env_name, seed=args.seed, render_mode=render_mode)
    
    # Create encoder and model based on model type
    if use_vision:
        print(f"Using vision-based model ({model_type})")
        enc = EventEncoder(model_cfg.height, model_cfg.width, model_cfg.diff_threshold, model_cfg.stride)
        n_actions = action_map.n_actions()
        model = make_model(model_type, in_ch=2, n_actions=n_actions, beta=model_cfg.beta, device=model_cfg.device)
    else:
        print(f"Using state-based model ({model_type})")
        enc = None  # No encoder needed for state-based models
        n_actions = action_map.n_actions()
        model = make_model(model_type, state_dim=4, n_actions=n_actions, beta=model_cfg.beta, device=model_cfg.device)
    
    # Load model weights - handle different checkpoint formats
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'policy_state_dict' in checkpoint:
        # DQN checkpoint format
        model.load_state_dict(checkpoint['policy_state_dict'])
    else:
        # Legacy format - assume checkpoint is the state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Determine model type for inference
    is_dqn_model = model_type in ["dqn", "snn_dqn", "vision_dqn"]
    
    action_names = ["LEFT", "RIGHT"] if env_name == "cartpole" else ["NOOP", "UP", "DOWN"]
    print(f"Model loaded. Actions: {n_actions} ({', '.join(action_names)})")
    print(f"Model type: {model_type} {'(DQN)' if is_dqn_model else '(Actor-Critic)'}")
    print(f"Evaluating for {args.episodes} episodes on {env_name}...")
    print(f"Close the {env_name} window to stop evaluation early.")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        if hasattr(model, 'reset_state'):
            model.reset_state()
        if enc:
            enc.reset()
        
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"\nEpisode {episode}/{args.episodes}")
        
        while not done and steps < args.max_steps:
            # Render the environment (this will display the Pong game)
            try:
                env.render()
            except Exception as e:
                print(f"Rendering failed: {e}")
                break
            
            # Process observation based on model type
            with torch.no_grad():
                if use_vision:
                    # Vision-based processing (Pong)
                    frame = env.render()
                    if frame is None or not isinstance(frame, np.ndarray):
                        frame = obs
                    x = enc.encode(frame).unsqueeze(0)
                else:
                    # State-based processing (CartPole)
                    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                if is_dqn_model:
                    # DQN model - output is Q-values
                    q_values = model(x)
                    action_idx = torch.argmax(q_values, dim=1)  # Always greedy for DQN evaluation
                else:
                    # Actor-Critic model - output is logits and value
                    logits, value = model(x)
                    
                    if args.deterministic:
                        # Use deterministic policy (argmax)
                        action_idx = torch.argmax(logits, dim=1)
                    else:
                        # Sample from policy distribution
                        dist = Categorical(logits=logits)
                        action_idx = dist.sample()
                
                # Convert to environment action
                env_action = action_map.to_env(action_idx.item())
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Print some info periodically
            if steps % 200 == 0:
                if is_dqn_model:
                    q_vals = q_values[0].detach().cpu().numpy()
                    print(f"  Step {steps}: Action={action_names[action_idx.item()]} "
                          f"Reward={reward} Q-values={q_vals}")
                else:
                    action_probs = torch.softmax(logits, dim=1)[0]
                    print(f"  Step {steps}: Action={action_names[action_idx.item()]} "
                          f"Reward={reward} Value={value.item():.3f} "
                          f"Probs={action_probs.detach().cpu().numpy()}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode} finished: {steps} steps, reward={episode_reward}")
        
        # Brief pause between episodes
        import time
        time.sleep(1)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max Reward: {np.max(episode_rewards)}")
    print(f"Min Reward: {np.min(episode_rewards)}")
    
    # Close environment
    env.close()
    
    return 0


if __name__ == "__main__":
    exit(main())