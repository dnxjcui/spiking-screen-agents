#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.utils import Config, set_seed
from src.envs.factory import make_env
from src.models import make_model
from src.encoders.event_encoder import EventEncoder


@dataclass
class PPOConfig:
    gamma: float = 0.99
    clip_eps: float = 0.2
    lam: float = 0.95
    lr: float = 1e-3
    train_iters: int = 4
    batch_size: int = 2048


def compute_gae(rewards, values, dones, gamma, lam):
    adv = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (0.0 if dones[t] else next_value) - values[t]
        gae = delta + gamma * lam * (0.0 if dones[t] else gae)
        adv.insert(0, gae)
        next_value = values[t]
    returns = [a + v for a, v in zip(adv, values)]
    return np.array(adv, dtype=np.float32), np.array(returns, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SNN Actor-Critic with PPO on Gymnasium Pong")
    parser.add_argument("--episodes", type=int, default=Config.max_episodes)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save", type=str, default=Config.save_path)
    parser.add_argument("--threshold", type=int, default=Config.diff_threshold)
    parser.add_argument("--stride", type=int, default=Config.stride)
    parser.add_argument("--lr", type=float, default=Config.lr)
    args = parser.parse_args()

    cfg = Config(diff_threshold=args.threshold, stride=args.stride, render=args.render, save_path=args.save, lr=args.lr)
    ppo = PPOConfig(lr=cfg.lr)
    print(cfg)

    set_seed(cfg.seed)
    env, action_map = make_env("pong", seed=cfg.seed)
    enc = EventEncoder(cfg.height, cfg.width, cfg.diff_threshold, cfg.stride)

    n_actions = action_map.n_actions()
    ac = make_model("snn_ac", in_ch=2, n_actions=n_actions, beta=cfg.beta, device=cfg.device)
    optimizer = optim.Adam(ac.parameters(), lr=ppo.lr)

    for ep in range(1, cfg.max_episodes + 1):
        obs, info = env.reset(seed=cfg.seed + ep)
        enc.reset()
        ac.reset_state()

        frames = 0
        done = False

        traj_obs = []
        traj_actions = []
        traj_logp = []
        traj_rewards = []
        traj_values = []
        traj_dones = []

        while not done and frames < 10_000:
            frame = env.render()
            if frame is None or not isinstance(frame, np.ndarray):
                frame = obs
            x = enc.encode(frame).unsqueeze(0)
            if cfg.render:
                on = (x[0, 0].cpu().numpy() * 255).astype(np.uint8)
                off = (x[0, 1].cpu().numpy() * 255).astype(np.uint8)
                cv2.imshow("Encoder ON (left) / OFF (right)", np.hstack([on, off]))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cfg.render = False
                    cv2.destroyAllWindows()

            logits, value = ac(x)
            dist = Categorical(logits=logits)
            action_idx = dist.sample()
            logp = dist.log_prob(action_idx)

            env_action = action_map.to_env(action_idx.item())
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            traj_obs.append(x.detach())
            traj_actions.append(action_idx.detach())
            traj_logp.append(logp.detach())
            traj_rewards.append(float(reward))
            traj_values.append(float(value.squeeze().item()))
            traj_dones.append(done)
            frames += 1

        adv, ret = compute_gae(traj_rewards, traj_values, traj_dones, ppo.gamma, ppo.lam)
        obs_batch = torch.cat(traj_obs, dim=0).to(cfg.device)
        act_batch = torch.stack(traj_actions).to(cfg.device)
        old_logp_batch = torch.stack(traj_logp).to(cfg.device)
        adv_batch = torch.tensor(adv, dtype=torch.float32, device=cfg.device)
        ret_batch = torch.tensor(ret, dtype=torch.float32, device=cfg.device)
        adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

        for _ in range(ppo.train_iters):
            logits, value = ac(obs_batch)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act_batch)
            ratio = torch.exp(logp - old_logp_batch)
            pg1 = ratio * adv_batch
            pg2 = torch.clamp(ratio, 1.0 - ppo.clip_eps, 1.0 + ppo.clip_eps) * adv_batch
            policy_loss = -torch.min(pg1, pg2).mean()
            value_loss = 0.5 * (ret_batch - value.squeeze()).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + value_loss * 0.5 - 0.01 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=1.0)
            optimizer.step()

        torch.save({"model_state": ac.state_dict(), "cfg": cfg.__dict__}, cfg.save_path)
        print(f"Episode {ep:03d} | steps={frames} | saved {cfg.save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

