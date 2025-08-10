#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.distributions import Categorical

from src.utils import Config, set_seed
from src.envs.factory import make_env
from src.models import make_model
from src.encoders.event_encoder import EventEncoder


def discount_cumsum(rews: List[float], gamma: float) -> List[float]:
    out = []
    g = 0.0
    for r in reversed(rews):
        g = r + gamma * g
        out.append(g)
    return list(reversed(out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SNN on Pong with event-encoded frames")
    parser.add_argument("--episodes", type=int, default=Config.max_episodes)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save", type=str, default=Config.save_path)
    parser.add_argument("--threshold", type=int, default=Config.diff_threshold)
    parser.add_argument("--stride", type=int, default=Config.stride)
    parser.add_argument("--lr", type=float, default=Config.lr)
    args = parser.parse_args()

    cfg = Config(diff_threshold=args.threshold, stride=args.stride, render=args.render, save_path=args.save, lr=args.lr)
    print(cfg)

    set_seed(cfg.seed)
    env, action_map = make_env("pong", seed=cfg.seed)
    enc = EventEncoder(cfg.height, cfg.width, cfg.diff_threshold, cfg.stride)

    n_actions = action_map.n_actions()
    policy = make_model("snn", in_ch=2, n_actions=n_actions, beta=cfg.beta, device=cfg.device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    running_return = None
    for ep in range(1, cfg.max_episodes + 1):
        obs, info = env.reset(seed=cfg.seed + ep)
        enc.reset()
        policy.reset_state()

        logps, rewards = [], []
        ep_reward = 0.0
        done = False

        while not done:
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

            logits = policy(x)
            dist = Categorical(logits=logits)
            action_idx = dist.sample()
            logps.append(dist.log_prob(action_idx))

            env_action = action_map.to_env(action_idx.item())
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            rewards.append(float(reward))
            ep_reward += float(reward)

        returns = discount_cumsum(rewards, cfg.gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=cfg.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for logp, Gt in zip(logps, returns):
            loss = loss - (logp.to(cfg.device) * Gt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimizer.step()

        running_return = ep_reward if running_return is None else 0.9 * running_return + 0.1 * ep_reward
        print(f"Episode {ep:03d} | return={ep_reward:6.2f} | running_return={running_return:6.2f}")

        if ep % max(1, cfg.max_episodes // 10) == 0 or ep == cfg.max_episodes:
            torch.save({"model_state": policy.state_dict(), "cfg": cfg.__dict__}, cfg.save_path)
            print(f"Saved checkpoint to {cfg.save_path}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

