#!/usr/bin/env python3
import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym

import snntorch as snn
from snntorch import surrogate

from src.utils import Config, set_seed
from src.envs.factory import make_env
from src.models import make_model


# ---------------------------
# Event Encoder (frame -> ON/OFF event maps)
# ---------------------------

class EventEncoder:
    """Converts consecutive frames to two binary ON/OFF maps using a simple diff threshold.
    Produces a 2 x H x W float32 tensor in [0,1]."""
    def __init__(self, height=84, width=84, diff_threshold=15, stride=2):
        self.h = height
        self.w = width
        self.th = diff_threshold
        self.stride = stride
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # frame is RGB (H,W,3) uint8 from Gymnasium
        if self.stride > 1:
            frame = frame[::self.stride, ::self.stride]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return gray

    def encode(self, frame: np.ndarray) -> torch.Tensor:
        gray = self._preprocess(frame)
        if self.prev_gray is None:
            self.prev_gray = gray
            on = np.zeros_like(gray, dtype=np.uint8)
            off = np.zeros_like(gray, dtype=np.uint8)
        else:
            diff = gray.astype(np.int16) - self.prev_gray.astype(np.int16)
            on = (diff > self.th).astype(np.uint8)
            off = (diff < -self.th).astype(np.uint8)
            self.prev_gray = gray
        # 2 x H x W float tensor
        stacked = np.stack([on, off], axis=0).astype(np.float32)
        return torch.from_numpy(stacked)


# ---------------------------
# Action Mapping Helper
# ---------------------------

class ActionMapper:  # backward-compat import alias; real impl in src.envs.action_mapper
    pass


# ---------------------------
# SNN Policy Network
# ---------------------------

class SNNPolicy(nn.Module):
    pass  # kept for backward-compat in imports; actual impl in src.models.snn_policy


# ---------------------------
# Training Loop (REINFORCE)
# ---------------------------

def discount_cumsum(rews: List[float], gamma: float) -> List[float]:
    out = []
    g = 0.0
    for r in reversed(rews):
        g = r + gamma * g
        out.append(g)
    return list(reversed(out))


def train(cfg: Config):
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
        steps = 0

        while not done:
            frame = env.render()  # rgb array
            if frame is None or not isinstance(frame, np.ndarray):
                # If render_mode didn't provide a frame, use observation directly
                frame = obs  # some envs return frame as obs

            x = enc.encode(frame)  # 2 x H x W
            x = x.unsqueeze(0)     # B=1
            if cfg.render:
                # Optional: visualize encoder ON/OFF maps side-by-side
                on = (x[0, 0].cpu().numpy() * 255).astype(np.uint8)
                off = (x[0, 1].cpu().numpy() * 255).astype(np.uint8)
                stacked = np.hstack([on, off])
                cv2.imshow("Encoder ON (left) / OFF (right)", stacked)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cfg.render = False
                    cv2.destroyAllWindows()
            logits = policy(x)
            dist = Categorical(logits=logits)
            action_idx = dist.sample()  # in {0,1,2}
            logps.append(dist.log_prob(action_idx))

            env_action = action_map.to_env(action_idx.item())
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            rewards.append(float(reward))
            ep_reward += float(reward)
            steps += 1

        # Policy gradient update
        returns = discount_cumsum(rewards, cfg.gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=cfg.device)
        # simple baseline: normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for logp, Gt in zip(logps, returns):
            loss = loss - (logp.to(cfg.device) * Gt)

        # entropy bonus
        if len(logps) > 0:
            entropy = Categorical(logits=logits).entropy().mean()
            loss = loss - cfg.entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimizer.step()

        running_return = ep_reward if running_return is None else 0.9 * running_return + 0.1 * ep_reward
        print(f"Episode {ep:03d} | steps={steps:4d} | return={ep_reward:6.2f} | running_return={running_return:6.2f}")

        # Save checkpoint occasionally
        if ep % max(1, cfg.max_episodes // 10) == 0 or ep == cfg.max_episodes:
            torch.save({
                "model_state": policy.state_dict(),
                "cfg": cfg.__dict__,
            }, cfg.save_path)
            print(f"Saved checkpoint to {cfg.save_path}")

    env.close()
    print("Training complete.")


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="End-to-end Pong SNN with snntorch (train or UDP inference)")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Train on Gymnasium ALE/Pong-v5 using event-encoded frames")
    p_train.add_argument("--episodes", type=int, default=Config.max_episodes)
    p_train.add_argument("--render", action="store_true")
    p_train.add_argument("--viz-encoder", action="store_true", help="Show side-by-side ON/OFF maps during training")
    p_train.add_argument("--save", type=str, default=Config.save_path)
    p_train.add_argument("--threshold", type=int, default=Config.diff_threshold)
    p_train.add_argument("--stride", type=int, default=Config.stride)
    p_train.add_argument("--lr", type=float, default=Config.lr)

    p_udp = sub.add_parser("infer-udp", help="Run a trained policy on a live UDP event stream")
    p_udp.add_argument("--model", type=str, default=Config.save_path)
    p_udp.add_argument("--port", type=int, default=9999)

    args = parser.parse_args()

    if args.mode == "train":
        cfg = Config(diff_threshold=args.threshold,
                     stride=args.stride,
                     render=args.render,
                     save_path=args.save,
                     lr=args.lr)
        print(cfg)
        if args.viz_encoder:
            print("Encoder visualization enabled.")
        train(cfg)

    elif args.mode == "infer-udp":
        from src.event_receiver import UDPEventInference
        cfg = Config()
        UDPEventInference(cfg, port=args.port).run(args.model)


if __name__ == "__main__":
    main()

