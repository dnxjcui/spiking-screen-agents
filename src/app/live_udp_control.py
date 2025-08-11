#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import threading
import numpy as np
import torch
import pyautogui
import keyboard

from src.utils import Config
from src.models import make_model
from src.streaming.udp import UDPEventReceiver


def events_to_frame(events, width: int, height: int, window_ms: float = 50.0):
    now = time.time()
    cutoff = now - (window_ms / 1000.0)
    on = np.zeros((height, width), dtype=np.uint8)
    off = np.zeros((height, width), dtype=np.uint8)
    for e in events:
        if e.get("received_time", 0.0) >= cutoff:
            x, y, pol = int(e["x"]), int(e["y"]), int(e["polarity"])
            if 0 <= x < width and 0 <= y < height:
                if pol > 0:
                    on[y, x] = 1
                else:
                    off[y, x] = 1
    stacked = np.stack([on, off], axis=0).astype(np.float32)
    return torch.from_numpy(stacked)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live UDP control using trained SNN policy")
    parser.add_argument("--model", type=str, default=Config.save_path)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--kill-key", type=str, default="esc", help="Hotkey to stop control loop")
    args = parser.parse_args()

    cfg = Config()
    model = make_model("snn", in_ch=2, n_actions=3, beta=cfg.beta, device=cfg.device)
    state = torch.load(args.model, map_location=cfg.device)
    model.load_state_dict(state["model_state"])
    model.eval()
    model.reset_state()

    recv = UDPEventReceiver(port=args.port)
    if not recv.start():
        return 1
    print("Press ESC to stop.")

    try:
        while True:
            if keyboard.is_pressed(args.kill_key):
                print("Kill key pressed. Exiting.")
                break
            events = recv.event_buffer.get_recent_events(time_window=args.window_ms / 1000.0)
            frame = events_to_frame(events, width=cfg.width, height=cfg.height, window_ms=args.window_ms)
            x = frame.unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                act = int(np.argmax(probs))
            # Map actions to keys: 0=noop, 1=up, 2=down
            if act == 1:
                keyboard.press_and_release('up')
            elif act == 2:
                keyboard.press_and_release('down')
            time.sleep(args.window_ms / 1000.0)
    finally:
        recv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

