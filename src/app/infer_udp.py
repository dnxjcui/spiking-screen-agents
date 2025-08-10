#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import numpy as np
import torch

from src.utils import Config
from src.models import make_model
from src.event_receiver import UDPEventInference


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a trained policy on a live UDP event stream")
    parser.add_argument("--model", type=str, default=Config.save_path)
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    cfg = Config()
    UDPEventInference(cfg, port=args.port).run(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

