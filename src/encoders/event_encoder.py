from __future__ import annotations

import numpy as np
import cv2
import torch


class EventEncoder:
    """Converts consecutive frames to two binary ON/OFF maps using a simple diff threshold.
    Produces a 2 x H x W float32 tensor in [0,1]."""

    def __init__(self, height: int = 84, width: int = 84, diff_threshold: int = 15, stride: int = 2):
        self.h = height
        self.w = width
        self.th = diff_threshold
        self.stride = stride
        self.prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
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
        stacked = np.stack([on, off], axis=0).astype(np.float32)
        return torch.from_numpy(stacked)

