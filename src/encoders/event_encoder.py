from __future__ import annotations

import numpy as np
import cv2
import torch


class EventEncoder:
    """Converts consecutive frames to event-based representation suitable for SNNs.
    
    Creates ON/OFF event channels from pixel differences between consecutive frames,
    mimicking the behavior of Dynamic Vision Sensor (DVS) cameras.
    Produces a 2 x H x W float32 tensor in [0,1] where:
    - Channel 0: ON events (brightness increases)
    - Channel 1: OFF events (brightness decreases)
    """

    def __init__(self, height: int = 84, width: int = 84, diff_threshold: int = 15, stride: int = 2, 
                 temporal_window: int = 1, noise_filter: bool = True):
        self.h = height
        self.w = width
        self.th = diff_threshold
        self.stride = stride
        self.temporal_window = temporal_window  # Number of frames to accumulate events over
        self.noise_filter = noise_filter
        self.prev_gray: np.ndarray | None = None
        self.event_buffer = []  # Store recent events for temporal accumulation

    def reset(self) -> None:
        self.prev_gray = None
        self.event_buffer.clear()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.stride > 1:
            frame = frame[::self.stride, ::self.stride]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return gray

    def _apply_noise_filter(self, events: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply morphological filtering to reduce noise in event data."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Remove isolated pixels (noise)
        events = cv2.morphologyEx(events, cv2.MORPH_OPEN, kernel)
        return events

    def encode(self, frame: np.ndarray) -> torch.Tensor:
        """Encode frame differences as event-based representation.
        
        Returns:
            torch.Tensor: Shape (2, H, W) with ON events in channel 0 and OFF events in channel 1
        """
        gray = self._preprocess(frame)
        
        if self.prev_gray is None:
            # First frame: no events generated
            self.prev_gray = gray
            on = np.zeros_like(gray, dtype=np.uint8)
            off = np.zeros_like(gray, dtype=np.uint8)
        else:
            # Compute pixel differences
            diff = gray.astype(np.int16) - self.prev_gray.astype(np.int16)
            
            # Generate ON/OFF events based on threshold
            on = (diff > self.th).astype(np.uint8)
            off = (diff < -self.th).astype(np.uint8)
            
            # Apply noise filtering if enabled
            if self.noise_filter:
                on = self._apply_noise_filter(on)
                off = self._apply_noise_filter(off)
            
            self.prev_gray = gray.copy()  # Use .copy() to avoid reference issues
        
        # Store in buffer for temporal accumulation
        event_frame = np.stack([on, off], axis=0).astype(np.float32)
        self.event_buffer.append(event_frame)
        
        # Keep only the most recent frames within the temporal window
        if len(self.event_buffer) > self.temporal_window:
            self.event_buffer.pop(0)
        
        # Accumulate events over temporal window
        if len(self.event_buffer) > 1:
            accumulated_events = np.sum(self.event_buffer, axis=0)
            # Normalize to [0, 1] range
            accumulated_events = np.clip(accumulated_events / len(self.event_buffer), 0, 1)
        else:
            accumulated_events = event_frame
        
        return torch.from_numpy(accumulated_events)

