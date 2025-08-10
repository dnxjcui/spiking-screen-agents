from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNPolicy(nn.Module):
    """A small CNN + LIF network. Readout head outputs logits over 3 actions (NOOP, UP, DOWN)."""

    def __init__(self, in_ch: int = 2, n_actions: int = 3, beta: float = 0.95, device: str = "cpu"):
        super().__init__()
        self.device = device
        spike_grad = surrogate.fast_sigmoid()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        # compute conv output size for 84x84 with strides 2,2,2
        # 84 -> 42 -> 21 -> 11 (approx with padding)
        self.fc_in = 64 * 11 * 11
        self.fc = nn.Linear(self.fc_in, 128)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False) # why did we do true?
        self.head = nn.Linear(128, n_actions)

        self.to(self.device)

    def reset_state(self) -> None:
        # Reset hidden states for all LIF layers
        # self.lif1.reset_states()
        # self.lif2.reset_states()
        # self.lif3.reset_states()
        # self.lif4.reset_states()
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        self.lif3.reset_hidden()
        self.lif4.reset_hidden()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, H, W), returns logits (B, n_actions)."""
        x = x.to(self.device)
        z = self.conv1(x)
        spk1, _ = self.lif1(z)

        z = self.conv2(spk1)
        spk2, _ = self.lif2(z)

        z = self.conv3(spk2)
        spk3, _ = self.lif3(z)

        z = spk3.flatten(1)
        z = self.fc(z)
        spk4, _ = self.lif4(z)

        logits = self.head(spk4)
        return logits

