from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNActorCritic(nn.Module):
    """Shared spiking backbone with separate policy and value heads.

    Input: (B, C=2, H=84, W=84)
    Outputs: policy logits (B, A), value (B, 1)
    """

    def __init__(self, in_ch: int = 2, n_actions: int = 3, beta: float = 0.95, device: str = "cpu"):
        super().__init__()
        self.device = device
        spike_grad = surrogate.fast_sigmoid()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

        self.fc_in = 64 * 11 * 11
        self.fc = nn.Linear(self.fc_in, 128)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

        self.to(self.device)

    def reset_state(self) -> None:
        # self.lif1.reset_states()
        # self.lif2.reset_states()
        # self.lif3.reset_states()
        # self.lif4.reset_states()
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()
        self.lif4.reset_mem()

    def forward(self, x: torch.Tensor):
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

        logits = self.policy_head(spk4)
        value = self.value_head(spk4)
        return logits, value

