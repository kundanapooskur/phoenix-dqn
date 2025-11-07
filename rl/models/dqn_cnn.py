from typing import Tuple

import torch
import torch.nn as nn


class AtariDQN(nn.Module):
    """
    Deep Q-Network used for Atari-like inputs with stacked grayscale frames.
    Input: (batch_size, num_frames, 84, 84)
    Output: (batch_size, num_actions)
    """

    def __init__(self, num_actions: int, num_frames: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # Determine linear layer input size by passing a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, num_frames, 84, 84)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

        # Initialize weights per DQN heuristic
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0  # normalize pixel range [0,1]
        x = self.features(x)
        x = torch.flatten(x, 1)
        q_values = self.head(x)
        return q_values



