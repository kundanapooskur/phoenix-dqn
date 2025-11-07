from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.models.dqn_cnn import AtariDQN


@dataclass
class DQNConfig:
    num_actions: int
    num_frames: int = 4
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    target_update_interval: int = 10_000
    grad_norm_clip: float = 10.0
    device: str = "cpu"


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.online = AtariDQN(config.num_actions, config.num_frames).to(self.device)
        self.target = AtariDQN(config.num_actions, config.num_frames).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.steps = 0

    @torch.no_grad()
    def act_epsilon_greedy(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.config.num_actions)
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.uint8)
        q_values = self.online(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    @torch.no_grad()
    def act_boltzmann(self, state: np.ndarray, temperature: float) -> int:
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.uint8)
        q_values = self.online(state_t)[0]
        probs = torch.softmax(q_values / max(1e-6, temperature), dim=0)
        action = torch.multinomial(probs, 1).item()
        return int(action)

    def learn(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
        states, actions, rewards, next_states, dones = batch

        states_t = torch.as_tensor(states, device=self.device, dtype=torch.uint8)
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        next_states_t = torch.as_tensor(next_states, device=self.device, dtype=torch.uint8)
        dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.bool)

        q_values = self.online(states_t)
        action_q = q_values.gather(1, actions_t.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target(next_states_t).max(dim=1).values
            target_q = rewards_t + self.config.gamma * next_q * (~dones_t)

        loss = self.loss_fn(action_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_norm_clip)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({
            "model": self.online.state_dict(),
            "steps": self.steps,
            "config": self.config.__dict__,
        }, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        data = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.online.load_state_dict(data["model"])
        self.target.load_state_dict(self.online.state_dict())
        self.steps = int(data.get("steps", 0))



