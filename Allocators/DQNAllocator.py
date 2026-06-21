import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class _QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAllocator:
    def __init__(
        self,
        total_prb=50,
        state_dim=12,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_size=10000,
    ):
        self.total_prb = total_prb

        # Enumerate all valid integer splits (a, b, c) with a+b+c=total_prb, each >= 1
        step = 5
        self.actions = np.array([
            [a, b, total_prb - a - b]
            for a in range(step, total_prb - step + 1, step)
            for b in range(step, total_prb - a, step)
            if total_prb - a - b >= step
        ])
        n_actions = len(self.actions)

        self.q_net     = _QNetwork(state_dim, n_actions)
        self.target_net = _QNetwork(state_dim, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.buffer        = deque(maxlen=buffer_size)

        self.last_action_idx = None

    def get_allocation(self, requests, state=None):
        if state is None or np.random.rand() < self.epsilon:
            idx = np.random.randint(len(self.actions))
        else:
            s = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                idx = self.q_net(s).argmax().item()

        self.last_action_idx = idx
        return self.actions[idx].copy()

    def store(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, float(done)))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        q_vals = self.q_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
