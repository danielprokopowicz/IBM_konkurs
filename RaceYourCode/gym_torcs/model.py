"""
model.py — Neural network for TORCS AI driver
Output: [steer, accel] — brake is handled by code
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim=14):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
        )
        self.steer_head = nn.Linear(64, 1)
        self.accel_head = nn.Linear(64, 1)
        self.brake_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.fc(x)
        steer = torch.tanh(self.steer_head(h))
        accel = torch.sigmoid(self.accel_head(h))
        brake = torch.sigmoid(self.brake_head(h))
        return torch.cat([steer, accel, brake], dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim=14, action_dim=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),                     nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, s, a):
        return self.fc(torch.cat([s, a], dim=1))