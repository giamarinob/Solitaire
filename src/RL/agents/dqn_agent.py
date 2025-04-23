import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from RL.models.dueling_dqn import DuelingDQN

class DQNAgent:
    def __init__(self, input_size, output_size, hidden_size=128, buffer=None, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=10000, target_update_freq=1000, device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.q_network = DuelingDQN(input_size, output_size, hidden_size, self.device)
        self.target_network = DuelingDQN(input_size, output_size, hidden_size, self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = buffer
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.target_update_freq = target_update_freq
        self.output_size = output_size

    def select_action(self, state):
        self.steps_done += 1
        epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)

        if random.random() < epsilon:
            return random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                return q_values.argmax().item()

    def train_step(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_network(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()
