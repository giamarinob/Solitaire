import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, input_size=29, output_size=83, hidden_size=128, device=None):
        super(DuelingDQN, self).__init__()

        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Shared base layers
        self.shared_fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)

        # Value stream
        self.value_fc = nn.Linear(hidden_size, hidden_size)
        self.value_out = nn.Linear(hidden_size, 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_size, hidden_size)
        self.advantage_out = nn.Linear(hidden_size, output_size)

        # Move model to the appropriate device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the right device

        # Shared layers
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        # Value stream
        v = F.relu(self.value_fc(x))
        v = self.value_out(v)  # shape: [batch, 1]

        # Advantage stream
        a = F.relu(self.advantage_fc(x))
        a = self.advantage_out(a)  # shape: [batch, output_size]

        # Combine streams
        q = v + (a - a.mean(dim=1, keepdim=True))  # shape: [batch, output_size]
        return q
