import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_size=8, action_size=6, hidden_size=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        return self.network(state)


if __name__ == "__main__":
    model = DQN()

    sample_state = torch.zeros(1, 8)
    q_values = model(sample_state)

    print("Input shape:", sample_state.shape)
    print("Output shape:", q_values.shape)
    print("Q-values:", q_values)
