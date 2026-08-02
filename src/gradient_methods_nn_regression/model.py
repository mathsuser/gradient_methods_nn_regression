# Minimal shallow neural-network regression model used for the baseline experiment.
import torch
import torch.nn as nn


class TinyRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 16)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.fc1(x))
        return self.fc2(hidden)

    def count_trainable_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
