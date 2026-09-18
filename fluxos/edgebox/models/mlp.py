"""Multilayer perceptron for fixed-horizon EdgeBox forecasting."""

from __future__ import annotations

import torch
from torch import nn

from .common import INPUT_SIZE, INPUT_WINDOW, OUTPUT_SIZE, run_model_cli, validate_model_input


class MLPForecaster(nn.Module):
    def __init__(
        self,
        input_window: int = INPUT_WINDOW,
        input_size: int = INPUT_SIZE,
        output_size: int = OUTPUT_SIZE,
    ) -> None:
        super().__init__()
        self.input_window = input_window
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_window * input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        validate_model_input(inputs, self.input_window, self.input_size)
        return self.network(inputs)


def build_model(
    input_window: int = INPUT_WINDOW,
    input_size: int = INPUT_SIZE,
    output_size: int = OUTPUT_SIZE,
) -> MLPForecaster:
    return MLPForecaster(input_window, input_size, output_size)


if __name__ == "__main__":
    run_model_cli("mlp", build_model, hidden_size=None, num_layers=None)
