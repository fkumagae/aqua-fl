"""GRU for fixed-horizon EdgeBox forecasting."""

from __future__ import annotations

import torch
from torch import nn

from .common import INPUT_SIZE, INPUT_WINDOW, OUTPUT_SIZE, run_model_cli, validate_model_input


class GRUForecaster(nn.Module):
    def __init__(
        self,
        input_window: int = INPUT_WINDOW,
        input_size: int = INPUT_SIZE,
        output_size: int = OUTPUT_SIZE,
    ) -> None:
        super().__init__()
        self.input_window = input_window
        self.input_size = input_size
        self.recurrent = nn.GRU(input_size=input_size, hidden_size=32, num_layers=1, batch_first=True)
        self.output = nn.Linear(32, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        validate_model_input(inputs, self.input_window, self.input_size)
        _, hidden = self.recurrent(inputs)
        return self.output(hidden[-1])


def build_model(
    input_window: int = INPUT_WINDOW,
    input_size: int = INPUT_SIZE,
    output_size: int = OUTPUT_SIZE,
) -> GRUForecaster:
    return GRUForecaster(input_window, input_size, output_size)


if __name__ == "__main__":
    run_model_cli("gru", build_model, hidden_size=32, num_layers=1)
