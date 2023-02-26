# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from typing import Optional

import torch
import torchmetrics
from torch import nn

from machine_learning_studies.timeseries.data import syntethic

# %%
n_steps = 50
series = syntethic.generate_univariate_time_series(
    batch_size=10000, n_steps=n_steps + 1
)
X_train, y_train = torch.Tensor(series[:7000, :n_steps]), torch.Tensor(
    series[:7000, -1]
)
X_valid, y_valid = torch.Tensor(series[7000:9000, :n_steps]), torch.Tensor(
    series[7000:9000, -1]
)
X_test, y_test = torch.Tensor(series[9000:, :n_steps]), torch.Tensor(
    series[9000:, -1],
)

# %%
# Defining  a Baseline
y_pred = X_valid[:, -1]
baseline_loss_fn = torchmetrics.MeanSquaredError()
print(f"Baseline error: {baseline_loss_fn(y_pred, y_valid)}")


# %%
# A simple RNN
class TorchRNN(nn.Module):
    def __init__(
        self,
        input_size: Optional[int],
        hidden_units: Optional[int],
        output_size: Optional[int],
    ):
        super().__init__()
        self._rnn_block = nn.RNN(
            input_size=input_size,
            hidden_size=output_size,
        )

        self.model = nn.Sequential(
            self._rnn_block,
        )

    def forward(self, X) -> torch.Tensor:
        return self.model(X)


# %%
# Test if the model works with a single sample
torch_rnn_model = TorchRNN(50, None, 1)
torch_rnn_model(X_train)
