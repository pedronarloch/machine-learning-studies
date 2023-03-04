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
import numpy as np
import torch

from machine_learning_studies.deep_learning.models.torch_rnn import TorchRNN
from machine_learning_studies.timeseries.data import syntethic

# %%

n_steps = 10
batch_size = 1
series = syntethic.generate_univariate_time_series(
    batch_size=batch_size,
    n_steps=n_steps + 1,
)
stat_quest_series = np.arange(0, n_steps + 1, dtype=np.float32).reshape(
    (batch_size, n_steps + 1)
)[..., np.newaxis]

# %%
X = torch.from_numpy(stat_quest_series[:, :n_steps])
y = torch.from_numpy(stat_quest_series[:, -1])

# %%
rnn_model = TorchRNN(input_size=1, hidden_units=1, output_size=1)

# %%
outcome, _ = rnn_model(X)

# %%
print(outcome)
