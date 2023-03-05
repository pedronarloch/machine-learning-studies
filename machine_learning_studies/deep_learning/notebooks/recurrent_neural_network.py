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
import torch
from torch import nn
from tqdm.auto import tqdm

from machine_learning_studies.deep_learning.models.torch_rnn import TorchRNN
from machine_learning_studies.timeseries.data import syntethic

# %%
n_steps = 50
batch_size = 10000
seq_output_size = 1
series = syntethic.generate_univariate_time_series(
    batch_size=batch_size,
    # +1 because the latest element will serve as Y
    n_steps=n_steps + seq_output_size,
)

print(series.shape)

# %%
X_train, y_train = torch.from_numpy(series[:7000, :n_steps]), torch.from_numpy(
    series[:7000, -seq_output_size]
)
X_valid, y_valid = (
    torch.from_numpy(series[7000:9000, :n_steps]),
    torch.from_numpy(series[7000:9000, -seq_output_size]),
)
X_test, y_test = (
    torch.from_numpy(series[9000:, :n_steps]),
    torch.from_numpy(series[9000:, -seq_output_size]),
)

print(f"X_train\t{X_train.shape} , y_train\t{y_train.shape}")
print(f"X_valid\t{X_valid.shape} , y_valid\t{y_valid.shape}")
print(f"X_test\t{X_test.shape} , y_train\t{y_test.shape}")

# %%
rnn_model = TorchRNN(
    input_size=1,
    hidden_units=1,
    output_size=seq_output_size,
    batch_first=False,
)  # For a single RNN unit the hidden_unit and output_size are 1.

# %%
outcome, hidden_states = rnn_model(X_train[0])
print(outcome.shape, hidden_states.shape)

# %%
outcome[-seq_output_size].shape, y_train[0].shape

# %%
loss_fn = nn.MSELoss()
loss_fn(outcome[-seq_output_size], y_train[0])

# %%
loss_fn(outcome[-1], y_train[0])

# %%
hidden_states

# %%
# Hyper parameters
n_epochs = 1000
lr = 0.01

rnn_model = TorchRNN(
    input_size=1, hidden_units=1, output_size=seq_output_size, batch_first=True
)

# Loss, Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=rnn_model.parameters(), lr=lr)

# %%
for epoch in tqdm(range(n_epochs)):
    rnn_model.train()

    # 1. Forward
    output, hidden = rnn_model.forward(X_train)

    # 2. Calculate the Loss
    loss = loss_fn(output[:, -seq_output_size], y_train)

    # 3. Optimizer.zero_grad
    optimizer.zero_grad()

    # 4. Backprop()
    loss.backward()

    # 5. Optimizer Step
    optimizer.step()

    rnn_model.eval()
    with torch.inference_mode():
        output_test, hidden_test = rnn_model.forward(X_test)
        test_loss = loss_fn(output_test[:, -seq_output_size], y_test)

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch}/{n_epochs} | "
            f"Train loss: {loss:.5f} | "
            f"Test loss: {test_loss:.5f}"
        )


# %%
