import torch
from torch import nn


class TorchRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_units: int,
        output_size: int,
        n_layers: int = 1,
        batch_first: bool = False,
    ):
        """
        Default RNN definition.

        Args:
             input_size: The size of the input
             hidden_units: the number of features in the RNN output
             output_size: the size of the output.
             n_layers: the number of layers that make up the RNN
             greater than 1 means that you'll create a stacked RNN
             batch_first: if the input/output of the RNN will have
             the batch_size as the first dimension
             (batch_size, seq_length, hidden_dim)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_units = hidden_units
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_first = batch_first

        self._rnn_block = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_units,
            num_layers=self.n_layers,
            batch_first=self.batch_first,
        )

        # Readout layer
        self.fc = nn.Linear(in_features=hidden_units, out_features=output_size)

    def forward(self, x):
        output = self._rnn_block(
            x,
            self._init_hidden(x.size(0)),
        )
        output = self.fc(output[:, -1, :])
        return output

    def _init_hidden(self, batch_size) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.hidden_units)
