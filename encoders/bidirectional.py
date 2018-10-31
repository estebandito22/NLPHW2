"""PyTorch classes for a recurrent network encoder."""

import torch
from torch import nn


class BidirectionalEncoder(nn.Module):

    """Bidirectional recurrent network to encode sentence."""

    def __init__(self, dict_args):
        """
        Initialize BidirectionalEncoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(BidirectionalEncoder, self).__init__()
        self.word_embdim = dict_args["word_embdim"]
        self.vocab_size = dict_args["vocab_size"]
        self.hidden_size = dict_args["hidden_size"]
        self.num_layers = dict_args["num_layers"]
        self.dropout = dict_args["dropout"]
        self.batch_size = dict_args["batch_size"]

        # GRU
        self.hidden = None
        self.init_hidden(self.batch_size)

        self.rnn = nn.GRU(
            input_size=self.word_embdim, hidden_size=self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout,
            bidirectional=True)

    def init_hidden(self, batch_size):
        """Initialize the hidden state of the RNN."""
        if torch.cuda.is_available():
            self.hidden = torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size).cuda()
        else:
            self.hidden = torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size)

    def detach_hidden(self, batch_size):
        """Detach the hidden state of the RNN."""
        _, hidden_batch_size, _ = self.hidden.size()
        if hidden_batch_size != batch_size:
            self.init_hidden(batch_size)
        else:
            detached_hidden = self.hidden.detach()
            detached_hidden.zero_()
            self.hidden = detached_hidden

    def forward(self, seq_word_embds):
        """Forward pass."""
        _, out = self.rnn(seq_word_embds, self.hidden)

        return out
