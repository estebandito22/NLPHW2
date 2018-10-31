"""PyTorch class for convnet sentence encoder."""

from torch import nn
import numpy as np


class ConvEncoder2(nn.Module):

    """ConvEncoder used to encode a sentence."""

    def __init__(self, dict_args):
        """
        Initialize ConvEncoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(ConvEncoder2, self).__init__()
        self.word_embdim = dict_args["word_embdim"]
        self.hidden_size = dict_args["hidden_size"] * 2
        self.input_dim = dict_args["max_sent_len"]
        self.dropout = dict_args["dropout"]
        self.kernel_size = dict_args["kernel_size"]
        self.bias = True
        # input_size = batch size x word_embdim x 78
        self.layer1 = nn.Conv1d(
            in_channels=self.word_embdim, out_channels=self.hidden_size,
            kernel_size=self.kernel_size, stride=1,
            padding=self.kernel_size // 2, bias=self.bias)
        self.drop1 = nn.Dropout(self.dropout)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Conv1d(
            in_channels=self.hidden_size, out_channels=self.hidden_size,
            kernel_size=self.kernel_size, stride=1,
            padding=self.kernel_size // 2, bias=self.bias)
        self.pool2 = nn.MaxPool1d(kernel_size=78)
        self.drop2 = nn.Dropout(self.dropout)
        self.relu2 = nn.ReLU()

        # initizlize weights
        nn.init.xavier_normal_(self.layer1.weight, np.sqrt(2))
        nn.init.xavier_normal_(self.layer2.weight, np.sqrt(2))

    def forward(self, x):
        """Execute forward pass."""
        x = self.layer1(x)
        # x = self.pool1(x)
        x = self.drop1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.relu2(x)

        return x.view(-1, self.hidden_size)
