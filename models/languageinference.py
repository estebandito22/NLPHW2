"""PyTorch classe for encoder-decoder netowork."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from embeddings.wordembedding import WordEmbeddings
from encoders.bidirectional import BidirectionalEncoder
from encoders.conv2 import ConvEncoder2


class LanguageInference(nn.Module):

    """Language Inference model."""

    def __init__(self, dict_args):
        """
        Initialize LanguageInference.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(LanguageInference, self).__init__()
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.enc_type = dict_args["enc_type"]
        self.hidden_dim = dict_args["hidden_dim"]
        self.dropout = dict_args["dropout"]
        self.vocab_size = dict_args["vocab_size"]
        self.batch_size = dict_args["batch_size"]
        self.num_layers = dict_args["num_layers"]
        self.max_sent_len = dict_args["max_sent_len"]
        self.kernel_size = dict_args["kernel_size"]

        # encoder
        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size,
                     'hidden_size': self.hidden_dim,
                     'num_layers': self.num_layers,
                     'dropout': self.dropout,
                     'batch_size': self.batch_size,
                     'max_sent_len': self.max_sent_len,
                     'kernel_size': self.kernel_size}

        if self.enc_type == 'rnn':
            self.encoder = BidirectionalEncoder(dict_args)
            self.encoder.init_hidden(self.batch_size)
        elif self.enc_type == 'conv':
            self.encoder = ConvEncoder2(dict_args)
        else:
            raise ValueError("Unrecognized encoder type!")

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size,
                     'word_embeddings': self.word_embeddings}
        self.word_embd = WordEmbeddings(dict_args)

        self.fc1 = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim * 4, 3)

    def forward(self, p_indexseq, len_p_indexseq, h_indexseq, len_h_indexseq):
        """Forward pass."""
        batch_size = p_indexseq.size()[0]

        # process p_indexseq
        # seqlen x batch_size x embedding dim
        p_seq_word_embds = self.word_embd(p_indexseq)

        if self.enc_type == 'rnn':
            self.encoder.detach_hidden(batch_size)
            len_p_indexseq, p_len2sort = len_p_indexseq.sort(
                0, descending=True)
            _, p_sort2len = p_len2sort.sort(0, descending=False)
            p_seq_word_embds = p_seq_word_embds[:, p_len2sort, :]
            p_seq_word_embds = pack_padded_sequence(
                p_seq_word_embds, len_p_indexseq)

        if self.enc_type == 'conv':
            p_seq_word_embds = p_seq_word_embds.permute(1, 2, 0)

        # RNN: num layers * num_directions x batch_size x hidden_size
        # Conv: batch_size x hidden_size * num_directions
        p_seq_enc_state = self.encoder(p_seq_word_embds)

        if self.enc_type == 'rnn':
            p_seq_enc_state = p_seq_enc_state[:, p_sort2len, :]
            p_seq_enc_state = p_seq_enc_state.permute(1, 0, 2).\
                contiguous().view(batch_size, -1)

        # process h_indexseq
        # seqlen x batch_size x embedding dim
        h_seq_word_embds = self.word_embd(h_indexseq)

        if self.enc_type == 'rnn':
            self.encoder.init_hidden(batch_size)
            len_h_indexseq, h_len2sort = len_h_indexseq.sort(
                0, descending=True)
            _, h_sort2len = h_len2sort.sort(0, descending=False)
            h_seq_word_embds = h_seq_word_embds[:, h_len2sort, :]

        if self.enc_type == 'conv':
            h_seq_word_embds = h_seq_word_embds.permute(1, 2, 0)

        # RNN: num layers * num_directions x batch_size x hidden_size
        # Conv: batch_size x hidden_size * num_directions
        h_seq_enc_state = self.encoder(h_seq_word_embds)

        if self.enc_type == 'rnn':
            h_seq_enc_state = h_seq_enc_state[:, h_sort2len, :]
            h_seq_enc_state = h_seq_enc_state.permute(1, 0, 2).\
                contiguous().view(batch_size, -1)

        # remainder of network
        fc1_in = torch.cat([p_seq_enc_state, h_seq_enc_state], dim=1)
        x = self.fc1(fc1_in)
        x = self.relu1(x)
        x = self.fc2(x)
        log_probs = F.log_softmax(x, dim=1)

        return log_probs


if __name__ == '__main__':
    lengths1 = torch.tensor([2,4,9,3]).long()
    lengths2 = torch.tensor([2,10,7,8]).long()

    lengths1, order1 = lengths1.sort(0, descending=True)
    lengths2, order2 = lengths2.sort(0, descending=True)

    output = torch.tensor([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])

    input = output.t()
    input
    input[:, order1]

    output
    output = output[order1]
    output

    sortedorder, sortedorderindex = order1.sort(0, descending=False)

    sortedorderindex

    output[sortedorderindex]
