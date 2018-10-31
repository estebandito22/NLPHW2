"""PyTorch class for word embedding."""

import torch
import torch.nn as nn
import numpy as np


class WordEmbeddings(nn.Module):

    """Class to embed words."""

    def __init__(self, dict_args):
        """
        Initialize WordEmbeddings.

        Args
            dict_args: dictionary containing the following keys:
                word_embdim: The dimension of the lookup embedding.
                vocab_size: The count of words in the data set.
                word_embeddings: Pretrained embeddings.
        """
        super(WordEmbeddings, self).__init__()

        self.word_embdim = dict_args["word_embdim"]
        self.vocab_size = dict_args["vocab_size"]
        self.word_embeddings = dict_args["word_embeddings"]

        self.embeddings = nn.Embedding(
            self.vocab_size, self.word_embdim, padding_idx=0)

        if self.word_embeddings is not None:
            we = torch.from_numpy(np.array(self.word_embeddings)).float()
            self.embeddings.weight = nn.Parameter(we)
            self.embeddings.weight.requires_grad = False

    def forward(self, indexseq):
        """
        Forward pass.

        Args
            indexseq: A tensor of sequences of word indexes of size
                      batch_size x seqlen.
        """
        # seqlen x batch_size x embd_dim
        return self.embeddings(indexseq).permute(1, 0, 2)
