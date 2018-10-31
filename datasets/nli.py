"""Class for loading MNLI dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset


class NLIDataset(Dataset):

    """Class to load MNLI dataset."""

    def __init__(self, p_sent, h_sent, y, max_sent_len):
        """
        Initialize NMTDataset.

        Args
            p_sent: list of index sequences from the premise sentence.
            h_sent: list of index sequences from the hypothesis sentence.
            y: list of classes.
            max_sent_len: integer representing the maximum sentence length.
        """
        self.p_sent = p_sent
        self.h_sent = h_sent
        self.y = y
        self.max_sent_len = max_sent_len

    def __len__(self):
        """Return length of dataset."""
        return len(self.p_sent)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        p_sent = self.p_sent[i]
        h_sent = self.h_sent[i]
        y = np.array(self.y[i])

        len_p_sent = min(self.max_sent_len, len(p_sent))
        len_h_sent = min(self.max_sent_len, len(h_sent))

        p_sent_pad = np.zeros(self.max_sent_len)
        for j in range(len_p_sent):
            p_sent_pad[j] = p_sent[j]

        h_sent_pad = np.zeros(self.max_sent_len)
        for j in range(len_h_sent):
            h_sent_pad[j] = h_sent[j]

        p_sent_pad = torch.from_numpy(p_sent_pad).long()
        h_sent_pad = torch.from_numpy(h_sent_pad).long()
        len_p_sent = torch.from_numpy(np.array(len_p_sent)).long()
        len_h_sent = torch.from_numpy(np.array(len_h_sent)).long()
        y = torch.from_numpy(y).long()

        return {'p_sent': p_sent_pad, 'h_sent': h_sent_pad, 'y': y,
                'len_p_sent': len_p_sent, 'len_h_sent': len_h_sent}
