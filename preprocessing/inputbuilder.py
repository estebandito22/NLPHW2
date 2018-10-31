"""Class to tokenize text of multiple languages."""

import string
from collections import Counter

import numpy as np
from tqdm import tqdm


class InputBuilder(object):

    """Class to transform raw text to index sequences and build vocab."""

    def __init__(self, vocab_size, embeddings=None):
        """Initialize InputBuilder."""
        self.vocab_size = vocab_size
        self.embeddings = embeddings

        self.PAD_IDX = 0
        self.UNK_IDX = 1

        self.punctuations = string.punctuation

        self.token2id = None
        self.id2token = None
        self.vocab_embedding = None

    def _lower_case_remove_punc(self, tokens):
        return [token.lower() for token in tokens
                if token not in self.punctuations
                and token not in ['\n']]

    def _build_vocab(self, all_tokens):
        """Build vocabulary and indexes."""
        if self.embeddings:
            all_tokens = np.array(all_tokens)
            embd_and_data_tokens = list(self.embeddings.keys() & all_tokens)
            mask = np.in1d(all_tokens, embd_and_data_tokens)
            all_tokens = all_tokens[mask]

        token_counter = Counter(all_tokens)
        vocab, _ = zip(*token_counter.most_common(self.vocab_size))

        id2token = list(vocab)
        token2id = dict(zip(vocab, range(2, 2+len(vocab))))
        id2token = ['<pad>', '<unk>'] + id2token
        token2id['<pad>'] = self.PAD_IDX
        token2id['<unk>'] = self.UNK_IDX

        return token2id, id2token

    def _build_embedding(self):
        self.vocab_embedding = []
        self.vocab_embedding += [np.random.rand(300).tolist()]
        self.vocab_embedding += [np.random.rand(300).tolist()]
        for token in self.id2token:
            if token not in ['<pad>', '<unk>']:
                self.vocab_embedding += [list(self.embeddings[token])]

    def fit_transform(self, p_samples, h_samples):
        """Transform raw text samles to tokens."""
        p_token_dataset = []
        p_index_dataset = []
        h_token_dataset = []
        h_index_dataset = []
        all_tokens = []
        max_sent_len = 0

        for samples, dataset in [[p_samples, p_token_dataset],
                                 [h_samples, h_token_dataset]]:
            for tokens in tqdm(samples):
                tokens = self._lower_case_remove_punc(tokens.split(' '))
                dataset.append(tokens)
                all_tokens += tokens

                if len(tokens) > max_sent_len:
                    max_sent_len = len(tokens)

        self.token2id, self.id2token = self._build_vocab(all_tokens)

        if self.embeddings:
            self._build_embedding()

        for t_dataset, i_dataset in [[p_token_dataset, p_index_dataset],
                                     [h_token_dataset, h_index_dataset]]:
            for tokens in t_dataset:
                i_dataset += [[self.token2id[token] if token in self.token2id
                               else self.UNK_IDX for token in tokens]]

        return p_index_dataset, h_index_dataset, self.token2id, self.id2token,\
            max_sent_len, self.vocab_embedding

    def transform(self, p_samples, h_samples):
        """Transform raw text samles to tokens."""
        p_token_dataset = []
        p_index_dataset = []
        h_token_dataset = []
        h_index_dataset = []
        max_sent_len = 0

        for samples, dataset in [[p_samples, p_token_dataset],
                                 [h_samples, h_token_dataset]]:
            for tokens in tqdm(samples):
                tokens = self._lower_case_remove_punc(tokens.split(' '))
                dataset.append(tokens)

                if len(tokens) > max_sent_len:
                    max_sent_len = len(tokens)

        for t_dataset, i_dataset in [[p_token_dataset, p_index_dataset],
                                     [h_token_dataset, h_index_dataset]]:
            for tokens in t_dataset:
                i_dataset += [[self.token2id[token] if token in self.token2id
                               else self.UNK_IDX for token in tokens]]

        return p_index_dataset, h_index_dataset, self.token2id, self.id2token,\
            max_sent_len, None
