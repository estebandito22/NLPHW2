"""Script to build model inputs."""

import os
import json
from argparse import ArgumentParser

import pandas as pd

from preprocessing.inputbuilder import InputBuilder
from preprocessing.word2vec_embeddings import load_vectors


def main(dataset, vocab_size):
    """Build the inputs."""
    cwd = os.getcwd()

    embd = load_vectors(os.path.join(cwd, 'hw2_data', 'wiki-news-300d-1M.vec'))
    ib = InputBuilder(vocab_size=vocab_size, embeddings=embd)

    snli = pd.read_table(os.path.join(cwd, 'hw2_data', 'snli_train.tsv'))
    train = pd.read_table(os.path.join(cwd, 'hw2_data', dataset+'_train.tsv'))
    val = pd.read_table(os.path.join(cwd, 'hw2_data', dataset+'_val.tsv'))

    ib.fit_transform(snli['sentence1'], snli['sentence2'])
    data_train = ib.transform(train['sentence1'], train['sentence2'])
    data_val = ib.transform(val['sentence1'], val['sentence2'])

    file_names = ['prop', 'hyp', 'token2id', 'id2token', 'max_sent_len', 'emb']
    files_train = [os.path.join(
        cwd, 'inputs', x + '_' + dataset + '.train') for x in file_names]
    files_val = [os.path.join(
        cwd, 'inputs', x + '_' + dataset + '.val') for x in file_names]

    for files, datas in [[files_train, data_train], [files_val, data_val]]:
        for file_name, data in zip(files, datas):
            with open(file_name, 'w') as f:
                json.dump(data, f)

    file_names = [os.path.join(cwd, 'inputs', 'labels' + dataset + '.train'),
                  os.path.join(cwd, 'inputs', 'labels' + dataset + '.val')]
    datas = [train['label'].astype('category').cat.codes.values.tolist(),
             val['label'].astype('category').cat.codes.values.tolist()]
    for file_name, data in zip(file_names, datas):
        with open(file_name, 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-d", "--data_set", default='snli',
                    help="Dataset to build input for.")
    ap.add_argument("-s", "--vocab_size", default=50000, type=int,
                    help="Size of the vocabulary.")
    args = vars(ap.parse_args())
    main(args["data_set"], args["vocab_size"])
