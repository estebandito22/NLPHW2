
import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np

from datasets.nli import NLIDataset
from training.li_trainer import LI

inputs_dir = os.path.join(os.getcwd(), 'inputs')
embeddings = os.path.join(inputs_dir, 'emb_snli.train')
prop_train = os.path.join(inputs_dir, 'prop_mnli.train')
hyp_train = os.path.join(inputs_dir, 'hyp_mnli.train')
prop_val = os.path.join(inputs_dir, 'prop_mnli.val')
hyp_val = os.path.join(inputs_dir, 'hyp_mnli.val')
y_train = os.path.join(inputs_dir, 'labels_mnli.train')
y_val = os.path.join(inputs_dir, 'labels_mnli.val')
max_sent_train = os.path.join(inputs_dir, 'max_sent_len_snli.train')
max_sent_val = os.path.join(inputs_dir, 'max_sent_len_mnli.val')

files = [embeddings, prop_train, hyp_train, prop_val, hyp_val, y_train, y_val,
         max_sent_train, max_sent_val]

data = defaultdict(list)
for file in files:
    with open(file, 'r') as f:
        r = json.load(f)
        name = file.split('/')[-1]
        data[name] = r

train_df = pd.read_table(os.path.join(os.getcwd(), 'hw2_data', 'mnli_train.tsv'))
val_df = pd.read_table(os.path.join(os.getcwd(), 'hw2_data', 'mnli_val.tsv'))

genres = train_df['genre'].unique()
train_genre_masks = [np.where(train_df['genre'] == genre)[0] for genre in genres]
val_genre_masks = [np.where(val_df['genre'] == genre)[0] for genre in genres]

rnn_model_dir = os.path.join(
    os.getcwd(), 'outputs',
    'LI_wed_300_we_True_et_rnn_hd_32_ks_3_do_0_vs_50000_nl_1_lr_0.01_wd_1e-05')
rnn_epoch = 8

rnn_li = LI()
rnn_li.load(rnn_model_dir, rnn_epoch)

for genre, train_mask, val_mask in zip(genres, train_genre_masks, val_genre_masks):
    save_dir = os.path.join(os.getcwd(), 'outputs', 'finetune', genre)
    prop_train = [data['prop_mnli.train'][i] for i in train_mask]
    hyp_train = [data['hyp_mnli.train'][i] for i in train_mask]
    labels_train = [data['labels_mnli.train'][i] for i in train_mask]
    prop_val = [data['prop_mnli.val'][i] for i in val_mask]
    hyp_val = [data['hyp_mnli.val'][i] for i in val_mask]
    labels_val = [data['labels_mnli.val'][i] for i in val_mask]
    #
    rnn_li = LI()
    rnn_li.load(rnn_model_dir, rnn_epoch)
    setattr(rnn_li, 'best_score', 0.0)
    #
    train_dataset = NLIDataset(
        prop_train, hyp_train, labels_train, data['max_sent_len_snli.train'])
    val_dataset = NLIDataset(
        prop_val, hyp_val, labels_val, data['max_sent_len_snli.train'])
    rnn_li.fit(train_dataset, val_dataset, save_dir, True)
