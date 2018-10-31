
import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
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
    #
train_df = pd.read_table(os.path.join(os.getcwd(), 'hw2_data', 'mnli_train.tsv'))
val_df = pd.read_table(os.path.join(os.getcwd(), 'hw2_data', 'mnli_val.tsv'))

genres = train_df['genre'].unique()
train_genre_masks = [np.where(train_df['genre'] == genre)[0] for genre in genres]
val_genre_masks = [np.where(val_df['genre'] == genre)[0] for genre in genres]

rnn_model_dir = os.path.join(
    os.getcwd(), 'outputs',
    'LI_wed_300_we_True_et_rnn_hd_32_ks_3_do_0_vs_50000_nl_1_lr_0.01_wd_1e-05')
rnn_epoch = 8

conv_model_dir = os.path.join(
    os.getcwd(), 'outputs',
    'LI_wed_300_we_True_et_conv_hd_512_ks_3_do_0.2_vs_50000_nl_1_lr_0.001_wd_0')
conv_epoch = 8

rnn_li = LI()
rnn_li.load(rnn_model_dir, rnn_epoch)

conv_li = LI()
conv_li.load(conv_model_dir, conv_epoch)

rnn_scores = []
conv_scores = []
for mask in val_genre_masks:
    prop = [data['prop_mnli.val'][i] for i in mask]
    hyp = [data['hyp_mnli.val'][i] for i in mask]
    labels = [data['labels_mnli.val'][i] for i in mask]
    #
    val_dataset = NLIDataset(prop, hyp, labels, data['max_sent_len_snli.train'])
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=True, num_workers=4)
    rnn_scores += [rnn_li.score(val_loader)]
    conv_scores += [conv_li.score(val_loader)]
#
results = pd.DataFrame({'genre': genres,
                        'conv_accuracy': conv_scores,
                        'rnn_accuracy': rnn_scores})
#
results.to_csv(os.path.join(os.getcwd(), 'outputs', 'genre_report.csv'))
