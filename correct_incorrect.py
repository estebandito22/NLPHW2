import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets.nli import NLIDataset
from training.li_trainer import LI

inputs_dir = os.path.join(os.getcwd(), 'inputs')
embeddings = os.path.join(inputs_dir, 'emb_snli.train')
prop_train = os.path.join(inputs_dir, 'prop_snli.train')
hyp_train = os.path.join(inputs_dir, 'hyp_snli.train')
prop_val = os.path.join(inputs_dir, 'prop_snli.val')
hyp_val = os.path.join(inputs_dir, 'hyp_snli.val')
y_train = os.path.join(inputs_dir, 'labels_snli.train')
y_val = os.path.join(inputs_dir, 'labels_snli.val')
max_sent_train = os.path.join(inputs_dir, 'max_sent_len_snli.train')
max_sent_val = os.path.join(inputs_dir, 'max_sent_len_snli.val')

files = [embeddings, prop_train, hyp_train, prop_val, hyp_val, y_train, y_val,
         max_sent_train, max_sent_val]

data = defaultdict(list)
for file in files:
    with open(file, 'r') as f:
        r = json.load(f)
        name = file.split('/')[-1]
        data[name] = r

val_dataset = NLIDataset(
    data['prop_snli.val'], data['hyp_snli.val'], data['labels_snli.val'],
    data['max_sent_len_snli.train'])

rnn_model_dir = os.path.join(
    os.getcwd(), 'outputs',
    'LI_wed_300_we_True_et_rnn_hd_32_ks_3_do_0_vs_50000_nl_1_lr_0.01_wd_1e-05')
rnn_epoch = 8

rnn_li = LI()
rnn_li.load(rnn_model_dir, rnn_epoch)

val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=4)

log_probs, truth = rnn_li.predict(val_loader)
predictions = torch.argmax(log_probs, dim=1)
correct = np.where(predictions == truth)[0]
incorrect = np.where(predictions != truth)[0]

val_df = pd.read_table(os.path.join(os.getcwd(), 'hw2_data', 'snli_val.tsv'))

correct_df = val_df.iloc[correct]
incorrect_df = val_df.iloc[incorrect]

correct_df['preds'] = predictions[correct]
incorrect_df['preds'] = predictions[incorrect]
correct_df['truth'] = truth[correct]
incorrect_df['truth'] = truth[incorrect]

correct_df.to_csv(os.path.join(os.getcwd(), 'outputs', 'correct.csv'), index=False)
incorrect_df.to_csv(os.path.join(os.getcwd(), 'outputs', 'incorrect.csv'), index=False)
