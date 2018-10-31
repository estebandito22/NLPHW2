
import os
import json
from collections import defaultdict
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

train_dataset = NLIDataset(
    data['prop_snli.train'], data['hyp_snli.train'], data['labels_snli.train'],
    data['max_sent_len_snli.train'])
val_dataset = NLIDataset(
    data['prop_snli.val'], data['hyp_snli.val'], data['labels_snli.val'],
    data['max_sent_len_snli.train'])

save_dir = os.path.join(os.getcwd(), 'outputs')


for dropout in [0, 0.025, 0.05, 0.1, 0.2]:
    for hidden_dim in [32, 64, 128, 256, 512]:
        for kernel_size in [3, 5]:
# for weight_decay in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
#     for hidden_dim in [4, 8, 16, 32, 64, 128, 256]:
#         for kernel_size in [3]:
            li = LI(word_embdim=300, word_embeddings=data['emb_snli.train'],
                    enc_type='conv', hidden_dim=hidden_dim, dropout=dropout,
                    num_layers=1, kernel_size=kernel_size,
                    max_sent_len=data['max_sent_len_snli.train'],
                    vocab_size=50000, batch_size=64, lr=0.001, num_epochs=20)

            # li = LI(word_embdim=300, word_embeddings=data['emb_snli.train'],
            #         enc_type='rnn', hidden_dim=hidden_dim, dropout=0,
            #         num_layers=1, kernel_size=kernel_size,
            #         max_sent_len=data['max_sent_len_snli.train'],
            #         vocab_size=50000, batch_size=64, lr=0.01, num_epochs=10,
            #         weight_decay=weight_decay)

            li.fit(train_dataset, val_dataset, save_dir)


3/5*np.log(3) + 2/5*np.log(2)

np.log(3) + np.log(2)
