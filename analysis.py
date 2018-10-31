import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


rnn_report = pd.read_csv(os.path.join(os.getcwd(), 'outputs','rnn_report.csv'))
rnn_report = rnn_report[rnn_report['weight_decay'] < 0.001]

rnn_report.sort_values('val_score', ascending=False).head(10)[
    ['hidden_dim','weight_decay','train_loss','val_loss',
     'train_score','val_score']].sort_values(['weight_decay','hidden_dim'])

rnn_report[['hidden_dim','param_count']].drop_duplicates().sort_values('hidden_dim')

rnn_summary = pd.pivot_table(
    rnn_report, index=['hidden_dim'],
    columns=['weight_decay'],
    values=['train_loss', 'val_loss'])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
hidden_dims = sorted(rnn_report['weight_decay'].unique())
for i in range(len(hidden_dims)):
    cur_ax = ax.flat[i]
    rnn_summary[[('train_loss', hidden_dims[i]), ('val_loss',hidden_dims[i])]].plot(ax=cur_ax)
    cur_ax.set_ylim([0.4,1.2])
    plt.suptitle("Losses: Over Weight Decay by Hidden Dimension", y=0.98)
    cur_ax.set_ylabel("Loss")
    cur_ax.set_title("Weight Decay: {}".format(hidden_dims[i]))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(os.getcwd(), 'outputs', 'rnn_losses.png'))

rnn_summary = pd.pivot_table(
    rnn_report[rnn_report['hidden_dim']<256], index=['weight_decay'],
    columns=['hidden_dim'],
    values=['train_loss', 'val_loss'])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
hidden_dims = sorted(rnn_report[rnn_report['hidden_dim']<256]['hidden_dim'].unique())
for i in range(len(hidden_dims)):
    cur_ax = ax.flat[i]
    rnn_summary[[('train_loss', hidden_dims[i]), ('val_loss',hidden_dims[i])]].plot(ax=cur_ax)
    cur_ax.set_ylim([0.4,1.2])
    plt.suptitle("Losses: Over Hidden Dimension by Weight Decay", y=0.98)
    cur_ax.set_ylabel("Loss")
    cur_ax.set_title("Hidden Dimension: {}".format(hidden_dims[i]))
    cur_ax.set_xscale('log')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(os.getcwd(), 'outputs', 'rnn_losses2.png'))

rnn_summary = pd.pivot_table(
    rnn_report, index=['hidden_dim'],
    columns=['weight_decay'],
    values=['train_score', 'val_score'])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
hidden_dims = sorted(rnn_report['weight_decay'].unique())
for i in range(len(hidden_dims)):
    cur_ax = ax.flat[i]
    rnn_summary[[('train_score', hidden_dims[i]), ('val_score',hidden_dims[i])]].plot(ax=cur_ax)
    cur_ax.set_ylim([0.3,1])
    plt.suptitle("Accuracies: Over Dropout by Hidden Dimension", y=0.98)
    cur_ax.set_ylabel("Accuracy")
    cur_ax.set_title("Weight Decay: {}".format(hidden_dims[i]))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(os.getcwd(), 'outputs', 'rnn_acc.png'))

rnn_summary = pd.pivot_table(
    rnn_report[rnn_report['hidden_dim']<256], index=['weight_decay'],
    columns=['hidden_dim'],
    values=['train_score', 'val_score'])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
hidden_dims = sorted(rnn_report[rnn_report['hidden_dim']<256]['hidden_dim'].unique())
for i in range(len(hidden_dims)):
    cur_ax = ax.flat[i]
    rnn_summary[[('train_score', hidden_dims[i]), ('val_score',hidden_dims[i])]].plot(ax=cur_ax)
    cur_ax.set_ylim([0.4,1.0])
    plt.suptitle("Accuracy: Over Hidden Dimension by Weight Decay", y=0.98)
    cur_ax.set_ylabel("Accuracy")
    cur_ax.set_title("Hidden Dimension: {}".format(hidden_dims[i]))
    cur_ax.set_xscale('log')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(os.getcwd(), 'outputs', 'rnn_acc2.png'))

######################## 2d heatmaps ###########################

fig, ax = plt.subplots(figsize=(12,8))
val_scores = pd.pivot_table(rnn_report, columns=['hidden_dim'],
                            index=['dropout'], values=['train_score'])['train_score']
heatmap = ax.pcolor(val_scores, cmap='RdBu')
plt.yticks(np.arange(0.5, len(val_scores.index), 1), val_scores.index)
plt.xticks(np.arange(0.5, len(val_scores.columns), 1), val_scores.columns)
plt.title('Train Scores: Dropout by Hidden Dimension')
plt.xlabel('Hidden Dimension')
plt.ylabel('Dropout Rate')
for y in range(val_scores.shape[0]):
    for x in range(val_scores.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % val_scores.iloc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.colorbar(heatmap)


fig, ax = plt.subplots(figsize=(12,8))
val_scores = pd.pivot_table(rnn_report, columns=['hidden_dim'],
                            index=['dropout'], values=['val_score'])['val_score']
heatmap = ax.pcolor(val_scores, cmap='RdBu')
plt.yticks(np.arange(0.5, len(val_scores.index), 1), val_scores.index)
plt.xticks(np.arange(0.5, len(val_scores.columns), 1), val_scores.columns)
plt.title('Validation Scores: Dropout by Hidden Dimension')
plt.xlabel('Hidden Dimension')
plt.ylabel('Dropout Rate')
for y in range(val_scores.shape[0]):
    for x in range(val_scores.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % val_scores.iloc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.colorbar(heatmap)

fig, ax = plt.subplots(figsize=(12,8))
val_scores = pd.pivot_table(rnn_report, columns=['hidden_dim'],
                            index=['dropout'], values=['train_loss'])['train_loss']
heatmap = ax.pcolor(val_scores, cmap='RdBu')
plt.yticks(np.arange(0.5, len(val_scores.index), 1), val_scores.index)
plt.xticks(np.arange(0.5, len(val_scores.columns), 1), val_scores.columns)
plt.title('Train Losses: Dropout by Hidden Dimension')
plt.xlabel('Hidden Dimension')
plt.ylabel('Dropout Rate')
for y in range(val_scores.shape[0]):
    for x in range(val_scores.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % val_scores.iloc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.colorbar(heatmap)

fig, ax = plt.subplots(figsize=(12,8))
val_scores = pd.pivot_table(rnn_report, columns=['hidden_dim'],
                            index=['dropout'], values=['val_loss'])['val_loss']
heatmap = ax.pcolor(val_scores, cmap='RdBu')
plt.yticks(np.arange(0.5, len(val_scores.index), 1), val_scores.index)
plt.xticks(np.arange(0.5, len(val_scores.columns), 1), val_scores.columns)
plt.title('Validation Losses: Dropout by Hidden Dimension')
plt.xlabel('Hidden Dimension')
plt.ylabel('Dropout Rate')
for y in range(val_scores.shape[0]):
    for x in range(val_scores.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % val_scores.iloc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.colorbar(heatmap)

#################### Conv ######################


conv_report = pd.read_csv(os.path.join(os.getcwd(), 'outputs','conv_report.csv'))
# conv_report = conv_report[conv_report['hidden_dim'] < 1024]
conv_report['hidden_dim'] *= 2

conv_report.sort_values('val_score', ascending=False).head(10)[
    ['hidden_dim','dropout','kernel_size','train_loss','val_loss',
     'train_score','val_score']].sort_values(['kernel_size','hidden_dim','dropout'])

conv_report[['hidden_dim', 'kernel_size', 'param_count']].drop_duplicates().sort_values(['hidden_dim','kernel_size'])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
hidden_dims = sorted(conv_report['dropout'].unique())
for k in [3,5]:
    conv_summary = pd.pivot_table(
        conv_report[conv_report['kernel_size']==k], index=['hidden_dim'],
        columns=['dropout'],
        values=['train_loss', 'val_loss'])
    for i in range(len(hidden_dims)):
        cur_ax = ax.flat[i]
        df = conv_summary[[('train_loss', hidden_dims[i]), ('val_loss',hidden_dims[i])]]
        df = pd.concat([df], keys=['Size: {}'.format(k)], names=['Kernel'], axis=1)
        df.plot(ax=cur_ax)
        cur_ax.set_ylim([0.3,1.2])
        plt.suptitle("Losses: Over Dropout by Hidden Dimension", y=0.98)
        cur_ax.set_ylabel("Loss")
        cur_ax.set_title("Dropout: {}".format(hidden_dims[i]))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(os.getcwd(), 'outputs', 'conv_loss.png'))


fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
hidden_dims = sorted(conv_report['hidden_dim'].unique())
for k in [3,5]:
    conv_summary = pd.pivot_table(
        conv_report[conv_report['kernel_size']==k], index=['dropout'],
        columns=['hidden_dim'],
        values=['train_loss', 'val_loss'])
    for i in range(len(hidden_dims)):
        cur_ax = ax.flat[i]
        df = conv_summary[[('train_loss', hidden_dims[i]), ('val_loss',hidden_dims[i])]]
        df = pd.concat([df], keys=['Size: {}'.format(k)], names=['Kernel'], axis=1)
        df.plot(ax=cur_ax)
        cur_ax.set_ylim([0.3,1.2])
        plt.suptitle("Losses: Over Hidden Dimension by Dropout", y=0.98)
        cur_ax.set_ylabel("Loss")
        cur_ax.set_title("Hidden Dimension: {}".format(hidden_dims[i]))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(os.getcwd(), 'outputs', 'conv_loss2.png'))

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
hidden_dims = sorted(conv_report['dropout'].unique())
for k in [3,5]:
    conv_summary = pd.pivot_table(
        conv_report[conv_report['kernel_size']==k], index=['hidden_dim'],
        columns=['dropout'],
        values=['train_score', 'val_score'])
    for i in range(len(hidden_dims)):
        cur_ax = ax.flat[i]
        df = conv_summary[[('train_score', hidden_dims[i]), ('val_score',hidden_dims[i])]]
        df = pd.concat([df], keys=['Size: {}'.format(k)], names=['Kernel'], axis=1)
        df.plot(ax=cur_ax)
        cur_ax.set_ylim([0.6,1])
        plt.suptitle("Accuracies: Over Dropout by Hidden Dimension", y=0.98)
        cur_ax.set_ylabel("Accuracy")
        cur_ax.set_title("Dropout: {}".format(hidden_dims[i]))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(os.getcwd(), 'outputs', 'conv_acc.png'))

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
hidden_dims = sorted(conv_report['hidden_dim'].unique())
for k in [3,5]:
    conv_summary = pd.pivot_table(
        conv_report[conv_report['kernel_size']==k], index=['dropout'],
        columns=['hidden_dim'],
        values=['train_score', 'val_score'])
    for i in range(len(hidden_dims)):
        cur_ax = ax.flat[i]
        df = conv_summary[[('train_score', hidden_dims[i]), ('val_score',hidden_dims[i])]]
        df = pd.concat([df], keys=['Size: {}'.format(k)], names=['Kernel'], axis=1)
        df.plot(ax=cur_ax)
        cur_ax.set_ylim([0.6,1])
        plt.suptitle("Accuracy: Over Hidden Dimension by Dropout", y=0.98)
        cur_ax.set_ylabel("Accuracy")
        cur_ax.set_title("Hidden Dimension: {}".format(hidden_dims[i]))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(os.getcwd(), 'outputs', 'conv_acc2.png'))

#################### 2d heatmaps ########################

fig, ax = plt.subplots(figsize=(12,8))
val_scores = pd.pivot_table(conv_report, columns=['kernel_size', 'hidden_dim'],
                            index=['dropout'], values=['train_score'])['train_score']
heatmap = ax.pcolor(val_scores, cmap='RdBu')
plt.yticks(np.arange(0.5, len(val_scores.index), 1), val_scores.index)
plt.xticks(np.arange(0.5, len(val_scores.columns), 1), val_scores.columns)
plt.title('Train Scores: Dropout by (Kernel Size, Hidden Dimension)')
plt.xlabel('(Kernel Size, Hidden Dimension)')
plt.ylabel('Dropout Rate')
for y in range(val_scores.shape[0]):
    for x in range(val_scores.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % val_scores.iloc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.colorbar(heatmap)

fig, ax = plt.subplots(figsize=(12,8))
val_scores = pd.pivot_table(conv_report, columns=['kernel_size', 'hidden_dim'],
                            index=['dropout'], values=['val_score'])['val_score']
heatmap = ax.pcolor(val_scores, cmap='RdBu')
plt.yticks(np.arange(0.5, len(val_scores.index), 1), val_scores.index)
plt.xticks(np.arange(0.5, len(val_scores.columns), 1), val_scores.columns)
plt.title('Validation Scores: Dropout by (Kernel Size, Hidden Dimension)')
plt.xlabel('(Kernel Size, Hidden Dimension)')
plt.ylabel('Dropout Rate')
for y in range(val_scores.shape[0]):
    for x in range(val_scores.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % val_scores.iloc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.colorbar(heatmap)


fig, ax = plt.subplots(figsize=(12,8))
val_scores = pd.pivot_table(conv_report, columns=['kernel_size', 'hidden_dim'],
                            index=['dropout'], values=['train_loss'])['train_loss']
heatmap = ax.pcolor(val_scores, cmap='RdBu')
plt.yticks(np.arange(0.5, len(val_scores.index), 1), val_scores.index)
plt.xticks(np.arange(0.5, len(val_scores.columns), 1), val_scores.columns)
plt.title('Train Losses: Dropout by (Kernel Size, Hidden Dimension)')
plt.xlabel('(Kernel Size, Hidden Dimension)')
plt.ylabel('Dropout Rate')
for y in range(val_scores.shape[0]):
    for x in range(val_scores.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % val_scores.iloc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.colorbar(heatmap)

fig, ax = plt.subplots(figsize=(12,8))
val_scores = pd.pivot_table(conv_report, columns=['kernel_size', 'hidden_dim'],
                            index=['dropout'], values=['val_loss'])['val_loss']
heatmap = ax.pcolor(val_scores, cmap='RdBu')
plt.yticks(np.arange(0.5, len(val_scores.index), 1), val_scores.index)
plt.xticks(np.arange(0.5, len(val_scores.columns), 1), val_scores.columns)
plt.title('Validation Losses: Dropout by (Kernel Size, Hidden Dimension)')
plt.xlabel('(Kernel Size, Hidden Dimension)')
plt.ylabel('Dropout Rate')
for y in range(val_scores.shape[0]):
    for x in range(val_scores.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % val_scores.iloc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.colorbar(heatmap)


####################### MNLI ########################

mnli_report = pd.read_csv(os.path.join(os.getcwd(), 'outputs','genre_report.csv'), index_col=0)
mnli_finetune_report = pd.read_csv(os.path.join(os.getcwd(), 'outputs','finetune','rnn_genre_report.csv'))

mnli_report[['genre','conv_accuracy', 'rnn_accuracy']]

mnli_finetune_report[['genre', 'train_loss','val_loss','train_score','val_score']]

############## Correct/Incorrect ##################

correct = pd.read_csv(os.path.join(os.getcwd(), 'outputs', 'correct.csv'))
incorrect = pd.read_csv(os.path.join(os.getcwd(), 'outputs', 'incorrect.csv'))

for i, row in correct.iloc[[0,4,16]].iterrows():
    print("Premise:\n{}\n\nHypothesis:\n\n{}\nLabel: {}\tTruth: {}\tPrediction: {}\n{}".format(
        row['sentence1'], row['sentence2'], row['label'], row['truth'], row['preds'],'-'*70))

for i, row in incorrect.iloc[[4,8,7]].iterrows():
    print("Premise:\n{}\n\nHypothesis:\n\n{}\nLabel: {}\tTruth: {}\tPrediction: {}\n{}".format(
        row['sentence1'], row['sentence2'], row['label'], row['truth'], row['preds'],'-'*70))
