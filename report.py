
import os
import glob
import pandas as pd

from training.li_trainer import LI

cwd = os.getcwd()

for model_type in ['conv']:
    #
    model_locs = os.path.join(cwd, 'outputs/*_{}_*'.format(model_type))
    files = glob.iglob(model_locs)
    #
    dropout_config = []
    weight_decay_config = []
    hidden_dim_config = []
    kernel_size_config = []
    train_loss = []
    val_loss = []
    train_score = []
    val_score = []
    param_count = []
    #
    for file in files:
        model_dir = file
        epoch = sorted(glob.glob(file + '/*.pth'))[-1].split('/')[-1].split('.')[0].split('_')[-1]
        #
        li = LI()
        try:
            li.load(model_dir, epoch)
        except RuntimeError:
            pass
        #
        dropout_config += [li.dropout]
        hidden_dim_config += [li.hidden_dim]
        weight_decay_config += [li.weight_decay]
        try:
            kernel_size_config += [li.kernel_size]
        except:
            kernel_size_config += [None]
        #
        train_loss += [li.best_loss_train]
        val_loss += [li.best_loss]
        #
        train_score += [li.best_score_train]
        val_score += [li.best_score]
        #
        param_count += [sum(p.numel() for p in li.model.parameters() if p.requires_grad is True)]
    #
    df = pd.DataFrame({'dropout': dropout_config,
                       'weight_decay': weight_decay_config,
                       'hidden_dim': hidden_dim_config,
                       'kernel_size': kernel_size_config,
                       'train_loss': train_loss,
                       'val_loss': val_loss,
                       'train_score': train_score,
                       'val_score': val_score,
                       'param_count': param_count})
    #
    df.to_csv(os.path.join(cwd, 'outputs','{}_report.csv'.format(model_type)),
              index=False)
