"""Class to train encoder decoder neural machine translation network."""

import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.metrics import accuracy_score

from models.languageinference import LanguageInference


class LI(object):

    """Class to train LI network."""

    def __init__(self, word_embdim=300, word_embeddings=None, enc_type='rnn',
                 hidden_dim=256, dropout=0, num_layers=1, kernel_size=3,
                 max_sent_len=78, vocab_size=20000, batch_size=64, lr=0.01,
                 num_epochs=100, weight_decay=0):
        """Initialize LI."""
        self.word_embdim = word_embdim
        self.word_embeddings = word_embeddings
        self.enc_type = enc_type
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.max_sent_len = max_sent_len
        self.kernel_size = kernel_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        # # Dataset attributes
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Model attributes
        self.model = None
        self.optimizer = None
        self.loss_func = None
        self.dict_args = None
        self.nn_epoch = None

        # Save load attributes
        self.save_dir = None
        self.model_dir = None

        # Performance attributes
        self.train_losses = []
        self.val_losses = []
        self.best_score = 0
        self.best_score_train = 0
        self.best_loss = float('inf')
        self.best_loss_train = float('inf')

        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

    def _init_nn(self):
        """Initialize the nn model for training."""
        self.dict_args = {'word_embdim': self.word_embdim,
                          'word_embeddings': self.word_embeddings,
                          'hidden_dim': self.hidden_dim,
                          'enc_type': self.enc_type,
                          'dropout': self.dropout,
                          'vocab_size': self.vocab_size,
                          'batch_size': self.batch_size,
                          'num_layers': self.num_layers,
                          'max_sent_len': self.max_sent_len,
                          'kernel_size': self.kernel_size}
        self.model = LanguageInference(self.dict_args)

        self.loss_func = nn.NLLLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), self.lr, weight_decay=self.weight_decay)

        if self.USE_CUDA:
            self.model = self.model.cuda()
            self.loss_func = self.loss_func.cuda()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in tqdm(loader):
            # prepare sample
            len_p_sent = batch_samples['len_p_sent']
            len_h_sent = batch_samples['len_h_sent']
            # max_sent_len * batch_size
            p_sent = batch_samples['p_sent']
            h_sent = batch_samples['h_sent']
            # batch_size * 1
            y = batch_samples['y']

            if self.USE_CUDA:
                p_sent = p_sent.cuda()
                h_sent = h_sent.cuda()
                y = y.cuda()

            # forward pass
            self.model.zero_grad()
            log_probs = self.model(p_sent, len_p_sent, h_sent, len_h_sent)

            # backward pass
            loss = self.loss_func(log_probs, y)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += log_probs.size()[0]
            train_loss += loss.item() * log_probs.size()[0]

        train_loss /= samples_processed

        return samples_processed, train_loss

    def _eval_epoch(self, loader):
        """Eval epoch."""
        self.model.eval()
        eval_loss = 0
        samples_processed = 0

        with torch.no_grad():
            for batch_samples in tqdm(loader):
                # prepare sample
                len_p_sent = batch_samples['len_p_sent']
                len_h_sent = batch_samples['len_h_sent']
                # max_sent_len * batch_size
                p_sent = batch_samples['p_sent']
                h_sent = batch_samples['h_sent']
                # batch_size * 1
                y = batch_samples['y']

                if self.USE_CUDA:
                    p_sent = p_sent.cuda()
                    h_sent = h_sent.cuda()
                    y = y.cuda()

                # forward pass
                self.model.zero_grad()
                log_probs = self.model(p_sent, len_p_sent, h_sent, len_h_sent)

                # compute eval loss
                loss = self.loss_func(log_probs, y)

                samples_processed += log_probs.size()[0]
                eval_loss += loss.item() * log_probs.size()[0]

            eval_loss /= samples_processed

        return samples_processed, eval_loss

    def fit(self, train_dataset, val_dataset, save_dir, warm_start=False):
        """
        Train the NN model.

        Args
            train_dataset: pytorch dataset for training data
            val_dataset: pytorch dataset for validation data
            save_dir: directory to save nn_model
        """
        # Print settings to output file
        print("Word Embedding Dim {}\n\
               Word Embeddings {}\n\
               Encoder Type {}\n\
               Hidden Dim {}\n\
               Dropout {}\n\
               Num Layers {}\n\
               Kernel Size {}\n\
               Vocabulary Size {}\n\
               Batch Size {}\n\
               Learning Rate {}\n\
               Save Dir: {}".format(
                   self.word_embdim, bool(self.word_embeddings), self.enc_type,
                   self.hidden_dim, self.dropout, self.num_layers,
                   self.kernel_size, self.vocab_size, self.batch_size, self.lr,
                   save_dir))

        self.model_dir = save_dir
        self.train_data = train_dataset
        self.val_data = val_dataset

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=4)
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=True,
            num_workers=4)

        if not warm_start:
            self._init_nn()

        # init training variables
        train_loss = 0
        train_score = 0
        samples_processed = 0

        # train loop
        for epoch in range(self.num_epochs + 1):
            self.nn_epoch = epoch
            if epoch > 0:
                print("Initializing train epoch...")
                sp, train_loss = self._train_epoch(train_loader)
                samples_processed += sp

                self.train_losses += [train_loss]

                # report
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss:{}".format(
                    epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss))

            if epoch % 1 == 0:
                # compute loss
                print("Initializing val epoch...")
                _, val_loss = self._eval_epoch(val_loader)

                self.val_losses += [val_loss]
                val_score = self.score(val_loader)

                if val_score > self.best_score:
                    self.best_score = val_score
                    self.best_loss = val_loss

                    train_score = self.score(train_loader)
                    self.best_score_train = train_score
                    self.best_loss_train = train_loss

                    self.save(save_dir)
                else:
                    train_score = None

                # report
                print("""Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss:
                {}\tTrain Score: {}\tValidation Loss: {}\tValidation Score: {}\
                """.format(epoch, self.num_epochs, samples_processed,
                           len(self.train_data)*self.num_epochs, train_loss,
                           train_score, val_loss, val_score))

    def predict(self, loader):
        """Predict input."""
        self.model.eval()
        log_proba = []
        truth = []

        with torch.no_grad():
            for batch_samples in tqdm(loader):
                # prepare sample
                len_p_sent = batch_samples['len_p_sent']
                len_h_sent = batch_samples['len_h_sent']
                # batch_size * max_sent_len
                p_sent = batch_samples['p_sent']
                h_sent = batch_samples['h_sent']
                # batch_size * 1
                y = batch_samples['y']

                if self.USE_CUDA:
                    p_sent = p_sent.cuda()
                    h_sent = h_sent.cuda()
                    y = y.cuda()

                # forward pass
                self.model.zero_grad()
                log_probs = self.model(p_sent, len_p_sent, h_sent, len_h_sent)
                log_proba += [log_probs]
                truth += [y]

            log_proba = torch.cat(log_proba, dim=0)
            truth = torch.cat(truth, dim=0)

        return log_proba, truth

    def score(self, loader):
        """Score model."""
        log_proba, truth = self.predict(loader)
        predictions = torch.argmax(log_proba, dim=1)
        return accuracy_score(truth, predictions)

    def save(self, models_dir=None):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = "LI_wed_{}_we_{}_et_{}_hd_{}_ks_{}_do_{}_vs_{}_nl_{}_lr_{}_wd_{}".\
                format(self.word_embdim, bool(self.word_embeddings),
                       self.enc_type, self.hidden_dim, self.kernel_size,
                       self.dropout, self.vocab_size, self.num_layers, self.lr,
                       self.weight_decay)

            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        """
        Load a previously trained model.

        Args
            model_dir: directory where models are saved.
            epoch: epoch of model to load.
        """
        epoch_file = "epoch_"+str(epoch)+".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_dict)
            else:
                checkpoint = torch.load(model_dict, map_location='cpu')

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self._init_nn()
        self.model.load_state_dict(checkpoint['state_dict'])
